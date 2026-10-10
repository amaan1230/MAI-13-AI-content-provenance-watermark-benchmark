"""Local HF instruction model exposed through the TextProvider interface (offline-capable)."""

from __future__ import annotations

import time
from typing import ClassVar

import torch

from backend.core.config import get_settings
from backend.core.errors import ProviderTimeout, ProviderUnavailable
from backend.providers.base import ProviderStatus, TextProvider


class LocalHFInstructProvider(TextProvider):
    """Chat-templated local model; used for optional local paraphrasing."""

    name = "hf_local"
    kind = "local"
    default_model: ClassVar[str] = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.default_model

    def status(self) -> ProviderStatus:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return ProviderStatus(self.name, False, self.kind, [], self.model_name, "transformers not installed")
        msg = "Available (model downloaded on first use)"
        if get_settings().hf_offline:
            msg = "Available if the model is present in the local HF cache (HF_OFFLINE=true)"
        return ProviderStatus(self.name, True, self.kind, ["generate", "paraphrase"], self.model_name, msg)

    def _complete(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: float):
        from backend.providers.hf_local import get_lm

        try:
            lm = get_lm(self.model_name)
        except ProviderUnavailable:
            raise
        tok = lm.tokenizer
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        enc = tok(chat, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = lm.model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                top_p=0.95,
                top_k=0,
                max_new_tokens=max_output_tokens,
                pad_token_id=tok.pad_token_id,
                max_time=timeout_s,
            )
        if time.perf_counter() - t0 > timeout_s:
            raise ProviderTimeout("Local generation exceeded timeout", details={"provider": self.name})
        text = tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return text, {"output_tokens": int(out.shape[1] - enc["input_ids"].shape[1])}
