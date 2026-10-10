"""Local Hugging Face causal language models (CPU-friendly), with process-level caching."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from backend.core.config import get_settings
from backend.core.errors import ProviderUnavailable
from backend.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class GenOutput:
    text: str
    n_new_tokens: int
    latency_ms: float


class HFCausalLM:
    """Thin wrapper around a HF causal LM + tokenizer.

    Sampling is configured explicitly: HF's own warpers are disabled (temperature=1,
    top_k=0, top_p=1) and the caller passes an ordered processor list, so a watermark
    bias is applied *before* temperature / nucleus filtering, as in Kirchenbauer et al.
    """

    def __init__(self, model: Any, tokenizer: Any, name: str, revision: str | None = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.name = name
        self.revision = revision
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_output_embeddings().weight.shape[0])

    @classmethod
    def load(cls, name: str, revision: str | None = None) -> HFCausalLM:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailable("transformers is not installed") from exc
        s = get_settings()
        token = s.hf_token.get_secret_value() if s.hf_token else None
        if s.torch_threads:
            torch.set_num_threads(s.torch_threads)
        t0 = time.perf_counter()
        try:
            tok = AutoTokenizer.from_pretrained(name, revision=revision, token=token, local_files_only=s.hf_offline)
            model = AutoModelForCausalLM.from_pretrained(
                name, revision=revision, token=token, local_files_only=s.hf_offline, dtype=torch.float32
            )
        except OSError as exc:
            raise ProviderUnavailable(
                f"Could not load Hugging Face model {name!r}. If HF_OFFLINE=true, download it first "
                f"(scripts/download_models.py).",
                details={"model": name},
            ) from exc
        log.info("loaded model", extra={"model": name, "load_s": round(time.perf_counter() - t0, 2)})
        return cls(model, tok, name, revision)

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 200,
        min_new_tokens: int = 0,
        seed: int = 0,
        processors: Sequence[Any] = (),
        batch_size: int = 8,
        max_prompt_tokens: int = 256,
    ) -> list[GenOutput]:
        from transformers import LogitsProcessorList

        outs: list[GenOutput] = []
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens
            )
            torch.manual_seed(seed + start)
            t0 = time.perf_counter()
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                logits_processor=LogitsProcessorList(list(processors)),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            per_item_ms = (time.perf_counter() - t0) * 1000 / len(batch)
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            for row in new_tokens:
                ids = [int(t) for t in row if int(t) != self.tokenizer.pad_token_id]
                # Trim at first EOS (pad==eos for many tokenizers, already removed above).
                if self.tokenizer.eos_token_id in ids:
                    ids = ids[: ids.index(self.tokenizer.eos_token_id)]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                outs.append(GenOutput(text=text, n_new_tokens=len(ids), latency_ms=per_item_ms))
        return outs

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def token_stats(self, text: str, max_tokens: int = 512) -> dict[str, Any]:
        """Per-token log-probabilities, ranks and entropies of `text` under this model."""
        ids = self.encode(text)[:max_tokens]
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        input_ids = torch.tensor([[bos, *ids]])
        logits = self.model(input_ids).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        target = input_ids[0, 1:]
        tok_logp = logp.gather(1, target[:, None]).squeeze(1)
        ranks = (logp > tok_logp[:, None]).sum(dim=1) + 1
        entropy = -(logp.exp() * logp).sum(dim=1)
        return {
            "n_tokens": len(ids),
            "logp": tok_logp.numpy(),
            "rank": ranks.numpy(),
            "entropy": entropy.numpy(),
            "logits": logits,  # used by Binoculars-style cross-perplexity
            "input_ids": input_ids,
        }


_cache: dict[tuple[str, str | None], HFCausalLM] = {}
_lock = threading.Lock()


def get_lm(name: str, revision: str | None = None) -> HFCausalLM:
    key = (name, revision)
    with _lock:
        if key not in _cache:
            _cache[key] = HFCausalLM.load(name, revision)
        return _cache[key]


def register_lm(lm: HFCausalLM) -> None:
    """Inject a pre-built model (tests, offline toy models)."""
    with _lock:
        _cache[(lm.name, lm.revision)] = lm


_tok_cache: dict[str, Any] = {}


def get_tokenizer(name: str) -> Any:
    """Tokenizer only (detection does not need model weights)."""
    with _lock:
        for (n, _), lm in _cache.items():
            if n == name:
                return lm.tokenizer
        if name not in _tok_cache:
            from transformers import AutoTokenizer

            s = get_settings()
            token = s.hf_token.get_secret_value() if s.hf_token else None
            try:
                _tok_cache[name] = AutoTokenizer.from_pretrained(name, token=token, local_files_only=s.hf_offline)
            except OSError as exc:
                raise ProviderUnavailable(f"Could not load tokenizer {name!r}", details={"model": name}) from exc
        return _tok_cache[name]
