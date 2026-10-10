"""Attacks that need a local model or an external provider. All are optional."""

from __future__ import annotations

import random
import re
import threading
from typing import Any, Literal

import torch
from pydantic import Field

from backend.attacks.base import Attack, AttackConfig
from backend.core.config import get_settings
from backend.core.errors import ProviderUnavailable
from backend.providers.registry import get_provider

_lock = threading.Lock()
_pipes: dict[str, Any] = {}


def _hf_load(kind: str, name: str):
    with _lock:
        key = f"{kind}:{name}"
        if key not in _pipes:
            s = get_settings()
            try:
                if kind == "mlm":
                    from transformers import AutoModelForMaskedLM, AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(name, local_files_only=s.hf_offline)
                    model = AutoModelForMaskedLM.from_pretrained(name, local_files_only=s.hf_offline).eval()
                else:
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(name, local_files_only=s.hf_offline)
                    model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=s.hf_offline).eval()
            except OSError as exc:
                raise ProviderUnavailable(f"Could not load model {name!r}", details={"model": name}) from exc
            _pipes[key] = (tok, model)
        return _pipes[key]


# ----------------------------------------------------------------------------- lexical
class LexicalSubstitutionConfig(AttackConfig):
    rate: float = Field(default=0.2, ge=0.0, le=1.0)
    model_name: str = "distilroberta-base"
    top_k: int = Field(default=5, ge=1, le=50)


_WORD_RE = re.compile(r"^[A-Za-z]{3,}$")


class LexicalSubstitution(Attack):
    """Replace a fraction of words with masked-LM suggestions (one mask at a time)."""

    name = "lexical_substitution"
    attack_type = "edit"
    config_cls = LexicalSubstitutionConfig
    requires = ["hf:distilroberta-base"]

    @torch.inference_mode()
    def _apply(self, text: str, seed: int):
        tok, model = _hf_load("mlm", self.config.model_name)
        rng = random.Random(seed)
        words = text.split()
        candidates = [i for i, w in enumerate(words) if _WORD_RE.match(w.strip(".,;:!?\"'()"))]
        n_target = round(len(candidates) * self.config.rate)
        targets = sorted(rng.sample(candidates, n_target)) if n_target else []
        replaced = 0
        for i in targets:
            core = words[i].strip(".,;:!?\"'()")
            prefix = words[i][: words[i].find(core)]
            suffix = words[i][words[i].find(core) + len(core) :]
            masked = words.copy()
            masked[i] = prefix + tok.mask_token + suffix
            enc = tok(" ".join(masked), return_tensors="pt", truncation=True, max_length=512)
            pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()
            if len(pos) == 0:
                continue
            logits = model(**enc).logits[0, pos[0, 0]]
            for cand_id in logits.topk(self.config.top_k).indices.tolist():
                cand = tok.decode([cand_id]).strip()
                if cand.isalpha() and cand.lower() != core.lower():
                    if core[0].isupper():
                        cand = cand[0].upper() + cand[1:]
                    words[i] = prefix + cand + suffix
                    replaced += 1
                    break
        return " ".join(words), {"replaced": replaced, "targeted": len(targets)}


# ----------------------------------------------------------------------------- translation
class BackTranslationConfig(AttackConfig):
    pivot_language: str = Field(default="de", pattern=r"^[a-z]{2}$")
    backend: Literal["marian", "provider"] = "marian"
    provider: str = "gemini"
    max_length: int = Field(default=512, ge=16, le=1024)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class BackTranslation(Attack):
    """English -> pivot -> English, via MarianMT (local) or a configured provider."""

    name = "back_translation"
    attack_type = "translation"
    config_cls = BackTranslationConfig
    deterministic = True  # greedy decoding for Marian; provider output is not deterministic
    requires = ["hf:Helsinki-NLP/opus-mt-en-<pivot>", "hf:Helsinki-NLP/opus-mt-<pivot>-en"]

    @torch.inference_mode()
    def _marian(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        tok, model = _hf_load("seq2seq", f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        out: list[str] = []
        for i in range(0, len(sentences), 8):
            batch = sentences[i : i + 8]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length)
            gen = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=self.config.max_length)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def _apply(self, text: str, seed: int):
        p = self.config.pivot_language
        if self.config.backend == "marian":
            sents = _split_sentences(text)
            pivot = self._marian(sents, "en", p)
            back = self._marian(pivot, p, "en")
            return " ".join(back), {"pivot": p, "backend": "marian", "n_sentences": len(sents)}
        prov = get_provider(self.config.provider)
        s = get_settings()
        kw = dict(
            temperature=0.0,
            timeout_s=s.provider_timeout_s,
            max_retries=s.provider_max_retries,
            backoff_base_s=s.provider_backoff_base_s,
            max_output_tokens=1024,
        )
        mid = prov.complete(
            f"Translate the following text into language code '{p}'. Output only the translation.\n\n{text}", **kw
        )
        back = prov.complete(
            f"Translate the following text into English. Output only the translation.\n\n{mid.text}", **kw
        )
        return back.text, {"pivot": p, "backend": prov.name, "model": back.model}


# ----------------------------------------------------------------------------- paraphrase
PARAPHRASE_PROMPT = (
    "Paraphrase the following passage. Preserve its meaning and approximate length, but change "
    "the wording and sentence structure. Output only the paraphrased passage.\n\nPassage:\n{text}"
)


class ParaphraseConfig(AttackConfig):
    provider: str = Field(default="gemini", description="gemini | hf_local")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=16, le=4096)


class Paraphrase(Attack):
    name = "paraphrase"
    attack_type = "paraphrase"
    config_cls = ParaphraseConfig
    deterministic = False
    requires = ["provider:<configured>"]

    def _apply(self, text: str, seed: int):
        prov = get_provider(self.config.provider)
        s = get_settings()
        res = prov.complete(
            PARAPHRASE_PROMPT.format(text=text),
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            timeout_s=s.provider_timeout_s,
            max_retries=s.provider_max_retries,
            backoff_base_s=s.provider_backoff_base_s,
        )
        return res.text, {"provider": res.provider, "model": res.model, "attempts": res.attempts}
