"""Generation-time "green list" watermarks.

KGW  — Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
       Green list for position t is seeded by the previous `context_width` tokens.
Unigram — Zhao et al., "Provable Robust Watermarking for AI-Generated Text", ICLR 2024.
       A single fixed green list (context_width = 0), which trades key-recoverability
       for robustness to local edits.

Embedding: add `delta` to the logits of green tokens before sampling.
Detection: count green tokens among the T scored tokens of the re-tokenised text and
compute the one-proportion z-score  z = (|s|_G - gamma*T) / sqrt(T*gamma*(1-gamma)).

Assumptions & limitations
- Detection requires the same tokenizer and key as embedding (not the model weights).
- Low-entropy text (code, lists, memorised passages) carries little signal; delta
  cannot move the distribution when one token dominates.
- The z-test assumes token-level independence under H0; repeated n-grams violate it,
  so `ignore_repeated_ngrams` counts each (context, token) pair once (Kirchenbauer et
  al., "On the Reliability of Watermarks", 2023).
- Short texts (< min_tokens scored) return decision "insufficient", never "negative".
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from pydantic import Field, model_validator
from scipy.stats import norm

from backend.core.config import get_settings
from backend.provenance.signing import key_fingerprint
from backend.providers.hf_local import HFCausalLM, get_lm, get_tokenizer
from backend.watermark import prf
from backend.watermark.base import DetectResult, EmbedResult, WatermarkConfig, WatermarkMethod


class GreenListConfig(WatermarkConfig):
    model_name: str = Field(default_factory=lambda: get_settings().default_generator_model)
    gamma: float = Field(default=0.25, gt=0.0, lt=1.0, description="Green-list fraction")
    delta: float = Field(default=2.0, ge=0.0, le=20.0, description="Logit bias for green tokens")
    context_width: int = Field(default=1, ge=0, le=8, description="h: preceding tokens used to seed")
    z_threshold: float = Field(default=4.0, gt=0.0)
    min_tokens: int = Field(default=16, ge=1)
    ignore_repeated_ngrams: bool = True
    # Generation parameters (embedding only)
    max_new_tokens: int = Field(default=200, ge=1, le=2048)
    min_new_tokens: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = 0
    batch_size: int = Field(default=8, ge=1, le=64)
    max_prompt_tokens: int = Field(default=256, ge=8)

    @model_validator(mode="after")
    def _check(self):
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens must be <= max_new_tokens")
        return self


class UnigramConfig(GreenListConfig):
    context_width: Literal[0] = 0


class GreenListScheme:
    """Shared green-list logic for embedding and detection."""

    def __init__(self, key: str, gamma: float, context_width: int, cache_size: int = 512):
        self.key_int = prf.key_to_int(key)
        self.gamma = gamma
        self.h = context_width
        self._mask_cache: OrderedDict[tuple[tuple[int, ...], int], np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def seed(self, context: tuple[int, ...]) -> np.uint64:
        return prf.context_seed(self.key_int, context[-self.h :] if self.h else ())

    def mask(self, context: tuple[int, ...], vocab_size: int) -> np.ndarray:
        ctx = context[-self.h :] if self.h else ()
        k = (ctx, vocab_size)
        if k in self._mask_cache:
            self._mask_cache.move_to_end(k)
            return self._mask_cache[k]
        m = prf.green_mask(self.seed(ctx), vocab_size, self.gamma)
        self._mask_cache[k] = m
        if len(self._mask_cache) > self._cache_size:
            self._mask_cache.popitem(last=False)
        return m

    def score_ids(self, ids: list[int], ignore_repeated: bool) -> dict[str, Any]:
        green_flags: list[bool] = []
        seen: set[tuple[tuple[int, ...], int]] = set()
        for i in range(self.h, len(ids)):
            ctx = tuple(ids[i - self.h : i]) if self.h else ()
            if ignore_repeated:
                key = (ctx, ids[i])
                if key in seen:
                    continue
                seen.add(key)
            green_flags.append(prf.is_green(self.seed(ctx), ids[i], self.gamma))
        t = len(green_flags)
        g = int(sum(green_flags))
        return {"n_scored": t, "n_green": g, "green_flags": green_flags}

    def z_score(self, n_green: int, t: int) -> float:
        return (n_green - self.gamma * t) / math.sqrt(t * self.gamma * (1 - self.gamma))


class GreenListLogitsProcessor:
    """HF-compatible logits processor: add delta to green-token logits."""

    def __init__(self, scheme: GreenListScheme, delta: float):
        self.scheme = scheme
        self.delta = delta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab = scores.shape[-1]
        bias = torch.zeros_like(scores)
        for b in range(input_ids.shape[0]):
            ctx = tuple(int(t) for t in input_ids[b, -self.scheme.h :]) if self.scheme.h else ()
            bias[b] = torch.from_numpy(self.scheme.mask(ctx, vocab)).to(scores.dtype) * self.delta
        return scores + bias


class _Warper:
    """Temperature then nucleus filtering, applied after the watermark bias."""

    def __init__(self, temperature: float, top_p: float):
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum - probs > self.top_p  # keep the token that crosses the threshold
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        return scores


class GreenListWatermark(WatermarkMethod):
    name: ClassVar[str] = "kgw"
    version: ClassVar[str] = "1.0.0"
    kind: ClassVar = "generation_time"
    config_cls: ClassVar = GreenListConfig
    reference: ClassVar[str] = "Kirchenbauer et al. (2023), arXiv:2301.10226"

    def __init__(self, key: str | None = None, lm: HFCausalLM | None = None):
        self._key = key or get_settings().watermark_secret_key.get_secret_value()
        self._lm = lm
        self._schemes: dict[tuple[float, int], GreenListScheme] = {}

    def _scheme(self, cfg: GreenListConfig) -> GreenListScheme:
        k = (cfg.gamma, cfg.context_width)
        if k not in self._schemes:
            self._schemes[k] = GreenListScheme(self._key, cfg.gamma, cfg.context_width)
        return self._schemes[k]

    def _get_lm(self, cfg: GreenListConfig) -> HFCausalLM:
        if self._lm is not None and self._lm.name == cfg.model_name:
            return self._lm
        return get_lm(cfg.model_name)

    def processors(self, cfg: GreenListConfig) -> list[Any]:
        return [GreenListLogitsProcessor(self._scheme(cfg), cfg.delta), _Warper(cfg.temperature, cfg.top_p)]

    def embed_batch(self, texts: list[str], config: Any = None) -> list[EmbedResult]:
        cfg = self.parse_config(config)
        lm = self._get_lm(cfg)
        outs = lm.generate(
            texts,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            seed=cfg.seed,
            processors=self.processors(cfg),
            batch_size=cfg.batch_size,
            max_prompt_tokens=cfg.max_prompt_tokens,
        )
        meta = {"method": self.name, "config": cfg.model_dump(), "key_fingerprint": key_fingerprint(self._key)}
        return [
            EmbedResult(text=o.text, input_text=p, n_tokens=o.n_new_tokens, latency_ms=o.latency_ms, metadata=meta)
            for p, o in zip(texts, outs, strict=True)
        ]

    def embed(self, text: str, config: Any = None) -> EmbedResult:
        return self.embed_batch([text], config)[0]

    def _tokenizer(self, cfg: GreenListConfig):
        if self._lm is not None and self._lm.name == cfg.model_name:
            return self._lm.tokenizer
        return get_tokenizer(cfg.model_name)

    def detect(self, text: str, config: Any = None) -> DetectResult:
        cfg = self.parse_config(config)
        t0 = time.perf_counter()
        ids = self._tokenizer(cfg)(text, add_special_tokens=False)["input_ids"]
        scheme = self._scheme(cfg)
        s = scheme.score_ids(ids, cfg.ignore_repeated_ngrams)
        t = s["n_scored"]
        details = {"n_tokens_total": len(ids), "n_green": s["n_green"], "gamma": cfg.gamma}
        if t < cfg.min_tokens:
            return DetectResult(
                score=None,
                p_value=None,
                threshold=cfg.z_threshold,
                decision="insufficient",
                n_tokens_scored=t,
                latency_ms=(time.perf_counter() - t0) * 1000,
                details={**details, "reason": f"only {t} scorable tokens (< min_tokens={cfg.min_tokens})"},
            )
        z = scheme.z_score(s["n_green"], t)
        return DetectResult(
            score=z,
            p_value=float(norm.sf(z)),
            threshold=cfg.z_threshold,
            decision="positive" if z >= cfg.z_threshold else "negative",
            n_tokens_scored=t,
            latency_ms=(time.perf_counter() - t0) * 1000,
            details={**details, "green_fraction": s["n_green"] / t},
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "reference": self.reference,
            "key_fingerprint": key_fingerprint(self._key),
            "config_schema": self.config_cls.model_json_schema(),
            "detection_requires": ["tokenizer", "secret key"],
            "limitations": [
                "Requires identical tokenizer and key at detection time.",
                "Weak signal on low-entropy text and on short texts.",
                "Paraphrasing / translation re-samples tokens and erodes the signal.",
                "A detection is statistical evidence of the key's use, not proof of authorship.",
            ],
        }


class UnigramWatermark(GreenListWatermark):
    name: ClassVar[str] = "unigram"
    config_cls: ClassVar = UnigramConfig
    reference: ClassVar[str] = "Zhao et al. (2024), arXiv:2306.17439"
