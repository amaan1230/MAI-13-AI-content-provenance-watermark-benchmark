"""Offline, seeded text transformations (no models or network required)."""

from __future__ import annotations

import random
import re
import string
import unicodedata
from typing import Any, Literal

from pydantic import Field

from backend.attacks.base import Attack, AttackConfig


def _words(text: str) -> list[str]:
    return text.split()


class TruncationConfig(AttackConfig):
    keep_fraction: float | None = Field(default=0.5, gt=0.0, le=1.0)
    max_words: int | None = Field(default=None, ge=1)
    keep: Literal["start", "end"] = "start"


class Truncation(Attack):
    name = "truncation"
    attack_type = "truncation"
    config_cls = TruncationConfig

    def _apply(self, text: str, seed: int):
        w = _words(text)
        n = len(w)
        k = n
        if self.config.keep_fraction is not None:
            k = max(1, round(n * self.config.keep_fraction))
        if self.config.max_words is not None:
            k = min(k, self.config.max_words)
        kept = w[:k] if self.config.keep == "start" else w[n - k :]
        return " ".join(kept), {"words_before": n, "words_after": len(kept)}


class RateConfig(AttackConfig):
    rate: float = Field(default=0.1, ge=0.0, le=1.0)


class WordDeletion(Attack):
    name = "word_deletion"
    attack_type = "edit"
    config_cls = RateConfig

    def _apply(self, text: str, seed: int):
        rng = random.Random(seed)
        w = _words(text)
        kept = [x for x in w if rng.random() >= self.config.rate]
        if not kept:
            kept = w[:1]
        return " ".join(kept), {"deleted": len(w) - len(kept)}


class WordSwap(Attack):
    """Swap adjacent word pairs."""

    name = "word_swap"
    attack_type = "edit"
    config_cls = RateConfig

    def _apply(self, text: str, seed: int):
        rng = random.Random(seed)
        w = _words(text)
        swaps, i = 0, 0
        while i < len(w) - 1:
            if rng.random() < self.config.rate:
                w[i], w[i + 1] = w[i + 1], w[i]
                swaps += 1
                i += 2
            else:
                i += 1
        return " ".join(w), {"swaps": swaps}


_FILLERS = ["the", "a", "and", "of", "to", "in", "that", "it", "is", "was", "for", "on", "with", "as", "really", "also"]


class WordInsertion(Attack):
    """Insert common function words at random positions."""

    name = "word_insertion"
    attack_type = "edit"
    config_cls = RateConfig

    def _apply(self, text: str, seed: int):
        rng = random.Random(seed)
        out, inserted = [], 0
        for x in _words(text):
            out.append(x)
            if rng.random() < self.config.rate:
                out.append(rng.choice(_FILLERS))
                inserted += 1
        return " ".join(out), {"inserted": inserted}


class CharNoise(Attack):
    """Typo simulation: swap / delete / insert / substitute a character in selected words."""

    name = "char_noise"
    attack_type = "edit"
    config_cls = RateConfig

    def _apply(self, text: str, seed: int):
        rng = random.Random(seed)
        out, n = [], 0
        for x in _words(text):
            if len(x) > 2 and rng.random() < self.config.rate:
                i = rng.randrange(len(x) - 1)
                op = rng.choice(["swap", "delete", "insert", "sub"])
                c = rng.choice(string.ascii_lowercase)
                if op == "swap":
                    x = x[:i] + x[i + 1] + x[i] + x[i + 2 :]
                elif op == "delete":
                    x = x[:i] + x[i + 1 :]
                elif op == "insert":
                    x = x[:i] + c + x[i:]
                else:
                    x = x[:i] + c + x[i + 1 :]
                n += 1
            out.append(x)
        return " ".join(out), {"perturbed_words": n}


_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}


class FormattingConfig(AttackConfig):
    operations: list[
        Literal["normalize_unicode", "collapse_whitespace", "lowercase", "strip_punctuation", "homoglyphs"]
    ] = Field(default_factory=lambda: ["normalize_unicode", "collapse_whitespace"])
    homoglyph_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class Formatting(Attack):
    """Formatting-level changes. `normalize_unicode` = NFKC + strip format (Cf) chars."""

    name = "formatting"
    attack_type = "formatting"
    config_cls = FormattingConfig

    def _apply(self, text: str, seed: int):
        rng = random.Random(seed)
        meta: dict[str, Any] = {}
        for op in self.config.operations:
            if op == "normalize_unicode":
                before = len(text)
                text = unicodedata.normalize("NFKC", text)
                text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
                meta["removed_format_chars"] = before - len(text)
            elif op == "collapse_whitespace":
                text = re.sub(r"\s+", " ", text).strip()
            elif op == "lowercase":
                text = text.lower()
            elif op == "strip_punctuation":
                text = text.translate(str.maketrans("", "", string.punctuation))
            elif op == "homoglyphs":
                chars = list(text)
                n = 0
                for i, ch in enumerate(chars):
                    if ch in _HOMOGLYPHS and rng.random() < self.config.homoglyph_rate:
                        chars[i] = _HOMOGLYPHS[ch]
                        n += 1
                text = "".join(chars)
                meta["homoglyphs"] = n
        return text, meta
