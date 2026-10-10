from __future__ import annotations

from typing import Any

from backend.attacks.base import Attack
from backend.attacks.deterministic import CharNoise, Formatting, Truncation, WordDeletion, WordInsertion, WordSwap
from backend.attacks.model_based import BackTranslation, LexicalSubstitution, Paraphrase
from backend.core.errors import ValidationFailed

ATTACKS: dict[str, type[Attack]] = {
    cls.name: cls
    for cls in (
        Truncation,
        WordDeletion,
        WordSwap,
        WordInsertion,
        CharNoise,
        Formatting,
        LexicalSubstitution,
        BackTranslation,
        Paraphrase,
    )
}


def build_attack(name: str, config: dict[str, Any] | None = None) -> Attack:
    if name not in ATTACKS:
        raise ValidationFailed(f"Unknown attack {name!r}", details={"available": sorted(ATTACKS)})
    try:
        return ATTACKS[name](config or {})
    except ValueError as exc:
        raise ValidationFailed(f"Invalid config for attack {name!r}: {exc}") from exc
