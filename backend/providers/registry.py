from __future__ import annotations

from backend.core.errors import ValidationFailed
from backend.providers.base import ProviderStatus, TextProvider
from backend.providers.gemini import GeminiProvider
from backend.providers.local import LocalHFInstructProvider

_providers: dict[str, TextProvider] = {}


def _build() -> dict[str, TextProvider]:
    if not _providers:
        for p in (GeminiProvider(), LocalHFInstructProvider()):
            _providers[p.name] = p
    return _providers


def get_provider(name: str) -> TextProvider:
    providers = _build()
    if name not in providers:
        raise ValidationFailed(f"Unknown provider {name!r}", details={"available": list(providers)})
    return providers[name]


def set_provider(provider: TextProvider) -> None:
    """Override a provider instance (tests / custom adapters)."""
    _build()[provider.name] = provider


def reset_providers() -> None:
    _providers.clear()


def list_statuses() -> list[ProviderStatus]:
    return [p.status() for p in _build().values()]
