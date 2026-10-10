"""Domain exceptions. The API layer maps these to HTTP status codes."""

from __future__ import annotations


class MAIError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationFailed(MAIError):
    status_code = 422
    code = "validation_failed"


class NotFound(MAIError):
    status_code = 404
    code = "not_found"


class Conflict(MAIError):
    status_code = 409
    code = "conflict"


class InsufficientData(MAIError):
    status_code = 422
    code = "insufficient_data"


class ProviderError(MAIError):
    status_code = 502
    code = "provider_error"


class ProviderUnavailable(ProviderError):
    """Provider not configured (e.g. missing API key) or dependency missing."""

    status_code = 503
    code = "provider_unavailable"


class ProviderTimeout(ProviderError):
    status_code = 504
    code = "provider_timeout"


class ProviderRateLimited(ProviderError):
    status_code = 429
    code = "provider_rate_limited"


class ProviderQuotaExceeded(ProviderError):
    status_code = 429
    code = "provider_quota_exceeded"


class ProviderInvalidResponse(ProviderError):
    status_code = 502
    code = "provider_invalid_response"
