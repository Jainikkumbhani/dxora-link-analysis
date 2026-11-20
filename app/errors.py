"""Application errors. Each maps to an HTTP status and a stable error code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidURLError(AppError):
    status_code = 422
    code = "invalid_url"


class BlockedURLError(AppError):
    """URL points to a private, internal or otherwise forbidden address (SSRF)."""

    status_code = 400
    code = "blocked_url"


class WebsiteUnreachableError(AppError):
    status_code = 502
    code = "website_unreachable"


class WebsiteTimeoutError(AppError):
    status_code = 504
    code = "website_timeout"


class UnsupportedContentError(AppError):
    status_code = 422
    code = "unsupported_content"


class InsufficientContentError(AppError):
    status_code = 422
    code = "insufficient_content"


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(AppError):
    status_code = 504
    code = "llm_timeout"


class LLMUnavailableError(AppError):
    """All configured models are overloaded / unavailable (temporary)."""

    status_code = 503
    code = "llm_unavailable"


class LLMRateLimitedError(AppError):
    status_code = 503
    code = "llm_rate_limited"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
