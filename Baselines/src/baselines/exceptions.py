from __future__ import annotations


class CausalArenaBaselineError(Exception):
    """Base class for public baseline errors."""


class CATEValidationError(CausalArenaBaselineError, ValueError):
    """Raised when estimator input or output violates the CATE contract."""


class NotFittedError(CausalArenaBaselineError, RuntimeError):
    """Raised when prediction is requested before fitting."""


class UnsupportedOutcomeError(CausalArenaBaselineError, ValueError):
    """Raised when an estimator cannot represent the requested outcome."""


class DependencyUnavailableError(CausalArenaBaselineError, ImportError):
    """Raised when an optional model backend is unavailable."""


class RBackendError(CausalArenaBaselineError, RuntimeError):
    """Raised when an isolated R backend job fails."""

