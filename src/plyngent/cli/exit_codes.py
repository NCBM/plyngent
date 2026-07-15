"""Process exit codes for ``plyngent chat`` (one-shot and fatal paths)."""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1  # config / usage / fatal
EXIT_CANCELLED = 2
EXIT_TURN_FAILED = 3  # API / retry exhausted / incomplete turn
