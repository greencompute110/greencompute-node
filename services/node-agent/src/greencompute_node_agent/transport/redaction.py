"""Secret redaction for runtime observability responses.

Pure functions (no FastAPI dependency) so they are unit-testable and reusable
outside the transport layer. The observability routes must never leak SSH
private keys, HF tokens, or other credentials stored in runtime metadata; the
dedicated /ssh endpoint is the only path allowed to emit the SSH private key
(and only when include_private_key=True).
"""

from __future__ import annotations

from typing import Any

REDACT_PLACEHOLDER = "***redacted***"


def is_secret_key(key: str) -> bool:
    k = key.lower()
    # NB: keep ssh_public_keys / ssh_fingerprint visible — only private material.
    return (
        "private_key" in k
        # A pod's environment and start command, from the user: any of it
        # may be a credential, whatever the variable is called.
        or k == "env_vars"
        or "secret" in k
        or "password" in k
        or "hf_token" in k
        or k == "token"
        or k.endswith("_token")
    )


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: (REDACT_PLACEHOLDER if is_secret_key(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
