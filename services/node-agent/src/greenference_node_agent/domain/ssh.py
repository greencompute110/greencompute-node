"""Ephemeral SSH keypair generation and SSH access record building."""

from __future__ import annotations

import hashlib
import secrets
import socket
import subprocess
import tempfile
from pathlib import Path

from greenference_protocol import SSHAccessRecord, UnifiedRuntimeRecord


class SSHError(RuntimeError):
    pass


def generate_ssh_keypair() -> tuple[str, str]:
    """Generate an ephemeral ed25519 keypair. Returns (private_key_pem, public_key_openssh)."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "id_ed25519"
            result = subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "ssh-keygen",
                    "-t", "ed25519",
                    "-f", str(key_path),
                    "-N", "",
                    "-C", "greenference-ephemeral",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0:
                raise SSHError(f"ssh-keygen failed: {result.stderr}")
            private_key = key_path.read_text(encoding="utf-8")
            public_key = (key_path.with_suffix(".pub")).read_text(encoding="utf-8").strip()
            return private_key, public_key
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise SSHError(f"ssh-keygen not available: {exc}") from exc


def _fingerprint_from_public_key(public_key_openssh: str) -> str:
    """Compute a SHA256 fingerprint from the public key bytes."""
    parts = public_key_openssh.strip().split()
    if len(parts) >= 2:
        import base64
        try:
            key_bytes = base64.b64decode(parts[1])
            digest = hashlib.sha256(key_bytes).hexdigest()[:16]
            return f"SHA256:{digest}"
        except Exception:  # noqa: BLE001
            pass
    return f"SHA256:{hashlib.sha256(public_key_openssh.encode()).hexdigest()[:16]}"


def choose_free_port(start: int = 30000, end: int = 31000) -> int:
    """Pick a free port in [start, end)."""
    tried: set[int] = set()
    while len(tried) < (end - start):
        port = secrets.randbelow(end - start) + start
        if port in tried:
            continue
        tried.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise SSHError("no free SSH port found in range")


def is_port_free(port: int) -> bool:
    """Check if a specific host port can be bound on 0.0.0.0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def build_ssh_access(
    runtime: UnifiedRuntimeRecord,
    *,
    include_private_key: bool = False,
    private_key: str | None = None,
) -> SSHAccessRecord:
    return SSHAccessRecord(
        deployment_id=runtime.deployment_id,
        host=runtime.ssh_host or "127.0.0.1",
        port=runtime.ssh_port or 22,
        username=runtime.ssh_username,
        private_key=private_key if include_private_key else None,
        fingerprint=runtime.ssh_fingerprint,
        ready=runtime.status == "ready",
    )
