"""Ephemeral SSH keypair generation and SSH access record building."""

from __future__ import annotations

import hashlib
import secrets
import socket
import subprocess
import tempfile
from pathlib import Path

from greencompute_protocol import SSHAccessRecord, UnifiedRuntimeRecord


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
                    "-C", "greencompute-ephemeral",
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


def pod_ssh_keys(user_keys: list[str] | None) -> tuple[str | None, list[str]]:
    """The public keys a pod's sshd accepts, and the private half of any key
    the platform generated for it, as (private_key, public_keys).

    A workload that brings its own keys gets exactly those: no platform key is
    generated, so nobody but the user holds a key to the pod. Only a workload
    with none gets a generated keypair, whose private half the owner fetches
    from the API's /ssh endpoint."""
    own = [k.strip() for k in (user_keys or []) if isinstance(k, str) and k.strip()]
    if own:
        return None, own
    private_key, public_key = generate_ssh_keypair()
    return private_key, [public_key]


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


def _docker_bound_ports() -> set[int]:
    """All host ports currently claimed by Docker containers.

    Needed because Docker's iptables-NAT port publishing doesn't show up as a
    listening socket on the host, so socket.bind() alone can succeed even
    though `docker run -p <port>:...` will then fail with "port already
    allocated". Calling this before returning a port avoids that race.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["docker", "ps", "--format", "{{.Ports}}"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    ports: set[int] = set()
    for line in result.stdout.splitlines():
        # Examples:
        #   "0.0.0.0:30375->22/tcp, :::30375->22/tcp"
        #   "0.0.0.0:31042->8080/tcp"
        for chunk in line.split(","):
            chunk = chunk.strip()
            # pull the part before "->"
            if "->" not in chunk:
                continue
            lhs = chunk.split("->", 1)[0]
            # lhs looks like "0.0.0.0:30375" or ":::30375" or "30375"
            tail = lhs.rsplit(":", 1)[-1]
            try:
                ports.add(int(tail))
            except ValueError:
                continue
    return ports


def choose_free_port(start: int = 30000, end: int = 31000) -> int:
    """Pick a free port in [start, end).

    Checks both kernel socket binding AND Docker's current port allocations
    (see `_docker_bound_ports` for why both are required).
    """
    tried: set[int] = set()
    docker_used = _docker_bound_ports()
    while len(tried) < (end - start):
        port = secrets.randbelow(end - start) + start
        if port in tried:
            continue
        tried.add(port)
        if port in docker_used:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise SSHError("no free SSH port found in range")


def is_port_free(port: int) -> bool:
    """Check if a specific host port can be bound on 0.0.0.0 AND isn't held by Docker."""
    if port in _docker_bound_ports():
        return False
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
