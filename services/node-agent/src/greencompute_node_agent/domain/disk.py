"""Per-pod disk quota enforcement — auto-detect + loop-mount helpers.

Priority order of modes:
  1. LOOP_MOUNT       (we're root) — strongest: /workspace is an ext4 loop mount.
  2. LOOP_MOUNT_SUDO  (passwordless sudo for mount/losetup/etc.) — same semantics.
  3. STORAGE_OPT      (Docker supports --storage-opt size=, typical on XFS+pquota)
                      — partial: caps the container overlay, /workspace bind-mount is still unbounded.
  4. NONE             — no enforcement; matches legacy behavior. Warn at startup.
"""

from __future__ import annotations

import enum
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class DiskMode(str, enum.Enum):
    LOOP_MOUNT = "loop_mount"
    LOOP_MOUNT_SUDO = "loop_mount_sudo"
    STORAGE_OPT = "storage_opt"
    NONE = "none"


def detect_disk_mode(override: str | None = None) -> DiskMode:
    """Pick the strongest disk-enforcement mode this host supports."""
    if override:
        try:
            return DiskMode(override.lower())
        except ValueError:
            logger.warning("invalid GREENCOMPUTE_DISK_ENFORCEMENT_MODE=%r — falling back to auto", override)

    # 1. Root — can loop-mount directly.
    if _has_mount_tools() and os.geteuid() == 0:
        return DiskMode.LOOP_MOUNT

    # 2. Non-root but has passwordless sudo — can loop-mount via sudo.
    if _has_mount_tools() and _can_sudo_nopasswd():
        return DiskMode.LOOP_MOUNT_SUDO

    # 3. Docker supports --storage-opt size= (XFS+pquota typically).
    if _docker_storage_opt_supported():
        return DiskMode.STORAGE_OPT

    # 4. Give up — legacy unbounded behavior.
    return DiskMode.NONE


def _has_mount_tools() -> bool:
    return all(shutil.which(t) for t in ("mount", "umount", "mkfs.ext4", "truncate"))


def _can_sudo_nopasswd() -> bool:
    """Can we run `sudo -n mount` without a password prompt?"""
    try:
        r = subprocess.run(
            ["sudo", "-n", "-l", "mount"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_storage_opt_supported() -> bool:
    """Try to create a throwaway container with --storage-opt size=1G. If Docker
    rejects the flag (wrong storage driver / no pquota), we fall back."""
    try:
        r = subprocess.run(
            ["docker", "create", "--storage-opt", "size=1G", "alpine:latest", "true"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            cid = r.stdout.strip()
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)
            return True
        # Some Docker versions need an existing image to even validate the flag;
        # try pulling alpine once and retrying.
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _sudo_prefix(mode: DiskMode) -> list[str]:
    return ["sudo", "-n"] if mode == DiskMode.LOOP_MOUNT_SUDO else []


def create_loop_volume(
    mount_path: Path,
    image_path: Path,
    size_gb: int,
    mode: DiskMode,
) -> None:
    """Create a sparse ext4 image, format it, and loop-mount at mount_path.

    Raises DiskError on any failure; leaves no partial state (best effort cleanup).
    """
    if size_gb < 1:
        raise DiskError(f"invalid size_gb={size_gb}", failure_class="disk_quota_setup")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mount_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Sparse file of requested size (truncate doesn't allocate blocks upfront —
        #    ext4 will grow real usage on demand, capped at size_gb).
        subprocess.run(
            ["truncate", "-s", f"{size_gb}G", str(image_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        # 2. Format as ext4.
        subprocess.run(
            ["mkfs.ext4", "-F", "-q", str(image_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        # 3. Loop-mount.
        subprocess.run(
            [*_sudo_prefix(mode), "mount", "-o", "loop", str(image_path), str(mount_path)],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        # Best-effort rollback so we don't leak partial state.
        try:
            subprocess.run(
                [*_sudo_prefix(mode), "umount", str(mount_path)],
                capture_output=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        image_path.unlink(missing_ok=True)
        raise DiskError(
            f"loop volume setup failed: {exc}",
            failure_class="disk_quota_setup",
        ) from exc


def destroy_loop_volume(mount_path: Path, image_path: Path, mode: DiskMode) -> None:
    """Unmount + remove the loop image. Swallows errors — called on teardown."""
    try:
        subprocess.run(
            [*_sudo_prefix(mode), "umount", str(mount_path)],
            capture_output=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        image_path.unlink(missing_ok=True)
    except OSError:
        pass


class DiskError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str = "disk_error") -> None:
        super().__init__(message)
        self.failure_class = failure_class
