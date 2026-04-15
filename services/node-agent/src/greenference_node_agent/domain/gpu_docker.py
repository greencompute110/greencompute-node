"""Detect the correct Docker GPU passthrough method for this host.

Docker + NVIDIA has three eras of GPU passthrough:
  1. nvidia-docker2: --runtime=nvidia + NVIDIA_VISIBLE_DEVICES env var
  2. NVIDIA Container Toolkit (legacy): --gpus flag using nvidia-container-cli
  3. NVIDIA Container Toolkit (CDI): --gpus flag using CDI device specs

Newer Docker versions (25+) default to CDI for --gpus, which fails if
`nvidia-ctk cdi generate` was never run. This module probes once at import
time and caches the working method.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Cached result: "gpus", "runtime", or "env_only"
_gpu_mode: str | None = None


def _probe_gpu_mode() -> str:
    """Try each GPU method with a lightweight container and return the first that works."""

    test_image = "nvidia/cuda:12.4.1-base-ubuntu22.04"

    # Method 1: --gpus all (works if CDI is set up or on older Docker)
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", test_image, "nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            logger.info("GPU mode: --gpus (CDI or legacy nvidia-container-cli)")
            return "gpus"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: --runtime=nvidia (nvidia-docker2 / manually registered runtime)
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--runtime=nvidia",
             "-e", "NVIDIA_VISIBLE_DEVICES=all", test_image, "nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            logger.info("GPU mode: --runtime=nvidia")
            return "runtime"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: try generating CDI specs then retry --gpus
    try:
        gen = subprocess.run(
            ["nvidia-ctk", "cdi", "generate", "--output=/etc/cdi/nvidia.yaml"],
            capture_output=True, text=True, timeout=30,
        )
        if gen.returncode == 0:
            logger.info("Generated CDI specs, retrying --gpus")
            r = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", test_image, "nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                logger.info("GPU mode: --gpus (after CDI generate)")
                return "gpus"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: just NVIDIA_VISIBLE_DEVICES env (requires default runtime = nvidia)
    logger.warning("No GPU passthrough method verified — falling back to NVIDIA_VISIBLE_DEVICES env only")
    return "env_only"


def get_gpu_mode() -> str:
    """Return the cached GPU mode, probing on first call."""
    global _gpu_mode
    if _gpu_mode is None:
        _gpu_mode = _probe_gpu_mode()
    return _gpu_mode


def gpu_docker_flags(device_ids: list[int] | None) -> list[str]:
    """Return Docker CLI flags to pass GPUs to a container.

    Args:
        device_ids: specific GPU device IDs, or None for all GPUs.

    Returns:
        List of CLI args to insert into `docker run ...` command.
    """
    mode = get_gpu_mode()
    device_str = ",".join(str(d) for d in device_ids) if device_ids else "all"

    if mode == "gpus":
        if device_ids:
            return ["--gpus", f"device={device_str}"]
        return ["--gpus", "all"]
    elif mode == "runtime":
        return ["--runtime=nvidia", "-e", f"NVIDIA_VISIBLE_DEVICES={device_str}"]
    else:
        # env_only — hope the default runtime handles it
        return ["-e", f"NVIDIA_VISIBLE_DEVICES={device_str}"]
