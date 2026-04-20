"""Live pod stats — wraps `docker stats`, `nvidia-smi`, and `docker exec df`.

Called by the /pods/{deployment_id}/stats endpoint. Best-effort: any probe
that fails is silently omitted from the result so the UI can render partial
data instead of erroring.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def collect_pod_stats(
    container_id: str,
    gpu_device_ids: list[int] | None = None,
    workspace_path: str = "/workspace",
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # 1. docker stats — CPU %, mem, network
    cpu_pct, mem_used, mem_limit, net_rx, net_tx = _docker_stats(container_id)
    if cpu_pct is not None:
        out["cpu_pct"] = cpu_pct
    if mem_used is not None:
        out["memory_used_mb"] = mem_used
    if mem_limit is not None:
        out["memory_limit_mb"] = mem_limit
    if net_rx is not None:
        out["net_rx_bytes"] = net_rx
    if net_tx is not None:
        out["net_tx_bytes"] = net_tx

    # 2. GPU util + VRAM
    gpu_util, vram_used, vram_total = _nvidia_smi(gpu_device_ids)
    if gpu_util is not None:
        out["gpu_util_pct"] = gpu_util
    if vram_used is not None:
        out["vram_used_mb"] = vram_used
    if vram_total is not None:
        out["vram_total_mb"] = vram_total

    # 3. Disk — df inside the container
    disk_used, disk_total = _disk_usage(container_id, workspace_path)
    if disk_used is not None:
        out["disk_used_gb"] = disk_used
    if disk_total is not None:
        out["disk_total_gb"] = disk_total

    return out


def _docker_stats(container_id: str) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Run `docker stats --no-stream`. Returns (cpu_pct, mem_mb, mem_limit_mb, net_rx, net_tx)."""
    try:
        r = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "docker", "stats", "--no-stream", "--format",
                "{{json .}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None, None, None, None, None
        data = json.loads(r.stdout.strip().splitlines()[0])
        cpu_pct = _parse_pct(data.get("CPUPerc", ""))
        mem_used, mem_limit = _parse_mem_usage(data.get("MemUsage", ""))
        net_rx, net_tx = _parse_net_io(data.get("NetIO", ""))
        return cpu_pct, mem_used, mem_limit, net_rx, net_tx
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None, None, None, None, None


def _nvidia_smi(device_ids: list[int] | None) -> tuple[float | None, float | None, float | None]:
    """Aggregate GPU util + VRAM across the pod's allocated devices."""
    if not device_ids:
        return None, None, None
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(d) for d in device_ids)}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)  # noqa: S603
        if r.returncode != 0 or not r.stdout.strip():
            return None, None, None
        utils: list[float] = []
        used: list[float] = []
        total: list[float] = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    utils.append(float(parts[0]))
                    used.append(float(parts[1]))
                    total.append(float(parts[2]))
                except ValueError:
                    pass
        if not utils:
            return None, None, None
        avg_util = sum(utils) / len(utils)
        return avg_util, sum(used), sum(total)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None, None


def _disk_usage(container_id: str, path: str) -> tuple[float | None, float | None]:
    """Run `df -B1 <path>` inside container. Returns (used_gb, total_gb)."""
    try:
        r = subprocess.run(  # noqa: S603
            ["docker", "exec", container_id, "df", "-B1", path],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if r.returncode != 0:
            return None, None
        # Output format:
        # Filesystem     1B-blocks       Used Available Use% Mounted on
        # /dev/loop0  53,687,091,200  ...
        lines = r.stdout.strip().splitlines()
        if len(lines) < 2:
            return None, None
        fields = lines[-1].split()
        if len(fields) < 4:
            return None, None
        total_bytes = float(fields[1])
        used_bytes = float(fields[2])
        return used_bytes / 1e9, total_bytes / 1e9
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None, None


# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r"([\d.]+)")
_MEM_RE = re.compile(r"([\d.]+)\s*([KMGT]?i?B)\s*/\s*([\d.]+)\s*([KMGT]?i?B)", re.IGNORECASE)
_NET_RE = re.compile(r"([\d.]+)\s*([KMGT]?i?B)\s*/\s*([\d.]+)\s*([KMGT]?i?B)", re.IGNORECASE)


def _parse_pct(s: str) -> float | None:
    m = _PCT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_mem_usage(s: str) -> tuple[float | None, float | None]:
    """Parse strings like '123.4MiB / 16GiB'. Returns values in MB."""
    m = _MEM_RE.search(s)
    if not m:
        return None, None
    try:
        return _to_mb(float(m.group(1)), m.group(2)), _to_mb(float(m.group(3)), m.group(4))
    except ValueError:
        return None, None


def _parse_net_io(s: str) -> tuple[float | None, float | None]:
    """Parse strings like '1.2MB / 3.4MB'. Returns bytes."""
    m = _NET_RE.search(s)
    if not m:
        return None, None
    try:
        return _to_bytes(float(m.group(1)), m.group(2)), _to_bytes(float(m.group(3)), m.group(4))
    except ValueError:
        return None, None


_UNIT_SCALE = {
    "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
    "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4,
}


def _to_bytes(value: float, unit: str) -> float:
    return value * _UNIT_SCALE.get(unit, 1)


def _to_mb(value: float, unit: str) -> float:
    return _to_bytes(value, unit) / 1e6
