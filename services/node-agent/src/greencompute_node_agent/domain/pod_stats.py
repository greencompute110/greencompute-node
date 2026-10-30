"""Live pod stats — wraps `docker stats`, `nvidia-smi` (per GPU), and `docker exec df`.

Called by the /pods/{deployment_id}/stats endpoint. Best-effort: any probe
that fails is silently omitted from the result so the UI can render partial
data instead of erroring.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

#: Shared by every stats request (three probes each), not a pool per request.
_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="pod-stats")
PROBE_TIMEOUT = 4.0


def collect_pod_stats(
    container_id: str,
    gpu_device_ids: list[int] | None = None,
    workspace_path: str = "/workspace",
) -> dict[str, Any]:
    """The three probes run side by side: the slowest one, not their sum
    (docker stats alone takes about 2 s), bounds a reading, and each gives
    up after 4 s, inside the gateway's 6 s wait."""
    docker_f = _POOL.submit(_docker_stats, container_id)
    gpus_f = _POOL.submit(gpu_details, gpu_device_ids)
    disk_f = _POOL.submit(_disk_usage, container_id, workspace_path)
    cpu_pct, mem_used, mem_limit, net_rx, net_tx = docker_f.result()
    gpus = gpus_f.result()
    disk_used, disk_total = disk_f.result()

    out: dict[str, Any] = {}
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

    # Each GPU (use, memory, temperature, power, clocks, fan), and the totals
    # the dashboard shows. VRAM totals only over GPUs that report both, so
    # used / total is never over 100%.
    if gpus:
        out["gpus"] = gpus
        utils = [g["util_pct"] for g in gpus if g.get("util_pct") is not None]
        both = [g for g in gpus if g.get("memory_used_mb") is not None and g.get("memory_total_mb") is not None]
        if utils:
            out["gpu_util_pct"] = sum(utils) / len(utils)
        if both:
            out["vram_used_mb"] = sum(g["memory_used_mb"] for g in both)
            out["vram_total_mb"] = sum(g["memory_total_mb"] for g in both)

    if disk_used is not None:
        out["disk_used_gb"] = disk_used
    if disk_total is not None:
        out["disk_total_gb"] = disk_total
    return out


#: nvidia-smi --query-gpu fields, and the names the API gives them.
_GPU_QUERY = (
    ("index", None),
    ("uuid", "uuid"),
    ("name", "name"),
    ("utilization.gpu", "util_pct"),
    ("memory.used", "memory_used_mb"),
    ("memory.total", "memory_total_mb"),
    ("temperature.gpu", "temperature_c"),
    ("power.draw", "power_w"),
    ("power.limit", "power_limit_w"),
    ("clocks.sm", "sm_clock_mhz"),
    ("clocks.mem", "mem_clock_mhz"),
    ("fan.speed", "fan_pct"),
)
_TEXT_FIELDS = {"uuid", "name"}


def _number(value: str) -> float | None:
    """A reading, or None for "[N/A]", "[Not Supported]" and the like."""
    try:
        return float(value)
    except ValueError:
        return None


def gpu_details(device_ids: list[int] | None) -> list[dict[str, Any]]:
    """One entry per GPU of the pod, numbered as the pod sees them (0, 1, …
    in the order of the machine's numbering), with the UUID `nvidia-smi -L`
    shows inside the pod. Readings the card doesn't report are null. One
    nvidia-smi call on the machine, which the renter can't influence."""
    if not device_ids:
        return []
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(d) for d in sorted(device_ids))}",
        "--query-gpu=" + ",".join(field for field, _ in _GPU_QUERY),
        "--format=csv,noheader,nounits",
    ]
    order = {d: i for i, d in enumerate(sorted(device_ids))}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)  # noqa: S603
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    # A card that has failed makes nvidia-smi exit with an error but still
    # print the others: keep those, and name the one that didn't answer.
    gpus: list[dict[str, Any]] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_QUERY):
            continue
        try:
            host_index = int(parts[0])
        except ValueError:
            continue
        if host_index not in order:
            continue  # not one of the pod's GPUs
        gpu: dict[str, Any] = {"index": order[host_index]}
        for (field, key), value in zip(_GPU_QUERY[1:], parts[1:]):
            gpu[key] = value if field in _TEXT_FIELDS else _number(value)
        gpus.append(gpu)
    if not gpus and r.returncode != 0:
        return []  # nvidia-smi itself isn't working: nothing to say per card
    answered = {g["index"] for g in gpus}
    for host_index, pod_index in order.items():
        if pod_index not in answered:
            gpus.append({"index": pod_index, "error": "this GPU didn't answer nvidia-smi"})
    return sorted(gpus, key=lambda g: g["index"])


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
            timeout=PROBE_TIMEOUT,
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


def _disk_usage(container_id: str, path: str) -> tuple[float | None, float | None]:
    """`df -B1 <path>` inside the container: (used_gb, total_gb). Its output
    comes from the renter's image, so it is read with a size cap."""
    from greencompute_node_agent.domain.pod import bounded_output

    raw = bounded_output(["docker", "exec", container_id, "df", "-B1", path], limit=16384, timeout=PROBE_TIMEOUT)
    if raw is None:
        return None, None
    # Filesystem     1B-blocks       Used Available Use% Mounted on
    # /dev/loop0   53687091200  ...
    lines = raw.decode("utf-8", "replace").strip().splitlines()
    if len(lines) < 2:
        return None, None
    fields = lines[-1].split()
    if len(fields) < 4:
        return None, None
    try:
        total_bytes = float(fields[1])
        used_bytes = float(fields[2])
    except ValueError:
        return None, None
    return used_bytes / 1e9, total_bytes / 1e9


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
