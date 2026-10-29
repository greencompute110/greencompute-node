"""Run a command in a renter's pod without SSH (external API user, request
#32: updating the miner on every card meant logging in to each one).

As root, with `sh -c`, from /workspace. The renter is root in the pod over
SSH anyway, so this gives nothing SSH doesn't. Output (stdout and stderr) is
kept to its last 64 KiB. After `timeout_seconds` we stop waiting: the command
itself may still be running in the pod (docker can't stop it from outside),
so long jobs belong in the background with their own log.
"""
from __future__ import annotations

from typing import Any

from greencompute_node_agent.domain.pod import run_capped
from greencompute_node_agent.domain.pod_logs import clean_terminal_text

MAX_COMMAND_CHARS = 8192
MAX_TIMEOUT_SECONDS = 120
KEEP_BYTES = 64 * 1024


def run_in_pod(container_id: str, command: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
    timeout_seconds = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
    code, output, dropped, timed_out = run_capped(
        ["docker", "exec", "-u", "0", "-w", "/workspace", container_id, "sh", "-c", command],
        keep=KEEP_BYTES, timeout=float(timeout_seconds))
    return {
        "exit_code": code,
        "output": clean_terminal_text(output.decode("utf-8", "replace")),
        "output_truncated": dropped,
        "timed_out": timed_out,
    }
