"""Run a command in a renter's pod without SSH (external API user, request
#32: updating the miner on every card meant logging in to each one).

As root, with `sh -c`, from /workspace. The renter is root in the pod over
SSH anyway, so this gives nothing SSH doesn't.

The command's output goes to a file in the pod's /workspace, not to our pipe,
and when it finishes a small wrapper prints the file's last 64 KiB and the
exit code. So a command that starts a job in the background returns as soon
as it does (the job writes to that file, and never to a pipe we might
close), and endless output costs the renter's disk and CPU, not the host's.
After `timeout_seconds` we stop waiting; the command may still be running.

At most MAX_RUNNING commands run at once on a machine (BusyError beyond).
"""
from __future__ import annotations

import secrets
import threading
from typing import Any

from greencompute_node_agent.domain.pod import run_capped
from greencompute_node_agent.domain.pod_logs import clean_terminal_text

MAX_COMMAND_CHARS = 8192
MAX_TIMEOUT_SECONDS = 120
KEEP_BYTES = 64 * 1024
MAX_RUNNING = 4
_SLOTS = threading.BoundedSemaphore(MAX_RUNNING)

# $1: the command, $2: a marker only this call knows. The output file lives in
# /workspace (the pod's own, size-capped disk) and is removed afterwards.
_WRAPPER = (
    'f="/workspace/.greencompute-exec-$2.log"; '
    '( eval "$1" ) >"$f" 2>&1 </dev/null; code=$?; '
    'tail -c ' + str(KEEP_BYTES) + ' "$f" 2>/dev/null || cat "$f"; rm -f "$f"; '
    'printf "\\n%s %s\\n" "$2" "$code"'
)


class BusyError(RuntimeError):
    """Too many commands are running on this machine already."""


def run_in_pod(container_id: str, command: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
    timeout_seconds = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
    if not _SLOTS.acquire(blocking=False):
        raise BusyError("this machine is running other commands; try again in a few seconds")
    try:
        marker = "__gc_exit_" + secrets.token_hex(8)
        code, output, dropped, timed_out = run_capped(
            ["docker", "exec", "-u", "0", "-w", "/workspace", container_id, "sh", "-c", _WRAPPER, "sh",
             command, marker],
            keep=KEEP_BYTES + 256, timeout=float(timeout_seconds))
    finally:
        _SLOTS.release()
    text = output.decode("utf-8", "replace")
    exit_code: int | None = None
    head, sep, tail = text.rpartition("\n" + marker + " ")
    if sep:
        try:
            exit_code = int(tail.strip())
        except ValueError:
            exit_code = None
        text = head
    elif code is not None and not timed_out:
        exit_code = code  # the wrapper itself couldn't run (no sh?): docker's own code
    return {
        "exit_code": exit_code,
        "output": clean_terminal_text(text),
        "output_truncated": dropped or len(text.encode()) >= KEEP_BYTES,
        "timed_out": timed_out,
    }
