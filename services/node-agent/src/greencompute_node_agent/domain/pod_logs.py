"""A pod's recent output, for GET /pods/{id}/logs (external API user,
request #17: "the only way to check a card is to SSH in").

Two sources: the workload's start command writes to
/workspace/greencompute-command.log (see the entrypoint), and the container's
own output (the image's start command, our entrypoint's lines, a model
server's log) is its docker log. Both are the renter's own text and can be
anything, so each is read keeping only its end, on this side; terminal colour
codes and progress-bar redraws are taken out so it reads as text. When a
source is missing, its note says why, so "nothing yet" and "couldn't read"
aren't confused.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from greencompute_node_agent.domain.pod import bounded_tail

COMMAND_LOG = "/workspace/greencompute-command.log"
MAX_LINES = 2000
#: How much of each source's end is kept.
KEEP_BYTES = 256 * 1024
#: Both sources are read side by side (a reply well within the gateway's wait).
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pod-logs")

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_terminal_text(text: str) -> str:
    """Text as a terminal would end up showing it: colour codes gone, and of
    a line redrawn with carriage returns (a progress bar), its last state."""
    text = _ANSI.sub("", text)
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        lines.append(_CONTROL.sub("", line))
    return "\n".join(lines)


def _last_lines(raw: bytes, lines: int) -> str:
    return "\n".join(clean_terminal_text(raw.decode("utf-8", "replace")).splitlines()[-lines:])


def _command_log(container_id: str) -> tuple[bytes, bool] | None:
    # As root: the log is root's, whatever the image's USER. Only its end
    # is read (tail seeks), and only that much is kept.
    return bounded_tail(["docker", "exec", "-u", "0", container_id, "tail", "-c", str(KEEP_BYTES), COMMAND_LOG],
                        keep=KEEP_BYTES, read_limit=KEEP_BYTES * 2, timeout=5.0)


def _container_log(container_id: str, lines: int) -> tuple[bytes, bool] | None:
    return bounded_tail(["docker", "logs", "--tail", str(lines), container_id],
                        keep=KEEP_BYTES, read_limit=64 * 1024 * 1024, timeout=6.0, merge_stderr=True)


def collect_pod_logs(container_id: str, *, lines: int = 200, has_command: bool = True) -> dict[str, Any]:
    lines = max(1, min(int(lines), MAX_LINES))
    out: dict[str, Any] = {"lines": lines, "command_log": None, "command_log_note": None,
                           "container_log": None, "container_log_note": None}
    command_f = _POOL.submit(_command_log, container_id) if has_command else None
    container_f = _POOL.submit(_container_log, container_id, lines)
    if command_f is None:
        out["command_log_note"] = "this rental has no start command (metadata.command)"
    else:
        got = command_f.result()
        if got is None:
            out["command_log_note"] = "not written yet, or couldn't be read (the image needs `tail`)"
        else:
            out["command_log"] = _last_lines(got[0], lines)
    got = container_f.result()
    if got is None:
        out["container_log_note"] = "couldn't be read"
    elif not got[1]:
        out["container_log_note"] = f"the last {lines} lines are too long to show; ask for fewer"
    else:
        out["container_log"] = _last_lines(got[0], lines)
    return out
