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

import io
import re
import tarfile
from typing import Any

from greencompute_node_agent.domain.pod import bounded_tail

COMMAND_LOG = "/workspace/greencompute-command.log"
MAX_LINES = 2000
#: How much of each source's end is kept.
KEEP_BYTES = 256 * 1024
#: docker cp of the command log (images without `tail`) reads at most this.
COPY_READ_LIMIT = 64 * 1024 * 1024

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


def _command_log_via_copy(container_id: str) -> tuple[bytes, bool] | None:
    """The log's end by `docker cp` (a tar stream from the daemon: works
    whatever the image has inside). The tar's padding is cut off."""
    got = bounded_tail(["docker", "cp", f"{container_id}:{COMMAND_LOG}", "-"],
                       keep=KEEP_BYTES + 2048, read_limit=COPY_READ_LIMIT, timeout=10.0)
    if got is None:
        return None
    data, complete = got
    if complete and len(data) < KEEP_BYTES:  # the whole archive: read it properly
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                member = tar.next()
                if member is None or not member.isfile():
                    return None
                extracted = tar.extractfile(member)
                return (extracted.read() if extracted else b""), True
        except tarfile.TarError:
            return None
    return data.rstrip(b"\x00"), complete


def collect_pod_logs(container_id: str, *, lines: int = 200, has_command: bool = True) -> dict[str, Any]:
    lines = max(1, min(int(lines), MAX_LINES))
    out: dict[str, Any] = {"lines": lines, "command_log": None, "command_log_note": None,
                           "container_log": None, "container_log_note": None}
    if not has_command:
        out["command_log_note"] = "this rental has no start command (metadata.command)"
    else:
        # As root: the log is root's, whatever the image's USER.
        got = bounded_tail(["docker", "exec", "-u", "0", container_id, "tail", "-c", str(KEEP_BYTES), COMMAND_LOG],
                           keep=KEEP_BYTES, read_limit=KEEP_BYTES * 2, timeout=5.0)
        if got is None:
            got = _command_log_via_copy(container_id)
        if got is None:
            out["command_log_note"] = "not written yet, or couldn't be read"
        else:
            out["command_log"] = _last_lines(got[0], lines)
            if not got[1]:
                out["command_log_note"] = "the log is too large to find its end; it is cut to size within a minute"
    got = bounded_tail(["docker", "logs", "--tail", str(lines), container_id],
                       keep=KEEP_BYTES, read_limit=64 * 1024 * 1024, timeout=8.0, merge_stderr=True)
    if got is None:
        out["container_log_note"] = "couldn't be read"
    elif not got[1]:
        out["container_log_note"] = f"the last {lines} lines are too long to show; ask for fewer"
    else:
        out["container_log"] = _last_lines(got[0], lines)
    return out
