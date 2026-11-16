"""An interactive shell in a renter's pod, for the website's terminal
(request #34: "Setting up SSH keys is the hardest step for anyone
non-technical").

`docker exec -it` as root, from /workspace, under a pseudo-terminal of our
own, so the pod gets a real terminal (colours, line editing, vim) whose size
follows the browser's. The renter is root in the pod over SSH anyway, so this
gives nothing SSH doesn't. The shell is a login shell, so it has the image's
environment the way SSH does (entrypoint.sh writes it to /etc/profile.d).

Closing ends it the way a dropped SSH session does: the shell in the pod gets
SIGHUP (bash passes it on to its jobs; nohup, tmux and setsid'd jobs keep
running), then SIGKILL if it is still there. Then the docker CLI here and
anything it started are killed and reaped, so nothing is left behind on the
machine, whatever state the pod or the docker daemon is in.

At most MAX_TERMINALS are open at once on a machine (BusyError beyond).
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import pty
import re
import secrets
import signal
import struct
import subprocess
import termios
import threading
import time

MAX_TERMINALS = 8
_SLOTS = threading.BoundedSemaphore(MAX_TERMINALS)

SHELLS = ("auto", "bash", "sh")
MIN_SIZE, MAX_COLS, MAX_ROWS = 2, 500, 200
READ_CHUNK = 64 * 1024
HANG_UP_TIMEOUT = 10.0
KILL_GRACE_SECONDS = 2.0

# docker's default detach keys are ctrl-p,ctrl-q: it would hold back every
# ctrl-p (bash's "previous command") to see whether a ctrl-q follows, and
# ctrl-p ctrl-q would cut the terminal off. Nobody types this sequence.
DETACH_KEYS = "ctrl-^,ctrl-@,ctrl-^,ctrl-@"

# What docker accepts as a container name or id. Ours come from `docker run`,
# but an id starting with "-" would be read as an option, so check anyway.
_CONTAINER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

# $1: a marker only this terminal knows (hex), $2: one of SHELLS, checked
# before. Both are arguments, never spliced into the script. The shell's PID
# and start time (field 22 of /proc/<pid>/stat, after the "(name)" that may
# hold spaces) go to a file named after the marker, so _HANG_UP finds exactly
# this process later (`exec` keeps the PID). Shell builtins only; a pod whose
# /tmp can't be written still gets its terminal.
_WRAPPER = (
    'm=$1; want=$2; '
    '{ read -r st </proc/$$/stat && st=${st##*) } && set -- $st && '
    'echo "$$ ${20}" >"/tmp/.greencompute-terminal-$m.pid"; } 2>/dev/null; '
    'case "$want" in '
    'sh) exec sh -l ;; '
    'bash) command -v bash >/dev/null 2>&1 && exec bash -l; '
    'echo "bash is not installed in this image; starting sh instead." >&2; exec sh -l ;; '
    '*) command -v bash >/dev/null 2>&1 && exec bash -l; exec sh -l ;; '
    'esac'
)

# $1: the marker. SIGHUP to the shell this terminal started, as a hung-up
# terminal would get, then SIGKILL if it ignores that. Only while that PID is
# still the same process (same start time, not a zombie): once the shell has
# gone, its PID may belong to something else of the renter's. Removes the
# PID file either way.
_HANG_UP = (
    'f="/tmp/.greencompute-terminal-$1.pid"; '
    'read -r p t 2>/dev/null <"$f" || exit 0; rm -f "$f"; '
    'case "$p" in ""|*[!0-9]*) exit 0 ;; esac; '
    'case "$t" in ""|*[!0-9]*) exit 0 ;; esac; '
    'ours() { read -r st 2>/dev/null </proc/$p/stat || return 1; st=${st##*) }; set -- $st; '
    '[ "$1" != Z ] && [ "${20}" = "$t" ]; }; '
    'ours || exit 0; '
    'kill -HUP "$p" 2>/dev/null || exit 0; '
    'for _ in 1 2 3; do ours || exit 0; sleep 1; done; '
    'ours && kill -KILL "$p" 2>/dev/null; exit 0'
)


class BusyError(RuntimeError):
    """Too many terminals are open on this machine already."""


def clamp_size(cols: object, rows: object, *, default: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """(cols, rows) within what a terminal can sensibly be; the default for
    anything that isn't a whole number."""
    def num(value: object, fallback: int, top: int) -> int:
        if isinstance(value, bool):
            return fallback
        try:
            n = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError, OverflowError):  # OverflowError: JSON's Infinity
            return fallback
        return max(MIN_SIZE, min(n, top))
    return num(cols, default[0], MAX_COLS), num(rows, default[1], MAX_ROWS)


def exec_argv(container_id: str, marker: str, shell: str) -> list[str]:
    """The docker command for one terminal. A list, run without a shell: the
    container id and marker can't be anything but what they are."""
    if not _CONTAINER_ID.fullmatch(container_id or ""):
        raise ValueError("not a container id")
    if shell not in SHELLS or not re.fullmatch(r"[0-9a-f]{16,64}", marker):
        raise ValueError("bad shell or marker")
    return ["docker", "exec", "-i", "-t", "-u", "0", "-w", "/workspace", "--detach-keys", DETACH_KEYS,
            "-e", "TERM=xterm-256color", "-e", f"GREENCOMPUTE_TERMINAL={marker}",
            container_id, "sh", "-c", _WRAPPER, "sh", marker, shell]


def hang_up_argv(container_id: str, marker: str) -> list[str]:
    return ["docker", "exec", "-u", "0", container_id, "sh", "-c", _HANG_UP, "sh", marker]


def _set_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _exited(pid: int) -> bool:
    """Whether the child has exited, without reaping it: while it is
    unreaped its PID, and so its process group's id, can't be reused."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


class Terminal:
    """One shell in one pod. start(), then read()/write()/resize() from one
    event loop (one reader at a time), then close(), always: it frees the
    machine's slot. Cancel tasks blocked in read()/write() before close()."""

    def __init__(self, container_id: str, *, cols: object = 80, rows: object = 24, shell: str = "auto") -> None:
        self.container_id = container_id
        self.marker = secrets.token_hex(12)
        self.cols, self.rows = clamp_size(cols, rows)
        self._argv = exec_argv(container_id, self.marker, shell)
        self._proc: subprocess.Popen | None = None
        self._fd: int | None = None
        self._slot = False
        self._closed = False
        self._eof = False
        self._hung_up = False
        self._hang_up_lock = threading.Lock()
        self.exit_code: int | None = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self) -> None:
        if not _SLOTS.acquire(blocking=False):
            raise BusyError("this machine has the most terminals open it allows; close one or try again shortly")
        self._slot = True
        try:
            master, slave = pty.openpty()
            try:
                _set_size(slave, self.cols, self.rows)
                # Its own session and process group, so close() can end
                # everything it starts with one signal.
                self._proc = subprocess.Popen(  # noqa: S603
                    self._argv, stdin=slave, stdout=slave, stderr=slave, start_new_session=True, close_fds=True)
            except BaseException:
                os.close(master)
                raise
            finally:
                os.close(slave)
            os.set_blocking(master, False)
            self._fd = master
        except BaseException:
            self._release()
            raise

    async def read(self) -> bytes:
        """The next output, b"" once the shell has ended."""
        while True:
            if self._fd is None:
                return b""
            try:
                chunk = os.read(self._fd, READ_CHUNK)
            except BlockingIOError:
                await self._ready(self._fd, write=False)
                continue
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                chunk = b""  # every end of the terminal is closed: the shell has ended
            self._eof = not chunk
            return chunk

    async def write(self, data: bytes) -> None:
        """Keystrokes; waits while the terminal's input is full, so a large
        paste is fed at the speed the pod reads it. Dropped once it ended."""
        view = memoryview(data)
        while view and self._fd is not None:
            try:
                view = view[os.write(self._fd, view):]
            except BlockingIOError:
                await self._ready(self._fd, write=True)
            except OSError:
                return

    def resize(self, cols: object, rows: object) -> None:
        self.cols, self.rows = clamp_size(cols, rows, default=(self.cols, self.rows))
        if self._fd is None or self._proc is None:
            return
        _set_size(self._fd, self.cols, self.rows)
        # docker exec passes the new size on to the pod when it gets SIGWINCH.
        # In a session of its own it has no controlling terminal, so the
        # kernel won't send it one: we do.
        if not _exited(self._proc.pid):
            try:
                os.kill(self._proc.pid, signal.SIGWINCH)
            except ProcessLookupError:
                pass

    async def close(self) -> int | None:
        """End it: hang up the shell in the pod, kill and reap the docker CLI
        and anything it started, free the slot. Returns the shell's exit code
        when it had ended by itself. Safe to call twice, and if the task is
        cancelled part way."""
        if self._closed:
            return self.exit_code
        self._closed = True
        proc, finished, killed = self._proc, False, False
        try:
            if proc is not None:
                # After the last output the docker CLI may take a moment to exit.
                ended = _exited(proc.pid) or (self._eof and await self._wait_exit(KILL_GRACE_SECONDS))
                # Even when docker exec has ended, the shell in the pod may not
                # have (its detach keys were typed, or dockerd restarted); the
                # check in _HANG_UP makes this safe either way, and it removes
                # the PID file.
                await asyncio.to_thread(self._hang_up)
                if not ended and not await self._wait_exit(KILL_GRACE_SECONDS):
                    self._signal_group(proc.pid, signal.SIGTERM)
                    await self._wait_exit(KILL_GRACE_SECONDS)
                # The leader is unreaped, so the group id is still ours: this
                # catches anything it started that outlived it.
                self._signal_group(proc.pid, signal.SIGKILL)
                killed = True
                code = await asyncio.to_thread(proc.wait)
                self.exit_code = code if ended and code >= 0 else None
            finished = True
        finally:
            if proc is not None and not finished:
                # Cancelled part way (the server is stopping): no more waiting,
                # but nothing left behind either. A cancelled to_thread may
                # never have run its hang-up; this one runs unless it did.
                threading.Thread(target=self._hang_up, daemon=True).start()
                if not killed:  # once killed, a reaping thread may have freed the group id
                    self._signal_group(proc.pid, signal.SIGKILL)
                threading.Thread(target=proc.wait, daemon=True).start()  # reap without blocking the loop
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._release()
        return self.exit_code

    def _hang_up(self) -> None:
        with self._hang_up_lock:  # once, whichever thread gets here first
            if self._hung_up:
                return
            self._hung_up = True
        try:
            subprocess.run(hang_up_argv(self.container_id, self.marker),  # noqa: S603
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=HANG_UP_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError):
            pass  # the docker CLI is killed below either way

    async def _wait_exit(self, seconds: float) -> bool:
        assert self._proc is not None
        deadline = time.monotonic() + seconds
        while not _exited(self._proc.pid):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    async def _ready(fd: int, *, write: bool) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def ready() -> None:
            if not fut.done():
                fut.set_result(None)
        (loop.add_writer if write else loop.add_reader)(fd, ready)
        try:
            await fut
        finally:
            (loop.remove_writer if write else loop.remove_reader)(fd)

    def _release(self) -> None:
        if self._slot:
            self._slot = False
            _SLOTS.release()
