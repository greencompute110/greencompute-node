"""The website's terminal (request #34): a root shell in a running pod under a
real pseudo-terminal, behind the node secret, that leaves nothing running on
the machine when it closes.

`docker exec` is swapped for a local shell, so this runs anywhere; the shell
scripts that run in the pod are run here with sh."""
import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
from greencompute_node_agent.domain import pod_terminal

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


async def read_until(term, needle: bytes, timeout: float = 10.0) -> bytes:
    seen = b""
    deadline = time.monotonic() + timeout
    while needle not in seen:
        left = deadline - time.monotonic()
        assert left > 0, f"never saw {needle!r} in {seen[-500:]!r}"
        chunk = await asyncio.wait_for(term.read(), timeout=left)
        assert chunk, f"the shell ended before {needle!r}: {seen[-500:]!r}"
        seen += chunk
    return seen


def gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return True
    return state == "Z"  # dead, waiting for whoever adopted it to reap it


def wait_gone(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while not gone(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@pytest.fixture
def local(monkeypatch):
    """Terminals run `argv` here instead of docker exec; hang-ups are recorded."""
    hang_ups = []
    state = {"argv": ["sh", "-i"]}
    monkeypatch.setattr(pod_terminal, "exec_argv", lambda cid, marker, shell: list(state["argv"]))
    monkeypatch.setattr(pod_terminal, "hang_up_argv", lambda cid, marker: hang_ups.append((cid, marker)) or ["true"])
    monkeypatch.setattr(pod_terminal, "KILL_GRACE_SECONDS", 0.5)
    return state, hang_ups


# --- the docker command and the scripts that run in the pod ----------------------


def test_the_docker_command_is_a_fixed_list_with_a_real_terminal():
    marker = "ab" * 12
    argv = pod_terminal.exec_argv("3f2a9c1d", marker, "auto")
    assert argv[:8] == ["docker", "exec", "-i", "-t", "-u", "0", "-w", "/workspace"]
    assert argv[argv.index("--detach-keys") + 1] == pod_terminal.DETACH_KEYS  # ctrl-p isn't held back
    assert "TERM=xterm-256color" in argv and f"GREENCOMPUTE_TERMINAL={marker}" in argv
    # The script is a constant; the marker and the shell are arguments after it.
    assert argv[-7:] == ["3f2a9c1d", "sh", "-c", pod_terminal._WRAPPER, "sh", marker, "auto"]


@pytest.mark.parametrize("container_id", ["", "-v", "--privileged", "c1; rm -rf /", "c1 c2", "$(id)", "a" * 200])
def test_nothing_but_a_container_id_reaches_docker(container_id):
    with pytest.raises(ValueError):
        pod_terminal.exec_argv(container_id, "ab" * 12, "auto")


@pytest.mark.parametrize("shell, marker", [
    ("bash; touch /tmp/pwned", "ab" * 12), ("zsh", "ab" * 12), ("", "ab" * 12),
    ("auto", "$(id)"), ("auto", "short"), ("auto", "AB" * 12),
])
def test_only_known_shells_and_our_own_markers(shell, marker):
    with pytest.raises(ValueError):
        pod_terminal.exec_argv("c1", marker, shell)


def test_a_terminal_for_an_unknown_shell_is_never_built():
    with pytest.raises(ValueError):
        pod_terminal.Terminal("c1", shell="bash -c 'touch /tmp/pwned'")


@pytest.fixture
def fake_shells(tmp_path):
    """A PATH holding stand-ins for bash and sh that say how they were run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("bash", "sh"):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\necho "{name} $*"\n')
        script.chmod(0o755)
    return bin_dir


def _wrapper(bin_dir: Path, shell: str, marker: str) -> str:
    real_sh = shutil.which("sh")
    return subprocess.run([real_sh, "-c", pod_terminal._WRAPPER, "sh", marker, shell], capture_output=True,
                          text=True, env={"PATH": str(bin_dir)}, timeout=10, check=True).stdout


@needs_sh
@pytest.mark.parametrize("shell, has_bash, expected", [
    ("auto", True, "bash -l"),
    ("auto", False, "sh -l"),
    ("bash", True, "bash -l"),
    ("bash", False, "sh -l"),
    ("sh", True, "sh -l"),
    # Refused before it gets here, and even then only ever a case label.
    ("x; echo injected", True, "bash -l"),
])
def test_the_wrapper_starts_a_login_shell_bash_when_there_is_one(fake_shells, shell, has_bash, expected):
    if not has_bash:
        (fake_shells / "bash").unlink()
    marker = os.urandom(12).hex()
    pid_file = Path(f"/tmp/.greencompute-terminal-{marker}.pid")
    try:
        out = _wrapper(fake_shells, shell, marker)
        assert out.strip().splitlines()[-1] == expected and "injected" not in out
        pid, started = pid_file.read_text().split()  # where the hang-up finds the shell: PID and start time
        assert pid.isdigit() and started.isdigit()
    finally:
        pid_file.unlink(missing_ok=True)


def _hang_up(marker: str) -> None:
    subprocess.run([shutil.which("sh"), "-c", pod_terminal._HANG_UP, "sh", marker], timeout=15, check=True)


def _start_time(pid: int) -> str:
    """Field 22 of /proc/<pid>/stat, as the wrapper records it."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


@needs_sh
@pytest.mark.skipif(shutil.which("sleep") is None, reason="needs sleep")
def test_the_hang_up_sends_sighup_to_the_shell_it_started():
    marker = os.urandom(12).hex()
    shell = subprocess.Popen(["sleep", "60"])
    pid_file = Path(f"/tmp/.greencompute-terminal-{marker}.pid")
    pid_file.write_text(f"{shell.pid} {_start_time(shell.pid)}\n")
    try:
        _hang_up(marker)
        assert shell.wait(timeout=5) == -1  # SIGHUP
        assert not pid_file.exists()
    finally:
        shell.kill()
        shell.wait()
        pid_file.unlink(missing_ok=True)


@needs_sh
@pytest.mark.skipif(shutil.which("sleep") is None, reason="needs sleep")
def test_the_hang_up_kills_a_shell_that_ignores_it_and_spares_anything_else():
    marker = os.urandom(12).hex()
    # Its own process group, so its `sleep` goes with it below (in a pod, the
    # kernel hangs up the foreground job when the shell dies; not here).
    stubborn = subprocess.Popen(["sh", "-c", "trap '' HUP; sleep 60"], start_new_session=True)
    # A PID that has been reused by something else of the renter's: another start time.
    other = subprocess.Popen(["sleep", "60"])
    pid_file = Path(f"/tmp/.greencompute-terminal-{marker}.pid")
    try:
        time.sleep(0.2)  # let the trap be set
        pid_file.write_text(f"{stubborn.pid} {_start_time(stubborn.pid)}\n")
        _hang_up(marker)
        assert stubborn.wait(timeout=5) == -9
        pid_file.write_text(f"{other.pid} {int(_start_time(other.pid)) - 1}\n")
        _hang_up(marker)
        assert other.poll() is None  # not ours: left alone
        pid_file.write_text(f"{other.pid}\n")
        _hang_up(marker)
        assert other.poll() is None  # no start time: left alone
        _hang_up(marker)  # no pid file at all: nothing to do, no error
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(stubborn.pid, signal.SIGKILL)
        other.kill()
        for p in (stubborn, other):
            p.wait()
        pid_file.unlink(missing_ok=True)


# --- a session ---------------------------------------------------------------------


@needs_sh
def test_a_shell_runs_and_its_exit_code_comes_back(local):
    async def go():
        term = pod_terminal.Terminal("c1", cols=80, rows=24)
        term.start()
        try:
            await term.write(b"echo $((6*7))\n")
            await read_until(term, b"42")
            await term.write(b"exit 7\n")
            while await term.read():
                pass
        finally:
            code = await term.close()
        return code
    assert run(go()) == 7


@needs_sh
@pytest.mark.skipif(shutil.which("stty") is None, reason="needs stty")
def test_resizing_reaches_the_terminal_and_signals_docker(local):
    async def go():
        term = pod_terminal.Terminal("c1", cols=80, rows=24)
        term.start()
        try:
            await term.write(b"stty size\n")
            await read_until(term, b"24 80")
            await term.write(b"trap 'echo got-winch' WINCH\n")
            await asyncio.sleep(0.3)
            term.resize(132, 43)
            # docker exec has no controlling terminal; only our own SIGWINCH tells it.
            await read_until(term, b"got-winch")
            await term.write(b"stty size\n")
            await read_until(term, b"43 132")
            term.resize(10**6, -5)  # clamped, not an error
            assert (term.cols, term.rows) == (pod_terminal.MAX_COLS, pod_terminal.MIN_SIZE)
        finally:
            await term.close()
    run(go())


@needs_sh
@pytest.mark.skipif(shutil.which("stty") is None or shutil.which("head") is None, reason="needs stty and head")
def test_a_large_paste_is_fed_as_fast_as_the_pod_reads_it(local):
    state, _ = local
    state["argv"] = ["sh", "-c", "stty raw -echo; head -c 300000 >/dev/null; echo all-read"]

    async def go():
        term = pod_terminal.Terminal("c1")
        term.start()
        try:
            await asyncio.sleep(0.3)  # raw mode first
            await term.write(b"x" * 300000)  # far over the terminal's buffer
            await read_until(term, b"all-read")
        finally:
            await term.close()
    run(go())


@needs_sh
def test_closing_leaves_nothing_running_on_the_machine(local):
    """The docker CLI (here: sh) and anything it started are killed and reaped,
    the pod's shell is hung up, and the machine's slot is freed."""
    state, hang_ups = local
    state["argv"] = ["sh", "-c", "trap '' TERM HUP; sleep 300 & echo child=$!; sleep 300"]

    async def go():
        term = pod_terminal.Terminal("c1")
        term.start()
        try:
            out = await read_until(term, b"child=")
            while not re.search(rb"child=\d+\s", out):
                out += await term.read()
            child = int(re.search(rb"child=(\d+)", out).group(1))
        finally:
            leader = term.pid
            code = await term.close()
        return leader, child, code, term
    leader, child, code, term = run(go())
    assert wait_gone(leader) and wait_gone(child)
    assert code is None  # it didn't end by itself
    assert hang_ups == [("c1", term.marker)]
    assert pod_terminal._SLOTS._value == pod_terminal.MAX_TERMINALS
    assert run(term.close()) is None  # twice is fine


@needs_sh
def test_closing_still_cleans_up_when_cancelled_part_way(local):
    """The server stopping cancels the session mid-close: no waiting then,
    but the docker CLI is still killed and the pod's shell still hung up."""
    state, hang_ups = local
    state["argv"] = ["sh", "-c", "trap '' TERM HUP; sleep 300"]

    async def go():
        term = pod_terminal.Terminal("c1")
        term.start()
        await asyncio.sleep(0.2)
        closing = asyncio.create_task(term.close())
        await asyncio.sleep(0)  # into its first wait
        closing.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await closing
        return term
    term = run(go())
    assert wait_gone(term.pid)
    deadline = time.monotonic() + 5
    while not hang_ups and time.monotonic() < deadline:
        time.sleep(0.05)
    assert hang_ups == [("c1", term.marker)]
    assert pod_terminal._SLOTS._value == pod_terminal.MAX_TERMINALS


@needs_sh
def test_a_shell_that_ended_by_itself_still_has_the_pod_checked(local):
    """docker exec can end with the pod's shell still running (its detach
    keys, a dockerd restart); the hang-up checks, and clears the PID file."""
    state, hang_ups = local
    state["argv"] = ["sh", "-c", "exit 0"]

    async def go():
        term = pod_terminal.Terminal("c1")
        term.start()
        try:
            while await term.read():
                pass
        finally:
            code = await term.close()
        return code, term.marker
    code, marker = run(go())
    assert code == 0 and hang_ups == [("c1", marker)]


def test_a_busy_machine_says_so_and_a_failed_start_frees_its_slot(local, monkeypatch):
    taken = [pod_terminal._SLOTS.acquire(blocking=False) for _ in range(pod_terminal.MAX_TERMINALS)]
    try:
        assert all(taken)
        with pytest.raises(pod_terminal.BusyError):
            pod_terminal.Terminal("c1").start()
    finally:
        for _ in taken:
            pod_terminal._SLOTS.release()
    state, _ = local
    state["argv"] = ["/nonexistent/docker"]
    with pytest.raises(OSError):
        pod_terminal.Terminal("c1").start()
    assert pod_terminal._SLOTS._value == pod_terminal.MAX_TERMINALS
