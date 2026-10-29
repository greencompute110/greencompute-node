"""A pod can run a start command and get environment variables from its
workload, so an unattended job needs no SSH session after boot."""
import base64
import os
import pathlib
import shutil
import subprocess
import time

import pytest

from greencompute_node_agent.domain.ssh import pod_start_env
from greencompute_node_agent.transport.redaction import REDACT_PLACEHOLDER, redact

ENTRYPOINT = pathlib.Path(__file__).resolve().parents[2] / "infra" / "docker" / "entrypoint.sh"


def test_env_and_encoded_command_for_docker():
    env = pod_start_env({"env": {"HF_HOME": "/workspace/hf", "SEED": 7}, "command": "python train.py --tag \"a b\"\necho done"})
    assert env["HF_HOME"] == "/workspace/hf" and env["SEED"] == "7"
    assert base64.b64decode(env["GC_START_COMMAND_B64"]).decode() == "python train.py --tag \"a b\"\necho done"


@pytest.mark.parametrize("meta", [None, {}, {"command": "  "}, {"env": "not-a-dict"}])
def test_nothing_to_pass(meta):
    assert pod_start_env(meta) == {}


def test_env_never_appears_in_observability_responses():
    out = redact({"metadata": {"env_vars": {"OPENAI_API_KEY": "sk-x", "GC_START_COMMAND_B64": "abc"}, "image": "x"}})
    assert out["metadata"]["env_vars"] == REDACT_PLACEHOLDER and out["metadata"]["image"] == "x"


def _function_source() -> str:
    """keep_log_small, supervise_start_command and run_start_command."""
    text = ENTRYPOINT.read_text()
    start = text.index("keep_log_small() {")
    end = text.index("\n}\n", text.index("run_start_command() {")) + 3
    return text[start:end]


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("base64") is None, reason="needs bash and base64")
def test_entrypoint_runs_the_command_in_the_background_from_the_workdir(tmp_path):
    command = "echo \"hello $GREETING_NAME, it's here\" > result.txt\necho finished"
    subprocess.run(
        ["bash", "-c", _function_source() + "\nrun_start_command\n"],
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "GREETING_NAME": "agent",
            "GC_START_COMMAND_B64": base64.b64encode(command.encode()).decode(),
            "GC_COMMAND_WORKDIR": str(tmp_path / "workspace"),
            "GC_COMMAND_FILE": str(tmp_path / "start.sh"),
        },
        timeout=10,
    )
    result = tmp_path / "workspace" / "result.txt"
    log = tmp_path / "workspace" / "greencompute-command.log"
    for _ in range(50):  # it runs in the background
        if result.exists() and log.exists() and "finished" in log.read_text():
            break
        time.sleep(0.1)
    assert result.read_text() == "hello agent, it's here\n"
    assert log.read_text().splitlines()[0] == "finished"  # then the exit, logged by the supervisor


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_no_command_does_nothing(tmp_path):
    subprocess.run(
        ["bash", "-c", _function_source() + "\nrun_start_command\n"],
        check=True,
        env={"PATH": os.environ["PATH"], "GC_COMMAND_WORKDIR": str(tmp_path / "w")},
        timeout=10,
    )
    assert not (tmp_path / "w").exists()


# --- the deployment's name reaches the pod (#27) ------------------------------

def test_every_pod_gets_its_id_and_name():
    env = pod_start_env({"env": {"POOL": "x"}}, "0123456789ab-cdef", "rig-07")
    assert env == {"POOL": "x", "GC_DEPLOYMENT_ID": "0123456789ab-cdef", "GC_DEPLOYMENT_NAME": "rig-07"}


def test_without_a_name_the_short_id_is_used():
    assert pod_start_env({}, "0123456789ab-cdef")["GC_DEPLOYMENT_NAME"] == "01234567"


def test_the_lease_name_is_kept_on_the_runtime_for_the_pod(monkeypatch):
    from types import SimpleNamespace

    from greencompute_node_agent.application.services import NodeAgentService
    from greencompute_protocol import LeaseAssignment, WorkloadKind, WorkloadSpec

    svc = object.__new__(NodeAgentService)
    workload = WorkloadSpec(name="miner", image="img", kind="pod")
    svc.control_plane = SimpleNamespace(get_workload=lambda wid: workload)
    svc.repository = SimpleNamespace(upsert_runtime=lambda rt: rt)
    started = {}
    monkeypatch.setattr(NodeAgentService, "_start_pod_runtime",
                        lambda self, runtime, wl: started.setdefault("runtime", runtime))
    lease = LeaseAssignment(deployment_id="d1", workload_id=workload.workload_id, hotkey="h", node_id="n",
                            deployment_name="rig-07")
    svc._reconcile_workload(lease)
    assert started["runtime"].metadata["deployment_name"] == "rig-07"
    assert started["runtime"].workload_kind == WorkloadKind.POD


def test_a_workload_set_name_is_kept_unless_the_rental_has_its_own():
    meta = {"env": {"GC_DEPLOYMENT_NAME": "from-workload"}}
    assert pod_start_env(meta, "0123456789ab")["GC_DEPLOYMENT_NAME"] == "from-workload"
    assert pod_start_env(meta, "0123456789ab", "rig-07")["GC_DEPLOYMENT_NAME"] == "rig-07"


# --- restart_policy: run the start command again when it exits (#7) ---------


def _start(tmp_path, command: str, policy: str | None, **extra) -> pathlib.Path:
    env = {**extra,
        "PATH": os.environ["PATH"],
        "GC_START_COMMAND_B64": base64.b64encode(command.encode()).decode(),
        "GC_COMMAND_WORKDIR": str(tmp_path / "workspace"),
        "GC_COMMAND_FILE": str(tmp_path / "start.sh"),
        "GC_RESTART_DELAY_SECONDS": "1",
    }
    if policy:
        env["GC_RESTART_POLICY"] = policy
    # The comment puts tmp_path in the supervisor's command line, for no_strays.
    subprocess.run(["bash", "-c", f"# {tmp_path}\nset -e\n" + _function_source() + "\nrun_start_command\n"],
                   check=True, env=env, timeout=10)
    return tmp_path / "workspace"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _wait_for(predicate, seconds=15.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _runs(workspace: pathlib.Path) -> int:
    f = workspace / "runs"
    return len(f.read_text().splitlines()) if f.exists() else 0


@pytest.fixture
def no_strays(tmp_path):
    yield
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)  # supervisors and their scripts
    for name in ("kids", "kid"):  # background jobs the test commands started
        f = tmp_path / "workspace" / name
        for pid in (f.read_text().split() if f.exists() else []):
            try:
                os.kill(int(pid), 9)
            except (ProcessLookupError, ValueError):
                pass


needs_bash = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("base64") is None,
                                reason="needs bash and base64")


@needs_bash
def test_without_a_policy_a_failing_command_runs_once_and_the_exit_is_logged(tmp_path, no_strays):
    ws = _start(tmp_path, "echo run >> runs; exit 3", None)
    log = ws / "greencompute-command.log"
    assert _wait_for(lambda: log.exists() and "exited with code 3" in log.read_text())
    time.sleep(1.5)
    assert _runs(ws) == 1
    assert "restarting" not in log.read_text()


@needs_bash
def test_on_failure_restarts_until_it_succeeds(tmp_path, no_strays):
    # Fails twice, then exits 0: three runs (waits 1 s, then 2 s), then it stays down.
    ws = _start(tmp_path, 'echo run >> runs; [ "$(wc -l < runs)" -ge 3 ]', "on-failure")
    log = ws / "greencompute-command.log"
    assert _wait_for(lambda: log.exists() and "exited with code 0" in log.read_text())
    text = log.read_text()
    assert _runs(ws) == 3
    assert "restarting it in 1s (restart 1, restart_policy on-failure)" in text
    assert "restarting it in 2s (restart 2, restart_policy on-failure)" in text
    time.sleep(1.5)
    assert _runs(ws) == 3


@needs_bash
def test_always_restarts_even_after_a_clean_exit(tmp_path, no_strays):
    # The job stops its own supervisor after the third run so the test ends.
    ws = _start(tmp_path, 'echo run >> runs; [ "$(wc -l < runs)" -ge 3 ] && kill $PPID; exit 0', "always")
    assert _wait_for(lambda: _runs(ws) >= 3)
    assert "restart_policy always" in (ws / "greencompute-command.log").read_text()


def test_the_policy_reaches_the_pod_only_with_a_command():
    meta = {"command": "./mine.sh", "restart_policy": "always"}
    assert pod_start_env(meta)["GC_RESTART_POLICY"] == "always"
    assert "GC_RESTART_POLICY" not in pod_start_env({"command": "./mine.sh", "restart_policy": "never"})
    assert "GC_RESTART_POLICY" not in pod_start_env({"restart_policy": "always"})


@needs_bash
def test_a_restart_waits_for_what_the_last_run_started_and_kills_nothing(tmp_path, no_strays):
    # Run 1 starts a job in the background and fails at once. The job is left
    # to finish (a launcher's miner must not be killed) and only then is the
    # command run again, so two copies never run side by side.
    ws = _start(tmp_path, '(sleep 2; echo job-done >> runs) & echo run >> runs; [ "$(grep -c run runs)" -ge 3 ]',
                "on-failure")
    assert _wait_for(lambda: (ws / "runs").exists() and (ws / "runs").read_text().count("run\n") >= 2, seconds=20)
    lines = (ws / "runs").read_text().split()
    assert lines[:3] == ["run", "job-done", "run"]
    assert "waiting for the processes it started to end" in (ws / "greencompute-command.log").read_text()


@needs_bash
def test_never_leaves_a_job_the_command_started_alone(tmp_path, no_strays):
    ws = _start(tmp_path, "sleep 300 & echo $! > kid; exit 0", None)
    assert _wait_for(lambda: (ws / "kid").exists() and "exited with code 0" in (ws / "greencompute-command.log").read_text())
    time.sleep(1.5)
    assert _alive(int((ws / "kid").read_text()))


@needs_bash
def test_nothing_starts_while_the_pause_file_exists(tmp_path, no_strays):
    (tmp_path / "workspace").mkdir()
    pause = tmp_path / "workspace" / "greencompute-command.pause"
    pause.touch()
    ws = _start(tmp_path, "echo run >> runs; exit 1", "always")
    log = ws / "greencompute-command.log"
    assert _wait_for(lambda: log.exists() and "not starting the command while" in log.read_text())
    time.sleep(1.5)
    assert _runs(ws) == 0  # not even the first run: also holds across a resume
    pause.unlink()
    assert _wait_for(lambda: _runs(ws) >= 1, seconds=20)
    pause.touch()  # and between restarts
    time.sleep(7)
    settled = _runs(ws)
    time.sleep(3)
    assert _runs(ws) == settled


@needs_bash
def test_a_big_log_is_cut_while_the_command_runs(tmp_path, no_strays):
    ws = _start(tmp_path, "head -c 5000 /dev/zero | tr '\\0' x; echo; sleep 3; echo after", None,
                GC_LOG_LIMIT_BYTES="1000", GC_LOG_CHECK_SECONDS="1")
    log = ws / "greencompute-command.log"
    assert _wait_for(lambda: log.exists() and "exited with code 0" in log.read_text())
    assert (ws / "greencompute-command.log.1").stat().st_size > 5000
    text = log.read_text()
    assert text.startswith("after") and "xxxx" not in text


@needs_bash
def test_a_job_that_is_only_paused_is_not_counted_as_exited(tmp_path, no_strays):
    import signal

    ws = _start(tmp_path, "echo $$ > me; echo run >> runs; sleep 2; exit 1", "on-failure")
    assert _wait_for(lambda: (ws / "me").exists() and (ws / "me").read_text().strip())
    job = int((ws / "me").read_text())
    os.kill(job, signal.SIGSTOP)
    time.sleep(3)
    log = ws / "greencompute-command.log"
    assert _runs(ws) == 1 and "exited" not in (log.read_text() if log.exists() else "")
    os.kill(job, signal.SIGCONT)
    assert _wait_for(lambda: _runs(ws) >= 2, seconds=15)


def _pid1_branch() -> str:
    text = ENTRYPOINT.read_text()
    start = text.index('    echo "[greencompute] no CMD specified')
    return text[start:text.rindex("fi")]


@pytest.mark.skipif(os.geteuid() != 0 or shutil.which("unshare") is None, reason="needs root and unshare")
def test_as_pid_1_a_stop_gives_the_job_sigterm_and_time_to_finish(tmp_path):
    """docker stop signals PID 1 only; it passes SIGTERM on and waits."""
    import signal

    job = (f"( trap 'echo term > {tmp_path}/term; sleep 2; echo done > {tmp_path}/done; exit 0' TERM; "
           "while :; do sleep 0.2; done ) &\n")
    proc = subprocess.Popen(["unshare", "--pid", "--fork", "--mount-proc", "bash", "-c", job + _pid1_branch()])
    try:
        assert _wait_for(lambda: pathlib.Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().strip(), 5)
        pid1 = int(pathlib.Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()[0])
        time.sleep(0.5)
        started = time.monotonic()
        os.kill(pid1, signal.SIGTERM)
        assert proc.wait(timeout=12) == 0
        took = time.monotonic() - started
    finally:
        proc.kill()
    assert (tmp_path / "term").exists() and (tmp_path / "done").exists()  # it finished its own way
    assert 1.5 < took < 9


def test_only_restart_policy_decides_not_a_leftover_env_value():
    env = pod_start_env({"command": "./mine.sh", "env": {"GC_RESTART_POLICY": "always"}})
    assert "GC_RESTART_POLICY" not in env
