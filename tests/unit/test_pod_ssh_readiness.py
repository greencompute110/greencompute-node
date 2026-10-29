"""A pod is `ready` only once its SSH server is confirmed up, checked
host-side through docker (the node can't reach its own public IP: NAT
hairpin), one non-blocking look per reconcile tick.

External API user, 2026-10: pods were reported ready while apt-get was
still installing openssh-server (a 90 s timeout that then reported ready
anyway), and the check matched "sshd" anywhere in the process list, which
the set-up steps (`mkdir -p /run/sshd`, `sed … sshd_config`) also do.
"""
import shutil
import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest

from greencompute_node_agent.application.services import NodeAgentService, _utcnow
from greencompute_node_agent.domain import pod as pod_module
from greencompute_node_agent.domain.pod import ProcessPodBackend, sshd_in_process_list
from greencompute_protocol import DeploymentState, UnifiedRuntimeRecord, WorkloadKind
from greencompute_protocol.ssh_host_keys import clean_ssh_host_keys

HEADER = "UID PID PPID C STIME TTY TIME CMD\n"


# --- only sshd itself counts, not the set-up steps that mention it ------------


@pytest.mark.parametrize("row", [
    "root 1 0 0 10:00 ? 00:00:00 /bin/bash /greencompute-entrypoint.sh",
    "root 40 1 0 10:00 ? 00:00:00 mkdir -p /run/sshd /root/.ssh",
    "root 41 1 0 10:00 ? 00:00:00 sed -i s/^#?Port .*/Port 22/ /etc/ssh/sshd_config",
    "root 42 1 0 10:00 ? 00:00:00 adduser --system --group --home /run/sshd sshd",
    "root 43 1 0 10:00 ? 00:00:00 apt-get install -y -qq openssh-server",
])
def test_set_up_steps_that_mention_sshd_are_not_sshd(row):
    assert sshd_in_process_list(HEADER + row + "\n") is False


@pytest.mark.parametrize("row", [
    "root 50 1 0 10:00 ? 00:00:00 /usr/sbin/sshd",
    "root 50 1 0 10:00 ? 00:00:00 /usr/sbin/sshd -D",
    "root 50 1 0 10:00 ? 00:00:00 sshd: /usr/sbin/sshd [listener] 0 of 10-100 startups",
])
def test_the_daemon_itself_is_recognised(row):
    assert sshd_in_process_list(HEADER + "root 1 0 0 10:00 ? 00:00:00 sleep infinity\n" + row + "\n") is True


def test_a_pid_comm_listing_works_too():
    assert sshd_in_process_list("PID COMMAND\n1 sleep\n50 sshd\n") is True
    assert sshd_in_process_list("PID COMMAND\n1 sleep\n") is False


def test_an_empty_or_header_only_listing_is_not_running():
    assert sshd_in_process_list("") is False
    assert sshd_in_process_list(HEADER) is False


def test_sshd_running_false_on_docker_error(monkeypatch):
    def _boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5.0)
    monkeypatch.setattr(subprocess, "run", _boom)
    assert ProcessPodBackend._sshd_running("c1") is False


# --- one look: up, pending, or failed with the reason -------------------------


def _docker(*, top="", logs="", running="true", restarting="false", restarts="0", exit_code="0", inspect_rc=0):
    """Fake `subprocess.run` answering docker top / inspect / logs."""
    def _run(cmd, **kwargs):
        if cmd[1] == "top":
            return SimpleNamespace(returncode=0, stdout=HEADER + top, stderr="")
        if cmd[1] == "inspect":
            return SimpleNamespace(returncode=inspect_rc, stdout=f"{running} {restarting} {restarts} {exit_code}\n",
                                   stderr="" if inspect_rc == 0 else "Error: No such object: c1")
        if cmd[1] == "logs":
            return SimpleNamespace(returncode=0, stdout=logs, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return _run


def _rt(container="c1"):
    return SimpleNamespace(container_id=container, ssh_port=32000, ssh_host="1.2.3.4")


def test_no_container_means_nothing_to_verify():
    assert ProcessPodBackend().ssh_status(_rt(container=None)) == ("up", None)


def test_pending_while_the_entrypoint_is_still_installing(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _docker(
        top="root 1 0 0 10:00 ? 00:00:00 apt-get install -y -qq openssh-server\n",
        logs="[greencompute] setting up SSH...\n"))
    assert ProcessPodBackend().ssh_status(_rt()) == ("pending", None)


def test_up_once_sshd_runs(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _docker(top="root 50 1 0 10:00 ? 00:00:00 /usr/sbin/sshd\n"))
    assert ProcessPodBackend().ssh_status(_rt()) == ("up", None)


def test_an_image_without_a_package_manager_fails_with_the_reason(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _docker(
        logs="[greencompute] WARN: cannot install openssh-server — unknown package manager\n"
             "[greencompute] SSH setup failed — container will run without SSH\n"))
    status, reason = ProcessPodBackend().ssh_status(_rt())
    assert status == "failed"
    assert "no apt-get, apk, yum or dnf" in reason and "openssh-server preinstalled" in reason


def test_a_failed_package_install_is_not_blamed_on_the_image_family(monkeypatch):
    # apt-get itself failed (mirror down): a transient problem, said as such.
    monkeypatch.setattr(subprocess, "run", _docker(logs="[greencompute] SSH setup failed — container will run without SSH\n"))
    status, reason = ProcessPodBackend().ssh_status(_rt())
    assert status == "failed"
    assert "installing the SSH server in this image failed" in reason
    assert "no apt-get" not in reason


def test_the_entrypoints_sshd_start_warning_is_not_final(monkeypatch):
    # The image's own start script may start sshd a moment later; the
    # deadline decides, not this line.
    monkeypatch.setattr(subprocess, "run", _docker(logs="[greencompute] WARN: sshd failed to start\n"))
    assert ProcessPodBackend().ssh_status(_rt()) == ("pending", None)


def test_a_container_that_keeps_exiting_fails_even_while_sshd_is_momentarily_up(monkeypatch):
    # Docker restarts it (--restart unless-stopped); sshd shows up for a
    # moment in each cycle. That must never read as ready.
    monkeypatch.setattr(subprocess, "run", _docker(
        top="root 50 1 0 10:00 ? 00:00:00 /usr/sbin/sshd\n", restarts="3", exit_code="137",
        logs="killed\n"))
    status, reason = ProcessPodBackend().ssh_status(_rt())
    assert status == "failed"
    assert "exits right after it starts (exit code 137, restarted 3 time(s))" in reason
    assert "Last output: killed" in reason


def test_a_container_removed_outside_the_platform_is_a_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _docker(inspect_rc=1))
    status, reason = ProcessPodBackend().ssh_status(_rt())
    assert (status, reason) == ("failed", "the container is gone (removed outside the platform)")


def test_docker_being_unavailable_is_pending_not_a_verdict(monkeypatch):
    def _boom(cmd, **kwargs):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", _boom)
    assert ProcessPodBackend().ssh_status(_rt()) == ("pending", None)


# --- the node agent: ready only when SSH is up, one look per tick -------------


def _agent(status, *, timeout=600.0):
    svc = object.__new__(NodeAgentService)
    svc.settings = SimpleNamespace(pod_ssh_ready_timeout_seconds=timeout)
    svc.pod_backend = SimpleNamespace(looks=0, stopped=[])
    def ssh_status(rt):
        svc.pod_backend.looks += 1
        return status() if callable(status) else status
    svc.pod_backend.ssh_status = ssh_status
    svc.pod_backend.stop_pod = lambda rt: svc.pod_backend.stopped.append(rt.container_id) or rt
    svc.repository = SimpleNamespace(runtimes={})
    svc.repository.upsert_runtime = lambda rt: svc.repository.runtimes.__setitem__(rt.deployment_id, rt) or rt
    svc.control_plane = SimpleNamespace(updates=[])
    svc.control_plane.update_deployment_status = lambda u: svc.control_plane.updates.append(u)
    svc.gpu_allocator = SimpleNamespace(release=lambda dep: None)
    svc._volume_records = {"d1": SimpleNamespace(path="/vol/d1")}
    svc.volume_manager = SimpleNamespace(deleted=[])
    svc.volume_manager.delete_volume = lambda v: svc.volume_manager.deleted.append(v.path)
    return svc


def _pod_runtime():
    return UnifiedRuntimeRecord(
        deployment_id="d1", workload_id="w1", hotkey="h", node_id="n", workload_kind=WorkloadKind.POD,
        container_id="c1", ssh_host="1.2.3.4", ssh_port=30123, ssh_username="root", status="ready",
    )


def test_a_warm_image_is_ready_in_the_same_tick():
    svc = _agent(("up", None))
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.READY
    assert update.endpoint == "ssh://root@1.2.3.4:30123"
    assert svc.repository.runtimes["d1"].status == "ready"
    assert svc.pod_backend.stopped == [] and svc.volume_manager.deleted == []


def test_a_cold_image_stays_starting_until_sshd_is_seen():
    svc = _agent(("pending", None))
    svc._finish_pod_start(_pod_runtime())
    assert svc.control_plane.updates == []  # nothing reported yet: still `starting` to the customer
    rt = svc.repository.runtimes["d1"]
    assert (rt.status, rt.current_stage) == ("starting", NodeAgentService.AWAIT_SSH)
    assert "ssh_deadline" in rt.metadata
    # Next tick: sshd is up → ready, no blocking wait anywhere.
    svc.pod_backend.ssh_status = lambda rt: ("up", None)
    svc._check_pod_ssh(rt, now=_utcnow() + timedelta(seconds=5))
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.READY


def test_looks_are_rate_limited_to_once_every_few_seconds():
    svc = _agent(("pending", None))
    svc._finish_pod_start(_pod_runtime())
    rt = svc.repository.runtimes["d1"]
    svc._check_pod_ssh(rt, now=_utcnow() + timedelta(seconds=1))
    assert svc.pod_backend.looks == 1  # the second look, 1 s later, was skipped
    svc._check_pod_ssh(rt, now=_utcnow() + timedelta(seconds=4))
    assert svc.pod_backend.looks == 2


def test_a_pod_whose_ssh_never_comes_up_fails_at_the_deadline_instead_of_reporting_ready():
    svc = _agent(("pending", None), timeout=600.0)
    svc._finish_pod_start(_pod_runtime())
    rt = svc.repository.runtimes["d1"]
    svc._check_pod_ssh(rt, now=_utcnow() + timedelta(seconds=599))
    assert svc.control_plane.updates == []
    svc._check_pod_ssh(rt, now=_utcnow() + timedelta(seconds=601))
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.FAILED
    assert "did not come up within 10 minutes" in update.error
    assert svc.pod_backend.stopped == ["c1"]  # the container is removed, not left running
    assert svc.volume_manager.deleted == ["/vol/d1"]  # and its disk
    failed = svc.repository.runtimes["d1"]
    assert (failed.status, failed.container_id, failed.failure_class) == ("failed", None, "pod_start_failure")


def test_a_pod_that_can_never_get_ssh_fails_at_once_with_the_reason():
    svc = _agent(("failed", "the SSH server could not be installed in this image"))
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.FAILED
    assert update.error == "SSH setup failed: the SSH server could not be installed in this image"
    assert svc.pod_backend.stopped == ["c1"]


def test_a_backend_without_ssh_reports_ready_directly():
    svc = _agent(("pending", None))
    del svc.pod_backend.ssh_status  # the stub backend
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.READY


# --- the pod's SSH host keys go out with the ready report (request #9) --------

ED = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFDpfaWJh7dYs3/G3VaELhN36Tj0pY7oUj+Afdy7rk93"
EC = ("ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEbBtMZtLYqFI5v+Dgiy8Hpi5QQd"
      "8pxTsAiqLATbh2bTDlNRKAcOaPO6FkAzthlAm4jQrj+KCUW2Ou65ZDcBAs8=")


def test_host_keys_are_read_with_one_exec_and_cleaned(monkeypatch):
    seen = []

    def _out(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return f"{ED} root@c1\n{EC} root@c1\n".encode()
    monkeypatch.setattr(pod_module, "bounded_output", _out)
    assert ProcessPodBackend().ssh_host_keys(_rt()) == [ED, EC]
    assert seen == [(["docker", "exec", "c1", "sh", "-c", ProcessPodBackend.HOST_KEYS_SCRIPT],
                     {"limit": 65536, "timeout": 5.0})]


def test_no_keys_when_the_pod_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(pod_module, "bounded_output", lambda cmd, **kw: None)
    assert ProcessPodBackend().ssh_host_keys(_rt()) == []
    assert ProcessPodBackend().ssh_host_keys(_rt(container=None)) == []
    monkeypatch.setattr(pod_module, "bounded_output", lambda cmd, **kw: b"junk\n")
    assert ProcessPodBackend().ssh_host_keys(_rt()) == []


# The output comes from the renter's container: it must never fill our memory.

@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_bounded_output_returns_what_a_quick_command_prints():
    assert pod_module.bounded_output(["sh", "-c", "echo hi"], limit=100, timeout=5) == b"hi\n"


@pytest.mark.skipif(shutil.which("yes") is None, reason="needs yes")
def test_bounded_output_stops_reading_endless_output():
    import time as _time

    started = _time.monotonic()
    assert pod_module.bounded_output(["yes"], limit=65536, timeout=5) is None
    assert _time.monotonic() - started < 2


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_bounded_output_gives_up_at_the_timeout_and_on_failure():
    assert pod_module.bounded_output(["sh", "-c", "sleep 5"], limit=100, timeout=0.3) is None
    assert pod_module.bounded_output(["sh", "-c", "echo x; exit 3"], limit=100, timeout=5) is None
    assert pod_module.bounded_output(["/nonexistent/binary"], limit=100, timeout=5) is None


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("awk") is None, reason="needs sh and awk")
def test_the_script_lists_the_keys_sshd_is_configured_with(tmp_path):
    """`sshd -T` names the keys sshd serves, wherever sshd_config puts them."""
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "host_ed25519.pub").write_text(f"{ED} root@c1\n")
    (keys / "host_ecdsa.pub").write_text(f"{EC} root@c1\n")
    fake_sshd = tmp_path / "bin" / "sshd"
    fake_sshd.parent.mkdir()
    fake_sshd.write_text(f"#!/bin/sh\necho port 22\necho hostkey {keys}/host_ed25519\n"
                         f"echo hostkey {keys}/host_ecdsa\necho hostkey {keys}/missing\n")
    fake_sshd.chmod(0o755)
    out = subprocess.run(["sh", "-c", ProcessPodBackend.HOST_KEYS_SCRIPT], capture_output=True, text=True,
                         env={"PATH": f"{fake_sshd.parent}:/usr/bin:/bin"}, timeout=10)
    assert out.returncode == 0
    assert clean_ssh_host_keys(out.stdout) == [ED, EC]


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_the_script_still_succeeds_without_sshd(tmp_path):
    out = subprocess.run(["sh", "-c", ProcessPodBackend.HOST_KEYS_SCRIPT], capture_output=True, text=True,
                         env={"PATH": "/nonexistent:/usr/bin:/bin"}, timeout=10)
    assert out.returncode == 0  # falls back to the usual files; whatever exists is printed


def test_ready_report_carries_the_host_keys():
    svc = _agent(("up", None))
    svc.pod_backend.ssh_host_keys = lambda rt: [ED]
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.ssh_host_keys == [ED]
    assert svc.repository.runtimes["d1"].metadata["ssh_host_keys"] == [ED]


def test_a_resumed_pod_whose_keys_are_known_is_not_asked_again():
    svc = _agent(("up", None))
    svc.gpu_allocator = SimpleNamespace(allocate_specific=lambda dep, devices: devices, release=lambda dep: None)
    svc.pod_backend.resume_pod = lambda rt: rt.model_copy(update={"status": "ready", "current_stage": "ready"})
    asked = []
    svc.pod_backend.ssh_host_keys = lambda rt: asked.append(rt) or [EC]
    suspended = _pod_runtime().model_copy(update={
        "status": "suspended", "metadata": {"gpu_devices": [0], "gpu_count": 1, "ssh_host_keys": [ED]},
    })
    svc._resume_runtime(suspended)
    [update] = svc.control_plane.updates
    assert update.ssh_host_keys == [ED] and asked == []


def test_a_failure_to_read_keys_never_holds_up_ready():
    svc = _agent(("up", None))

    def _boom(rt):
        raise RuntimeError("docker exec hung")
    svc.pod_backend.ssh_host_keys = _boom
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.READY and update.ssh_host_keys is None


def test_a_resumed_pod_that_cannot_be_asked_keeps_its_known_keys():
    svc = _agent(("up", None))
    svc.gpu_allocator = SimpleNamespace(allocate_specific=lambda dep, devices: devices, release=lambda dep: None)
    svc.pod_backend.resume_pod = lambda rt: rt.model_copy(update={"status": "ready", "current_stage": "ready"})
    svc.pod_backend.ssh_host_keys = lambda rt: None
    suspended = _pod_runtime().model_copy(update={
        "status": "suspended", "metadata": {"gpu_devices": [0], "gpu_count": 1, "ssh_host_keys": [ED]},
    })
    svc._resume_runtime(suspended)
    [update] = svc.control_plane.updates
    assert update.ssh_host_keys == [ED]


def test_a_backend_that_cannot_read_keys_sends_none_so_stored_keys_stay():
    svc = _agent(("up", None))  # the fake backend has no ssh_host_keys
    svc._finish_pod_start(_pod_runtime())
    [update] = svc.control_plane.updates
    assert update.ssh_host_keys is None


def test_a_resumed_pod_reports_its_keys_even_if_it_started_before_keys_were_sent():
    svc = _agent(("up", None))
    svc.gpu_allocator = SimpleNamespace(allocate_specific=lambda dep, devices: devices, release=lambda dep: None)
    svc.pod_backend.resume_pod = lambda rt: rt.model_copy(update={"status": "ready", "current_stage": "ready"})
    svc.pod_backend.ssh_host_keys = lambda rt: [ED]
    suspended = _pod_runtime().model_copy(update={
        "status": "suspended", "endpoint": "ssh://root@1.2.3.4:30123",
        "metadata": {"gpu_devices": [0], "gpu_count": 1},  # no ssh_host_keys: started before
    })
    svc._resume_runtime(suspended)
    [update] = svc.control_plane.updates
    assert update.state == DeploymentState.READY and update.ssh_host_keys == [ED]
