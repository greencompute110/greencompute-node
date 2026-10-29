"""GPU stats per card and a pod's recent output (request #17: "the only way
to check a card's health is to SSH in and run nvidia-smi")."""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from greencompute_node_agent.domain import pod as pod_module
from greencompute_node_agent.domain import pod_logs, pod_stats

SMI = (
    "3, GPU-aaaa, NVIDIA GeForce RTX 4090, 97, 20480, 24564, 71, 441.20, 450.00, 2745, 10501, 62\n"
    "1, GPU-bbbb, NVIDIA GeForce RTX 4090, 95, 20000, 24564, 68, 430.50, 450.00, 2730, 10501, [N/A]\n"
)


def test_each_gpu_is_reported_as_the_pod_numbers_it(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout=SMI, stderr="")
    monkeypatch.setattr(pod_stats.subprocess, "run", run)
    gpus = pod_stats.gpu_details([3, 1])
    assert seen[0][1] == "--id=1,3"
    # Machine GPUs 1 and 3 are the pod's 0 and 1.
    assert [(g["index"], g["uuid"]) for g in gpus] == [(0, "GPU-bbbb"), (1, "GPU-aaaa")]
    card = gpus[1]
    assert card == {
        "index": 1, "uuid": "GPU-aaaa", "name": "NVIDIA GeForce RTX 4090", "util_pct": 97.0,
        "memory_used_mb": 20480.0, "memory_total_mb": 24564.0, "temperature_c": 71.0, "power_w": 441.2,
        "power_limit_w": 450.0, "sm_clock_mhz": 2745.0, "mem_clock_mhz": 10501.0, "fan_pct": 62.0,
    }
    assert gpus[0]["fan_pct"] is None  # "[N/A]": not reported, not an error


def test_totals_stay_for_the_dashboard(monkeypatch):
    monkeypatch.setattr(pod_stats.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=SMI if cmd[0] == "nvidia-smi" else "",
                                                          stderr=""))
    monkeypatch.setattr(pod_module, "bounded_output", lambda cmd, **kw: None)
    out = pod_stats.collect_pod_stats("c1", [1, 3])
    assert out["gpu_util_pct"] == 96.0 and out["vram_used_mb"] == 40480.0 and out["vram_total_mb"] == 49128.0
    assert len(out["gpus"]) == 2


def test_no_gpus_or_no_nvidia_smi_is_no_gpu_section(monkeypatch):
    assert pod_stats.gpu_details([]) == []

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(pod_stats.subprocess, "run", missing)
    assert pod_stats.gpu_details([0]) == []


def test_disk_usage_is_read_with_a_cap(monkeypatch):
    calls = []

    def bounded(cmd, **kw):
        calls.append((cmd, kw))
        return (b"Filesystem 1B-blocks Used Available Use% Mounted on\n"
                b"/dev/loop0 53687091200 10737418240 42949672960 20% /workspace\n")
    monkeypatch.setattr(pod_module, "bounded_output", bounded)
    used, total = pod_stats._disk_usage("c1", "/workspace")
    assert (round(used, 2), round(total, 2)) == (10.74, 53.69)
    assert calls[0][1]["limit"] == 16384
    monkeypatch.setattr(pod_module, "bounded_output", lambda cmd, **kw: None)
    assert pod_stats._disk_usage("c1", "/workspace") == (None, None)


def test_logs_are_the_last_lines_of_each_source(monkeypatch):
    calls = []

    def tail(cmd, **kw):
        calls.append((cmd, kw))
        if cmd[1] == "exec":
            return "".join(f"hashrate {i}\n" for i in range(500)).encode(), True
        return b"[greencompute] SSH ready on port 22\nminer started\n", True
    monkeypatch.setattr(pod_logs, "bounded_tail", tail)
    out = pod_logs.collect_pod_logs("c1", lines=3)
    assert out["command_log"] == "hashrate 497\nhashrate 498\nhashrate 499"
    assert out["container_log"] == "[greencompute] SSH ready on port 22\nminer started"
    assert out["command_log_note"] is None and out["container_log_note"] is None
    exec_call, logs_call = calls
    # As root, whatever the image's USER; only the end of the file.
    assert exec_call[0] == ["docker", "exec", "-u", "0", "c1", "tail", "-c", str(pod_logs.KEEP_BYTES), pod_logs.COMMAND_LOG]
    assert logs_call[0] == ["docker", "logs", "--tail", "3", "c1"] and logs_call[1]["merge_stderr"] is True


def test_notes_say_why_a_source_is_empty(monkeypatch):
    monkeypatch.setattr(pod_logs, "bounded_tail", lambda cmd, **kw: None)
    out = pod_logs.collect_pod_logs("c1", lines=999999, has_command=False)
    assert out["lines"] == pod_logs.MAX_LINES
    assert out["command_log"] is None and "no start command" in out["command_log_note"]
    assert out["container_log"] is None and out["container_log_note"] == "couldn't be read"
    out = pod_logs.collect_pod_logs("c1")
    assert out["command_log_note"] == "not written yet, or couldn't be read (the image needs `tail`)"
    # Lines too long to reach the end of: said, not shown as empty.
    monkeypatch.setattr(pod_logs, "bounded_tail", lambda cmd, **kw: (b"x" * 10, False))
    out = pod_logs.collect_pod_logs("c1", lines=50)
    assert out["container_log"] is None and "too long" in out["container_log_note"]


def test_a_failed_gpu_is_named_and_the_others_still_reported(monkeypatch):
    one_row = SMI.splitlines()[1] + "\n"  # GPU 1 answers, GPU 3 has fallen off the bus
    monkeypatch.setattr(pod_stats.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        returncode=15, stdout=one_row + "Unable to determine the device handle for gpu 0000:41:00.0: Unknown Error\n",
        stderr=""))
    gpus = pod_stats.gpu_details([1, 3])
    assert gpus[0]["uuid"] == "GPU-bbbb" and gpus[0]["util_pct"] == 95.0
    assert gpus[1] == {"index": 1, "error": "this GPU didn't answer nvidia-smi"}
    # nvidia-smi failing outright says nothing per card.
    monkeypatch.setattr(pod_stats.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        returncode=9, stdout="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n", stderr=""))
    assert pod_stats.gpu_details([1, 3]) == []


def test_last_lines_in_a_failure_reason_never_come_from_the_middle(monkeypatch):
    monkeypatch.setattr(pod_module, "bounded_tail", lambda cmd, **kw: (b"x" * 4096, False))
    assert pod_module._last_log_lines("c1") == "(the last lines are too long to show)"
    monkeypatch.setattr(pod_module, "bounded_tail", lambda cmd, **kw: (b"\x1b[31mKilled\x1b[0m\n", True))
    assert pod_module._last_log_lines("c1") == "Killed"


def test_terminal_output_reads_as_text():
    raw = "\x1b[32mINFO\x1b[0m loaded\n 10%|#   |\r 55%|#####|\r100%|##########|\nbell\x07 done\n"
    assert pod_logs.clean_terminal_text(raw) == "INFO loaded\n100%|##########|\nbell done\n"


@pytest.mark.skipif(shutil.which("yes") is None or shutil.which("sh") is None, reason="needs yes and sh")
def test_bounded_tail_keeps_the_end_and_never_the_whole_stream():
    import time as _time

    data, complete = pod_module.bounded_tail(["sh", "-c", "seq 1 100000"], keep=20, read_limit=10**7, timeout=5)
    assert complete and data.endswith(b"99999\n100000\n") and len(data) == 20
    started = _time.monotonic()
    data, complete = pod_module.bounded_tail(["yes"], keep=100, read_limit=1024 * 1024, timeout=5)
    assert complete is False and len(data) == 100 and _time.monotonic() - started < 2
    assert pod_module.bounded_tail(["sh", "-c", "sleep 5"], keep=10, read_limit=100, timeout=0.3) is None
    assert pod_module.bounded_tail(["sh", "-c", "echo x; exit 2"], keep=10, read_limit=100, timeout=5) is None


def test_a_gpu_that_isnt_the_pods_is_left_out_and_vram_totals_match(monkeypatch):
    smi = (SMI + "7, GPU-zzzz, NVIDIA GeForce RTX 4090, 1, 1, 24564, 30, 20, 450, 210, 405, 30\n").replace(
        "20000, 24564", "20000, [N/A]")
    monkeypatch.setattr(pod_stats.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=smi, stderr=""))
    monkeypatch.setattr(pod_module, "bounded_output", lambda cmd, **kw: None)
    out = pod_stats.collect_pod_stats("c1", [1, 3])
    assert [g["uuid"] for g in out["gpus"]] == ["GPU-bbbb", "GPU-aaaa"]  # GPU 7 isn't theirs
    # GPU-bbbb has no total, so neither of its numbers counts: 20480 / 24564.
    assert (out["vram_used_mb"], out["vram_total_mb"]) == (20480.0, 24564.0)


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_bounded_output_can_include_stderr():
    assert pod_module.bounded_output(["sh", "-c", "echo out; echo err >&2"], limit=100, timeout=5,
                                     merge_stderr=True) == b"out\nerr\n"
    assert pod_module.bounded_output(["sh", "-c", "echo out; echo err >&2"], limit=100, timeout=5) == b"out\n"
