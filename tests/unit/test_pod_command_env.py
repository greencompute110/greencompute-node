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
    text = ENTRYPOINT.read_text()
    start = text.index("run_start_command() {")
    end = text.index("\n}\n", start) + 3
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
    assert log.read_text().strip() == "finished"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_no_command_does_nothing(tmp_path):
    subprocess.run(
        ["bash", "-c", _function_source() + "\nrun_start_command\n"],
        check=True,
        env={"PATH": os.environ["PATH"], "GC_COMMAND_WORKDIR": str(tmp_path / "w")},
        timeout=10,
    )
    assert not (tmp_path / "w").exists()
