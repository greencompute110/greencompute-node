"""A model deployed with no context limit is served at the longest context
that fits, instead of its native maximum: Llama 3.1 8B on a 24 GB card failed
to start (2026-10-16) because 128k tokens needed 16 GB of KV cache and only
~5 GB was left. A model that still fails says why on the first line."""
import pytest

from greencompute_node_agent.domain import inference
from greencompute_node_agent.domain.inference import (
    ArtifactBundle,
    DockerInferenceBackend,
    _startup_error_summary,
    _vllm_image_auto_fits_context,
)
from greencompute_protocol import UnifiedRuntimeRecord, WorkloadKind


class _Captured(Exception):
    pass


def _vllm_args(monkeypatch, image, payload=None, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """The `docker run` command start_runtime builds, without running it."""
    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            raise _Captured(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    monkeypatch.setattr(inference, "gpu_docker_flags", lambda devices: [])
    backend = DockerInferenceBackend(default_image=image)
    runtime = UnifiedRuntimeRecord(deployment_id="d1", workload_id="w1", hotkey="h", node_id="n",
                                   workload_kind=WorkloadKind.INFERENCE, model_identifier=model)
    artifact = ArtifactBundle(build_id="b", image="vllm", artifact_uri="x", artifact_digest="x",
                              staged_artifact_path="/tmp", payload=payload or {})
    with pytest.raises(_Captured) as exc:
        backend.start_runtime(runtime, artifact)
    cmd = exc.value.args[0]
    return cmd[cmd.index(image) + 1:]


def _flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def test_current_vllm_picks_the_context_that_fits(monkeypatch):
    args = _vllm_args(monkeypatch, "vllm/vllm-openai:v0.19.1-cu130-ubuntu2404")
    assert _flag(args, "--max-model-len") == "auto"


def test_older_vllm_gets_a_cap_that_fits_a_24gb_card(monkeypatch):
    args = _vllm_args(monkeypatch, "vllm/vllm-openai:v0.8.5")
    assert _flag(args, "--max-model-len") == "32768"


def test_a_limit_set_on_the_workload_wins(monkeypatch):
    args = _vllm_args(monkeypatch, "vllm/vllm-openai:v0.19.1-cu130-ubuntu2404", payload={"max_model_len": 8192})
    assert _flag(args, "--max-model-len") == "8192"


def test_vision_models_keep_their_cap(monkeypatch):
    args = _vllm_args(monkeypatch, "vllm/vllm-openai:v0.19.1-cu130-ubuntu2404", model="Qwen/Qwen2-VL-7B-Instruct")
    assert _flag(args, "--max-model-len") == "16384"


def test_version_check():
    assert _vllm_image_auto_fits_context("vllm/vllm-openai:v0.19.1-cu130-ubuntu2404")
    assert _vllm_image_auto_fits_context("vllm/vllm-openai:v0.20.0")
    assert not _vllm_image_auto_fits_context("vllm/vllm-openai:v0.8.5")
    assert _vllm_image_auto_fits_context("vllm/vllm-openai:latest")


def test_the_reason_a_model_failed_comes_first():
    logs = """(APIServer pid=1) INFO 10-16 15:19:58 [utils.py:299]  version 0.19.1
(EngineCore pid=344) ERROR 10-16 15:21:36 [core.py:1108] EngineCore failed to start.
(EngineCore pid=344) ERROR 10-16 15:21:36 [core.py:1108] ValueError: To serve at least one request with the models's max seq len (131072), (16.0 GiB KV cache is needed, which is larger than the available KV cache memory (5.05 GiB).
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}"""
    assert _startup_error_summary(logs).startswith("ValueError: To serve at least one request")
    assert _startup_error_summary("no errors here") is None
