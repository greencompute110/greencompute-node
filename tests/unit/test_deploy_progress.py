"""Customers see each step of a start: the node agent reports PULLING before
a container starts (image download) and STARTING once it runs (SSH coming up,
or a model's weights downloading and loading), instead of one long
"scheduled" until ready. Reports are best effort."""
from types import SimpleNamespace

import pytest

from greencompute_node_agent.application.services import NodeAgentService
from greencompute_node_agent.domain import inference
from greencompute_node_agent.domain.inference import ArtifactBundle, DockerInferenceBackend
from greencompute_protocol import DeploymentState, UnifiedRuntimeRecord, WorkloadKind


def _runtime():
    return UnifiedRuntimeRecord(deployment_id="d1", workload_id="w1", hotkey="h", node_id="n",
                                workload_kind=WorkloadKind.INFERENCE, model_identifier="Qwen/Qwen2.5-7B-Instruct")


def test_model_backend_says_when_the_container_is_up_before_waiting_for_the_model(monkeypatch):
    events = []
    monkeypatch.setattr(inference.subprocess, "run",
                        lambda cmd, *a, **k: (events.append(cmd[1]) if cmd[:2] == ["docker", "run"] else None)
                        or SimpleNamespace(returncode=0, stdout="cid\n", stderr=""))
    monkeypatch.setattr(inference, "gpu_docker_flags", lambda devices: [])
    monkeypatch.setattr(DockerInferenceBackend, "_wait_for_health", lambda self, rt: events.append("health"))
    backend = DockerInferenceBackend(default_image="vllm/vllm-openai:v0.19.1-cu130-ubuntu2404")
    artifact = ArtifactBundle(build_id="b", image="vllm", artifact_uri="x", artifact_digest="x", staged_artifact_path="/tmp")
    backend.start_runtime(_runtime(), artifact, on_started=lambda: events.append("started"))
    assert events == ["run", "started", "health"]


def test_a_failing_progress_callback_does_not_stop_the_start(monkeypatch):
    monkeypatch.setattr(inference.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="cid\n", stderr=""))
    monkeypatch.setattr(inference, "gpu_docker_flags", lambda devices: [])
    monkeypatch.setattr(DockerInferenceBackend, "_wait_for_health", lambda self, rt: None)

    def boom():
        raise RuntimeError("control plane down")

    backend = DockerInferenceBackend(default_image="vllm/vllm-openai:v0.19.1-cu130-ubuntu2404")
    artifact = ArtifactBundle(build_id="b", image="vllm", artifact_uri="x", artifact_digest="x", staged_artifact_path="/tmp")
    assert backend.start_runtime(_runtime(), artifact, on_started=boom).container_id == "cid"


@pytest.fixture
def agent():
    svc = object.__new__(NodeAgentService)
    svc.control_plane = SimpleNamespace(updates=[])
    svc.control_plane.update_deployment_status = lambda update: svc.control_plane.updates.append(update)
    return svc


def test_progress_is_reported_to_the_control_plane(agent):
    agent._report_progress(_runtime(), DeploymentState.PULLING)
    agent._report_progress(_runtime(), DeploymentState.STARTING)
    assert [(u.deployment_id, u.state) for u in agent.control_plane.updates] == [
        ("d1", DeploymentState.PULLING), ("d1", DeploymentState.STARTING)]


def test_a_failed_progress_report_is_swallowed(agent):
    def down(update):
        raise ConnectionError("control plane unreachable")

    agent.control_plane.update_deployment_status = down
    agent._report_progress(_runtime(), DeploymentState.PULLING)  # no exception
