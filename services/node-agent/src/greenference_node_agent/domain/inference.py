"""Inference runtime backends — ProcessInferenceBackend, DockerInferenceBackend, and StagedArtifactStore."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, Field

from greenference_protocol import ChatCompletionRequest, ChatCompletionResponse, UnifiedRuntimeRecord
from greenference_node_agent.domain.model_backend import ModelBackendError, create_text_generation_backend

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


class InferenceRuntimeError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str, stage: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.stage = stage


class ArtifactBundle(BaseModel):
    build_id: str
    image: str
    artifact_uri: str
    artifact_digest: str
    registry_manifest_uri: str | None = None
    context_manifest_uri: str | None = None
    dockerfile_path: str | None = None
    staged_artifact_path: str
    payload: dict[str, Any] = Field(default_factory=dict)


class InferenceBackend:
    """Abstract inference runtime backend."""

    def start_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
    ) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def stop_runtime(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        raise NotImplementedError

    def invoke(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        raise NotImplementedError

    def stream(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> Iterator[str]:
        raise NotImplementedError


class LocalArtifactInferenceBackend(InferenceBackend):
    def __init__(self, *, backend_name: str = "local-artifact-backend") -> None:
        self.backend_name = backend_name

    def start_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
    ) -> UnifiedRuntimeRecord:
        manifest = self._runtime_manifest(artifact.payload, artifact.image)
        backend = create_text_generation_backend(manifest, image=artifact.image, allow_fallback=True)
        return runtime.model_copy(update={
            "backend_name": self.backend_name,
            "runtime_mode": "fallback",
            "model_identifier": backend.model_identifier,
            "metadata": {
                **runtime.metadata,
                "runtime_manifest": manifest,
                "backend_started": True,
            },
            "updated_at": utcnow(),
        })

    def stop_runtime(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        return runtime.model_copy(update={
            "metadata": {
                **runtime.metadata,
                "backend_started": False,
                "terminated_at": utcnow().isoformat(),
            },
            "updated_at": utcnow(),
        })

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        started = bool(runtime.metadata.get("backend_started"))
        if not started:
            raise InferenceRuntimeError(
                "runtime backend not started",
                failure_class="runtime_start_failure",
                stage="health_check",
            )
        manifest = self._runtime_manifest_from_runtime(runtime)
        return create_text_generation_backend(
            manifest,
            image=runtime.image or "unknown-image",
            allow_fallback=True,
        ).health()

    def invoke(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        manifest = self._runtime_manifest_from_runtime(runtime)
        backend = create_text_generation_backend(
            manifest,
            image=runtime.image or "unknown-image",
            allow_fallback=True,
        )
        return ChatCompletionResponse(
            model=payload.model,
            content=backend.generate_text(payload),
            deployment_id=runtime.deployment_id,
            routed_hotkey=runtime.hotkey,
        )

    def stream(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> Iterator[str]:
        manifest = self._runtime_manifest_from_runtime(runtime)
        backend = create_text_generation_backend(
            manifest,
            image=runtime.image or "unknown-image",
            allow_fallback=True,
        )
        response = ChatCompletionResponse(
            model=payload.model,
            content=backend.generate_text(payload.model_copy(update={"stream": False})),
            deployment_id=runtime.deployment_id,
            routed_hotkey=runtime.hotkey,
        )
        words = list(backend.stream_tokens(payload.model_copy(update={"stream": False})))
        for index, word in enumerate(words):
            chunk = {
                "id": response.id,
                "object": "chat.completion.chunk",
                "model": response.model,
                "deployment_id": response.deployment_id,
                "routed_hotkey": response.routed_hotkey,
                "choices": [{"index": 0, "delta": {"content": word if index == 0 else f" {word}"}}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield (
            "data: "
            + json.dumps(
                {
                    "id": response.id,
                    "object": "chat.completion.chunk",
                    "model": response.model,
                    "deployment_id": response.deployment_id,
                    "routed_hotkey": response.routed_hotkey,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    def _runtime_manifest(self, payload: dict[str, Any], image: str) -> dict[str, Any]:
        manifest = payload.get("runtime_manifest")
        if not isinstance(manifest, dict):
            raise InferenceRuntimeError(
                "runtime manifest missing from staged artifact",
                failure_class="artifact_validation_failure",
                stage="prepare_runtime",
            )
        try:
            create_text_generation_backend(manifest, image=image, allow_fallback=True)
        except ModelBackendError as exc:
            raise InferenceRuntimeError(
                f"runtime manifest invalid: {exc}",
                failure_class="artifact_validation_failure",
                stage="prepare_runtime",
            ) from exc
        return manifest

    def _runtime_manifest_from_runtime(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        manifest = runtime.metadata.get("runtime_manifest")
        if not isinstance(manifest, dict):
            raise InferenceRuntimeError(
                "runtime manifest missing from runtime state",
                failure_class="artifact_validation_failure",
                stage="invoke_inference",
            )
        return manifest


class ProcessInferenceBackend(InferenceBackend):
    def __init__(
        self,
        *,
        backend_name: str = "process-local-runtime",
        health_timeout_seconds: float = 10.0,
        fallback_backend: InferenceBackend | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.health_timeout_seconds = health_timeout_seconds
        self.fallback_backend = fallback_backend

    def start_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
    ) -> UnifiedRuntimeRecord:
        try:
            return self._start_process_runtime(runtime, artifact)
        except PermissionError as exc:
            return self._start_fallback_runtime(runtime, artifact, reason=str(exc))
        except InferenceRuntimeError as exc:
            if exc.stage not in {"start_inference_backend", "health_check"}:
                raise
            return self._start_fallback_runtime(runtime, artifact, reason=str(exc))

    def _start_fallback_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
        *,
        reason: str,
    ) -> UnifiedRuntimeRecord:
        if self.fallback_backend is None:
            raise InferenceRuntimeError(
                reason,
                failure_class="runtime_start_failure",
                stage="start_inference_backend",
            )
        if runtime.process_id is not None:
            _terminate_pid(runtime.process_id)
        runtime = runtime.model_copy(update={"process_id": None, "runtime_url": None})
        fallback_runtime = self.fallback_backend.start_runtime(runtime, artifact)
        return fallback_runtime.model_copy(update={
            "backend_name": "fallback-local-artifact-backend",
            "runtime_mode": "fallback",
            "metadata": {
                **fallback_runtime.metadata,
                "fallback_reason": reason,
            },
            "updated_at": utcnow(),
        })

    def _start_process_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
    ) -> UnifiedRuntimeRecord:
        if runtime.runtime_dir is None:
            raise InferenceRuntimeError(
                "runtime directory missing before start",
                failure_class="runtime_start_failure",
                stage="start_inference_backend",
            )
        existing_pid = runtime.process_id
        if existing_pid is not None and _pid_alive(existing_pid):
            return runtime.model_copy(update={
                "backend_name": self.backend_name,
                "metadata": {
                    **runtime.metadata,
                    "backend_started": True,
                    "reused_process": True,
                },
                "updated_at": utcnow(),
            })

        runtime_dir = Path(runtime.runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        payload_path = runtime_dir / "artifact-payload.json"
        payload_path.write_text(json.dumps(artifact.payload, sort_keys=True), encoding="utf-8")

        port = _choose_free_port()
        runtime_url = f"http://{_docker_host()}:{port}"
        command = [
            sys.executable,
            "-m",
            "greenference_node_agent.runtime_server",
            "--port",
            str(port),
            "--deployment-id",
            runtime.deployment_id,
            "--hotkey",
            runtime.hotkey,
            "--image",
            artifact.image,
            "--payload-path",
            str(payload_path),
        ]
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(runtime_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        model_identifier = str(artifact.payload.get("runtime_manifest", {}).get("model_identifier") or artifact.image)
        runtime = runtime.model_copy(update={
            "process_id": process.pid,
            "runtime_url": runtime_url,
            "backend_name": self.backend_name,
            "runtime_mode": "process",
            "model_identifier": model_identifier,
            "metadata": {
                **runtime.metadata,
                "backend_started": True,
                "payload_path": str(payload_path),
                "runtime_command": command,
                "runtime_port": port,
                "runtime_manifest": artifact.payload.get("runtime_manifest", {}),
            },
            "updated_at": utcnow(),
        })
        self._wait_for_health(runtime)
        return runtime

    def stop_runtime(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        if runtime.runtime_mode == "fallback" and self.fallback_backend is not None:
            stopped = self.fallback_backend.stop_runtime(runtime)
            return stopped.model_copy(update={
                "backend_name": runtime.backend_name,
                "updated_at": utcnow(),
            })
        if runtime.process_id is not None:
            _terminate_pid(runtime.process_id)
        return runtime.model_copy(update={
            "metadata": {
                **runtime.metadata,
                "backend_started": False,
                "terminated_at": utcnow().isoformat(),
            },
            "process_id": None,
            "runtime_url": None,
            "updated_at": utcnow(),
        })

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        if runtime.runtime_mode == "fallback" and self.fallback_backend is not None:
            return self.fallback_backend.health(runtime)
        return self._request_json(runtime, "/healthz", None, failure_class="health_check_failure")

    def invoke(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if runtime.runtime_mode == "fallback" and self.fallback_backend is not None:
            return self.fallback_backend.invoke(runtime, payload)
        response_payload = self._request_json(
            runtime,
            "/v1/chat/completions",
            payload.model_dump(mode="json"),
            failure_class="inference_execution_failure",
        )
        return ChatCompletionResponse(**response_payload)

    def stream(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> Iterator[str]:
        if runtime.runtime_mode == "fallback" and self.fallback_backend is not None:
            yield from self.fallback_backend.stream(runtime, payload)
            return
        if runtime.runtime_url is None:
            raise InferenceRuntimeError(
                "runtime URL missing during stream invoke",
                failure_class="inference_execution_failure",
                stage="stream_inference",
            )
        target = f"{runtime.runtime_url}/v1/chat/completions"
        body = json.dumps(payload.model_dump(mode="json")).encode()
        req = request.Request(target, data=body, method="POST")
        req.add_header("content-type", "application/json")
        try:
            with request.urlopen(req, timeout=5.0) as response:  # noqa: S310
                while True:
                    line = response.readline()
                    if not line:
                        break
                    yield line.decode()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise InferenceRuntimeError(
                f"runtime stream request failed: {exc}",
                failure_class="inference_execution_failure",
                stage="stream_inference",
            ) from exc

    def _wait_for_health(self, runtime: UnifiedRuntimeRecord) -> None:
        deadline = time.time() + self.health_timeout_seconds
        last_error: str | None = None
        while time.time() < deadline:
            try:
                payload = self._request_json(runtime, "/healthz", None, failure_class="health_check_failure")
                if payload.get("status") == "ok":
                    return
            except InferenceRuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise InferenceRuntimeError(
            f"runtime health check timed out: {last_error or 'no health response'}",
            failure_class="health_check_failure",
            stage="health_check",
        )

    def _request_json(
        self,
        runtime: UnifiedRuntimeRecord,
        path: str,
        payload: dict[str, Any] | None,
        *,
        failure_class: str,
    ) -> dict[str, Any]:
        if runtime.runtime_url is None:
            raise InferenceRuntimeError(
                "runtime URL missing",
                failure_class=failure_class,
                stage="health_check" if path == "/healthz" else "invoke_inference",
            )
        target = f"{runtime.runtime_url}{path}"
        encoded = None if payload is None else json.dumps(payload).encode()
        req = request.Request(target, data=encoded, method="GET" if encoded is None else "POST")
        if encoded is not None:
            req.add_header("content-type", "application/json")
        try:
            with request.urlopen(req, timeout=5.0) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise InferenceRuntimeError(
                f"runtime request failed for {path}: {exc}",
                failure_class=failure_class,
                stage="health_check" if path == "/healthz" else "invoke_inference",
            ) from exc


class DockerInferenceBackend(InferenceBackend):
    """Launches inference as a Docker container running vLLM or diffusion server."""

    DIFFUSION_DEFAULT_IMAGE = "ghcr.io/greenference/diffusion:latest"

    def __init__(
        self,
        *,
        backend_name: str = "docker-vllm-backend",
        health_timeout_seconds: float = 600.0,
        default_image: str = "vllm/vllm-openai:v0.7.3",
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        self.backend_name = backend_name
        self.health_timeout_seconds = health_timeout_seconds
        self.default_image = default_image
        self.gpu_memory_utilization = gpu_memory_utilization

    def _is_diffusion(self, artifact: ArtifactBundle) -> bool:
        runtime_manifest = artifact.payload.get("runtime_manifest", {})
        return str(runtime_manifest.get("runtime_kind", "")).lower() == "diffusion"

    def start_runtime(
        self,
        runtime: UnifiedRuntimeRecord,
        artifact: ArtifactBundle,
    ) -> UnifiedRuntimeRecord:
        model_id = runtime.model_identifier or artifact.image
        port = _choose_free_port()
        container_name = f"greenference-inf-{runtime.deployment_id[:12]}"
        is_diffusion = self._is_diffusion(artifact)
        image = artifact.payload.get("docker_image") or (self.DIFFUSION_DEFAULT_IMAGE if is_diffusion else self.default_image)

        cmd: list[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--shm-size", "8g",
            "-p", f"{port}:8000",
        ]

        # GPU allocation — use specific devices if assigned, otherwise all
        gpu_devices: list[int] | None = runtime.metadata.get("gpu_devices")
        if gpu_devices:
            device_str = ",".join(str(d) for d in gpu_devices)
            cmd += ["--gpus", f'"device={device_str}"']
        else:
            cmd += ["--gpus", "all"]

        # Pass HuggingFace token from miner env (miner operator provides their own credentials)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        if hf_token:
            cmd += ["-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"]
            cmd += ["-e", f"HF_TOKEN={hf_token}"]

        # Mount HuggingFace cache for faster repeated loads
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if Path(hf_cache).exists():
            cmd += ["-v", f"{hf_cache}:/root/.cache/huggingface"]

        cmd.append(image)

        if is_diffusion:
            # Diffusion server arguments
            cmd += [
                "--model", model_id,
                "--host", "0.0.0.0",
                "--port", "8000",
            ]
            logger.info("starting diffusion container for %s: model=%s image=%s port=%d", runtime.deployment_id, model_id, image, port)
        else:
            # vLLM serve arguments
            cmd += [
                "--model", model_id,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            ]

            # Optional: tensor parallel for multi-GPU
            tp_size = artifact.payload.get("tensor_parallel_size")
            if tp_size and int(tp_size) > 1:
                cmd += ["--tensor-parallel-size", str(tp_size)]

            # Optional: max model length
            max_model_len = artifact.payload.get("max_model_len")
            if max_model_len:
                cmd += ["--max-model-len", str(max_model_len)]

            logger.info("starting vLLM container for %s: model=%s image=%s port=%d", runtime.deployment_id, model_id, image, port)

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=600.0,
            )
            if result.returncode != 0:
                raise InferenceRuntimeError(
                    f"docker run failed: {result.stderr}",
                    failure_class="runtime_start_failure",
                    stage="start_inference_backend",
                )
            container_id = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise InferenceRuntimeError(
                f"docker not available: {exc}",
                failure_class="runtime_start_failure",
                stage="start_inference_backend",
            ) from exc

        runtime_url = f"http://{_docker_host()}:{port}"
        runtime = runtime.model_copy(update={
            "container_id": container_id,
            "runtime_url": runtime_url,
            "process_id": None,
            "runtime_mode": "docker",
            "backend_name": self.backend_name,
            "model_identifier": model_id,
            "metadata": {
                **runtime.metadata,
                "container_name": container_name,
                "docker_image": image,
                "runtime_port": port,
                "backend_started": True,
                "started_at": utcnow().isoformat(),
            },
            "updated_at": utcnow(),
        })

        # vLLM takes time to load — wait for health
        try:
            self._wait_for_health(runtime)
        except InferenceRuntimeError:
            # Clean up the container on health check failure
            self.stop_runtime(runtime)
            raise
        return runtime

    def stop_runtime(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        if runtime.container_id:
            try:
                subprocess.run(  # noqa: S603
                    ["docker", "rm", "-f", runtime.container_id],  # noqa: S607
                    capture_output=True,
                    timeout=30.0,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return runtime.model_copy(update={
            "container_id": None,
            "runtime_url": None,
            "metadata": {
                **runtime.metadata,
                "backend_started": False,
                "terminated_at": utcnow().isoformat(),
            },
            "updated_at": utcnow(),
        })

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        if runtime.runtime_url is None:
            return {"status": "not_started", "healthy": False}
        try:
            target = f"{runtime.runtime_url}/health"
            req = request.Request(target, method="GET")
            with request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                return {"status": "ok", "healthy": True, "backend": self.backend_name}
        except (HTTPError, URLError, TimeoutError):
            return {"status": "unhealthy", "healthy": False}

    def invoke(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if runtime.runtime_url is None:
            raise InferenceRuntimeError(
                "runtime URL missing",
                failure_class="inference_execution_failure",
                stage="invoke_inference",
            )
        target = f"{runtime.runtime_url}/v1/chat/completions"
        body = json.dumps(payload.model_dump(mode="json")).encode()
        req = request.Request(target, data=body, method="POST")
        req.add_header("content-type", "application/json")
        try:
            with request.urlopen(req, timeout=60.0) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
            # vLLM returns OpenAI-compatible response — extract content
            content = ""
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
            return ChatCompletionResponse(
                model=data.get("model", payload.model),
                content=content,
                deployment_id=runtime.deployment_id,
                routed_hotkey=runtime.hotkey,
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise InferenceRuntimeError(
                f"inference request failed: {exc}",
                failure_class="inference_execution_failure",
                stage="invoke_inference",
            ) from exc

    def stream(
        self,
        runtime: UnifiedRuntimeRecord,
        payload: ChatCompletionRequest,
    ) -> Iterator[str]:
        if runtime.runtime_url is None:
            raise InferenceRuntimeError(
                "runtime URL missing during stream",
                failure_class="inference_execution_failure",
                stage="stream_inference",
            )
        target = f"{runtime.runtime_url}/v1/chat/completions"
        stream_payload = payload.model_dump(mode="json")
        stream_payload["stream"] = True
        body = json.dumps(stream_payload).encode()
        req = request.Request(target, data=body, method="POST")
        req.add_header("content-type", "application/json")
        try:
            with request.urlopen(req, timeout=60.0) as resp:  # noqa: S310
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line.decode()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise InferenceRuntimeError(
                f"stream request failed: {exc}",
                failure_class="inference_execution_failure",
                stage="stream_inference",
            ) from exc

    def _wait_for_health(self, runtime: UnifiedRuntimeRecord) -> None:
        """Wait for vLLM to become healthy. Model loading can take minutes."""
        deadline = time.time() + self.health_timeout_seconds
        last_error: str | None = None
        check_interval = 2.0  # vLLM takes a while, no need to poll fast
        while time.time() < deadline:
            # First check container is still running
            if runtime.container_id:
                try:
                    result = subprocess.run(  # noqa: S603
                        ["docker", "inspect", "--format", "{{.State.Status}}", runtime.container_id],  # noqa: S607
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                    )
                    if result.returncode == 0 and result.stdout.strip() == "exited":
                        logs = subprocess.run(  # noqa: S603
                            ["docker", "logs", "--tail", "20", runtime.container_id],  # noqa: S607
                            capture_output=True,
                            text=True,
                            timeout=5.0,
                        )
                        raise InferenceRuntimeError(
                            f"container exited prematurely: {logs.stderr or logs.stdout}",
                            failure_class="runtime_start_failure",
                            stage="health_check",
                        )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            # Then check HTTP health
            try:
                target = f"{runtime.runtime_url}/health"
                req = request.Request(target, method="GET")
                with request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                    if resp.status == 200:
                        logger.info("vLLM healthy for %s after %.0fs", runtime.deployment_id, time.time() - (deadline - self.health_timeout_seconds))
                        return
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(check_interval)
        raise InferenceRuntimeError(
            f"vLLM health check timed out after {self.health_timeout_seconds}s: {last_error or 'no response'}",
            failure_class="health_check_failure",
            stage="health_check",
        )


class StagedArtifactStore:
    def __init__(self, artifact_cache_dir: str) -> None:
        self.cache_dir = Path(artifact_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def stage_artifact(
        self,
        *,
        deployment_id: str,
        build_id: str,
        image: str,
        artifact_uri: str,
        artifact_digest: str,
        registry_manifest_uri: str | None,
        context_manifest_uri: str | None,
        dockerfile_path: str | None,
        payload: dict[str, Any],
    ) -> ArtifactBundle:
        if not artifact_uri:
            raise InferenceRuntimeError(
                "published build artifact_uri is missing",
                failure_class="artifact_fetch_failure",
                stage="pull_artifact",
            )
        if not artifact_digest:
            raise InferenceRuntimeError(
                "published build artifact_digest is missing",
                failure_class="artifact_validation_failure",
                stage="pull_artifact",
            )
        artifact_path = self.cache_dir / f"{deployment_id}.artifact.json"
        artifact_body = {
            "build_id": build_id,
            "image": image,
            "artifact_uri": artifact_uri,
            "artifact_digest": artifact_digest,
            "registry_manifest_uri": registry_manifest_uri,
            "context_manifest_uri": context_manifest_uri,
            "dockerfile_path": dockerfile_path,
            "payload": payload,
            "staged_at": utcnow().isoformat(),
        }
        artifact_path.write_text(json.dumps(artifact_body, sort_keys=True), encoding="utf-8")
        return ArtifactBundle(
            build_id=build_id,
            image=image,
            artifact_uri=artifact_uri,
            artifact_digest=artifact_digest,
            registry_manifest_uri=registry_manifest_uri,
            context_manifest_uri=context_manifest_uri,
            dockerfile_path=dockerfile_path,
            staged_artifact_path=str(artifact_path),
            payload=payload,
        )

    def load_staged_artifact(self, staged_artifact_path: str) -> ArtifactBundle:
        artifact_path = Path(staged_artifact_path)
        if not artifact_path.exists():
            raise InferenceRuntimeError(
                f"staged artifact missing: {staged_artifact_path}",
                failure_class="artifact_fetch_failure",
                stage="pull_artifact",
            )
        body = json.loads(artifact_path.read_text(encoding="utf-8"))
        digest = body.get("artifact_digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise InferenceRuntimeError(
                "staged artifact digest is invalid",
                failure_class="artifact_validation_failure",
                stage="pull_artifact",
            )
        return ArtifactBundle(**body, staged_artifact_path=str(artifact_path))

    def delete_staged_artifact(self, staged_artifact_path: str | None) -> None:
        if not staged_artifact_path:
            return
        artifact_path = Path(staged_artifact_path)
        if artifact_path.exists():
            artifact_path.unlink()

    def runtime_dir(self, deployment_id: str) -> str:
        runtime_dir = self.cache_dir / f"{deployment_id}.runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return str(runtime_dir)

    def delete_runtime_dir(self, runtime_dir: str | None) -> None:
        if not runtime_dir:
            return
        runtime_path = Path(runtime_dir)
        if not runtime_path.exists():
            return
        for child in runtime_path.glob("**/*"):
            if child.is_file():
                child.unlink()
        for child in sorted(runtime_path.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        runtime_path.rmdir()

    def cache_state(self) -> dict[str, object]:
        staged_artifacts = sorted(self.cache_dir.glob("*.json"))
        runtime_dirs = sorted(path for path in self.cache_dir.glob("*.runtime") if path.is_dir())
        total_bytes = 0
        for artifact in staged_artifacts:
            total_bytes += artifact.stat().st_size
        for rd in runtime_dirs:
            for child in rd.glob("**/*"):
                if child.is_file():
                    total_bytes += child.stat().st_size
        return {
            "cache_dir": str(self.cache_dir),
            "staged_artifact_count": len(staged_artifacts),
            "runtime_dir_count": len(runtime_dirs),
            "total_bytes": total_bytes,
            "staged_artifacts": [artifact.name for artifact in staged_artifacts],
            "runtime_dirs": [rd.name for rd in runtime_dirs],
        }

    def evict_stale_cache(self, active_deployment_ids: set[str]) -> dict[str, object]:
        evicted_artifacts: list[str] = []
        evicted_runtime_dirs: list[str] = []
        for artifact in sorted(self.cache_dir.glob("*.artifact.json")):
            deployment_id = artifact.name.removesuffix(".artifact.json")
            if deployment_id in active_deployment_ids:
                continue
            artifact.unlink()
            evicted_artifacts.append(artifact.name)
        for rd in sorted(path for path in self.cache_dir.glob("*.runtime") if path.is_dir()):
            deployment_id = rd.name.removesuffix(".runtime")
            if deployment_id in active_deployment_ids:
                continue
            self.delete_runtime_dir(str(rd))
            evicted_runtime_dirs.append(rd.name)
        return {
            "evicted_artifacts": evicted_artifacts,
            "evicted_runtime_dirs": evicted_runtime_dirs,
            "evicted_artifact_count": len(evicted_artifacts),
            "evicted_runtime_dir_count": len(evicted_runtime_dirs),
        }


def _docker_host() -> str:
    """Return the host IP to reach sibling Docker containers.

    When the node-agent itself runs inside Docker (via docker-compose with the
    Docker socket mounted), ports mapped with ``-p HOST_PORT:CONTAINER_PORT``
    are reachable on the **host** network, not on 127.0.0.1 inside the
    node-agent container.  We resolve this by reading the default gateway from
    /proc/net/route (the Docker bridge gateway) which points back to the host.
    """
    if not Path("/.dockerenv").exists():
        return "127.0.0.1"
    try:
        with open("/proc/net/route") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == "00000000":  # default route
                    # Gateway is a hex 32-bit int in host (little-endian) byte order
                    gw_hex = fields[2]
                    gw_ip = ".".join(str(int(gw_hex[i : i + 2], 16)) for i in range(6, -1, -2))
                    return gw_ip
    except (OSError, IndexError, ValueError):
        pass
    return "172.17.0.1"  # common Docker bridge gateway fallback


def _choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        Path(f"/proc/{pid}").stat()
        return True
    except FileNotFoundError:
        return False


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        process = subprocess.Popen(  # noqa: S603
            ["kill", "-TERM", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=2.0)
    except Exception:  # noqa: BLE001
        return
