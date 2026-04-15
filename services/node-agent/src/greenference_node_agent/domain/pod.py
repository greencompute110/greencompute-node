"""Pod lifecycle management — ProcessPodBackend and StubPodBackend."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from math import floor
from pathlib import Path
from typing import Any

from greenference_protocol import UnifiedRuntimeRecord, WorkloadSpec
from greenference_node_agent.domain.gpu_docker import gpu_docker_flags

# Path to the SSH bootstrap entrypoint script (mounted into every pod container)
_ENTRYPOINT_PATH = Path(__file__).resolve().parents[4] / "infra" / "docker" / "entrypoint.sh"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PodError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str, stage: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.stage = stage


class PodBackend:
    """Abstract compute pod backend."""

    def start_pod(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def stop_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        raise NotImplementedError

    def exec_command(self, runtime: UnifiedRuntimeRecord, command: list[str]) -> str:
        raise NotImplementedError

    def stream_logs(self, runtime: UnifiedRuntimeRecord) -> Iterator[str]:
        raise NotImplementedError


class ProcessPodBackend(PodBackend):
    """Runs pods as Docker containers via subprocess."""

    def __init__(
        self,
        *,
        backend_name: str = "process-pod-backend",
    ) -> None:
        self.backend_name = backend_name

    def start_pod(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        image = runtime.metadata.get("image") or workload.image
        container_name = f"greenference-pod-{runtime.deployment_id[:12]}"

        cmd: list[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
        ]

        # SSH port forwarding — container always uses port 22 (injected sshd)
        container_ssh_port = 22
        if runtime.ssh_port:
            cmd += ["-p", f"0.0.0.0::{container_ssh_port}"]

        # Volume mount
        if runtime.volume_path:
            cmd += ["-v", f"{runtime.volume_path}:/workspace"]

        # GPU passthrough — method auto-detected at startup (see gpu_docker.py)
        gpu_devices: list[int] | None = runtime.metadata.get("gpu_devices")
        if gpu_devices or runtime.gpu_fraction > 0:
            cmd += gpu_docker_flags(gpu_devices if gpu_devices else None)

        # Environment variables
        env_vars: dict[str, str] = runtime.metadata.get("env_vars", {})
        for key, value in env_vars.items():
            cmd += ["-e", f"{key}={value}"]

        # SSH authorized keys — passed via AUTHORIZED_KEYS env var, read by entrypoint.sh
        ssh_public_keys: list[str] = runtime.metadata.get("ssh_public_keys", [])
        if ssh_public_keys:
            keys_str = "\n".join(ssh_public_keys)
            cmd += ["-e", f"AUTHORIZED_KEYS={keys_str}"]

        # Mount the SSH bootstrap entrypoint and override the container entrypoint.
        # This injects sshd into any image (Vast.ai / RunPod style).
        # The entrypoint installs openssh-server, writes keys as root, starts
        # sshd on port 22, then execs the original CMD or `sleep infinity`.
        #
        # When we run inside Docker (docker-compose), the path we pass to
        # `docker run -v` is resolved by the host daemon, not our container.
        # GREENFERENCE_HOST_REPO_DIR tells us where the repo lives on the host.
        host_repo = os.environ.get("GREENFERENCE_HOST_REPO_DIR", "")
        if host_repo:
            entrypoint = f"{host_repo}/infra/docker/entrypoint.sh"
        else:
            entrypoint = str(_ENTRYPOINT_PATH)
        cmd += ["-v", f"{entrypoint}:/greenference-entrypoint.sh:ro"]
        cmd += ["--entrypoint", "/greenference-entrypoint.sh"]

        cmd.append(image)

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=600.0,
            )
            if result.returncode != 0:
                raise PodError(
                    f"docker run failed: {result.stderr}",
                    failure_class="pod_start_failure",
                    stage="start_pod",
                )
            container_id = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise PodError(
                f"docker not available: {exc}",
                failure_class="pod_start_failure",
                stage="start_pod",
            ) from exc

        # Read back the actual host port Docker assigned
        actual_ssh_port = runtime.ssh_port
        if runtime.ssh_port:
            try:
                port_result = subprocess.run(  # noqa: S603
                    ["docker", "port", container_id, str(container_ssh_port)],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if port_result.returncode == 0 and port_result.stdout.strip():
                    # output like "0.0.0.0:31234" or ":::31234"
                    for line in port_result.stdout.strip().splitlines():
                        if ":" in line:
                            actual_ssh_port = int(line.rsplit(":", 1)[-1])
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass

        return runtime.model_copy(
            update={
                "container_id": container_id,
                "ssh_port": actual_ssh_port,
                "status": "ready",
                "current_stage": "ready",
                "metadata": {
                    **runtime.metadata,
                    "container_name": container_name,
                    "backend": self.backend_name,
                    "started_at": _utcnow().isoformat(),
                },
                "updated_at": _utcnow(),
            }
        )

    def wait_for_ready(
        self,
        runtime: UnifiedRuntimeRecord,
        *,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Wait for the pod's SSH port to be reachable. Returns True if ready."""
        if not runtime.ssh_port:
            return True  # No SSH port, nothing to wait for
        host = runtime.ssh_host or "127.0.0.1"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                with socket.create_connection((host, runtime.ssh_port), timeout=2.0):
                    return True
            except (ConnectionRefusedError, TimeoutError, OSError):
                time.sleep(0.5)
        return False

    def stop_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        if runtime.container_id:
            try:
                subprocess.run(  # noqa: S603
                    ["docker", "rm", "-f", runtime.container_id],  # noqa: S607
                    capture_output=True,
                    timeout=30.0,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return runtime.model_copy(
            update={
                "status": "terminated",
                "current_stage": "terminated",
                "container_id": None,
                "metadata": {**runtime.metadata, "terminated_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        if not runtime.container_id:
            return {"status": "not_started", "healthy": False}
        try:
            result = subprocess.run(  # noqa: S603
                ["docker", "inspect", "--format", "{{.State.Status}}", runtime.container_id],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                container_status = result.stdout.strip()
                return {"status": container_status, "healthy": container_status == "running"}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return {"status": "unknown", "healthy": False}

    def exec_command(self, runtime: UnifiedRuntimeRecord, command: list[str]) -> str:
        if not runtime.container_id:
            raise PodError(
                "pod not running",
                failure_class="pod_exec_failure",
                stage="exec_command",
            )
        try:
            result = subprocess.run(  # noqa: S603
                ["docker", "exec", runtime.container_id, *command],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise PodError(
                f"exec failed: {exc}",
                failure_class="pod_exec_failure",
                stage="exec_command",
            ) from exc

    def stream_logs(self, runtime: UnifiedRuntimeRecord) -> Iterator[str]:
        if not runtime.container_id:
            return
        try:
            process = subprocess.Popen(  # noqa: S603
                ["docker", "logs", "--follow", runtime.container_id],  # noqa: S607
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if process.stdout:
                for line in process.stdout:
                    yield line
        except (FileNotFoundError, OSError):
            return


class StubPodBackend(PodBackend):
    """Stub backend for testing — simulates pod lifecycle without Docker."""

    def __init__(self, *, backend_name: str = "stub-pod-backend") -> None:
        self.backend_name = backend_name

    def start_pod(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        return runtime.model_copy(
            update={
                "container_id": f"stub-container-{runtime.deployment_id[:8]}",
                "status": "ready",
                "current_stage": "ready",
                "metadata": {
                    **runtime.metadata,
                    "backend": self.backend_name,
                    "started_at": _utcnow().isoformat(),
                    "stub": True,
                },
                "updated_at": _utcnow(),
            }
        )

    def stop_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        return runtime.model_copy(
            update={
                "status": "terminated",
                "current_stage": "terminated",
                "container_id": None,
                "metadata": {**runtime.metadata, "terminated_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        return {"status": "running", "healthy": True, "stub": True}

    def exec_command(self, runtime: UnifiedRuntimeRecord, command: list[str]) -> str:
        return f"stub output for: {' '.join(command)}\n"

    def stream_logs(self, runtime: UnifiedRuntimeRecord) -> Iterator[str]:
        yield "stub log line 1\n"
        yield "stub log line 2\n"


def gpu_split_units_for_fraction(gpu_fraction: float, gpu_split_units: int) -> int:
    """Return the number of split units required for the given GPU fraction."""
    return floor(gpu_fraction * gpu_split_units)
