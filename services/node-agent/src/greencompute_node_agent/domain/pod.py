"""Pod lifecycle management — ProcessPodBackend and StubPodBackend."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from math import floor
from pathlib import Path
from typing import Any

from greencompute_protocol import UnifiedRuntimeRecord, WorkloadSpec
from greencompute_node_agent.domain.disk import DiskMode
from greencompute_node_agent.domain.gpu_docker import gpu_docker_flags

# Path to the SSH bootstrap entrypoint script (mounted into every pod container)
_ENTRYPOINT_PATH = Path(__file__).resolve().parents[4] / "infra" / "docker" / "entrypoint.sh"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _last_log_lines(container_id: str, count: int = 5) -> str:
    try:
        logs = subprocess.run(  # noqa: S603
            ["docker", "logs", "--tail", str(count), container_id],  # noqa: S607
            capture_output=True, text=True, timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "(unavailable)"
    lines = [line.strip() for line in ((logs.stdout or "") + (logs.stderr or "")).splitlines() if line.strip()]
    return " | ".join(lines[-count:]) or "(none)"


def sshd_in_process_list(docker_top_output: str) -> bool:
    """Is sshd itself among these `docker top` rows? The command is the last
    column (named CMD or COMMAND; it may contain spaces) and its first word is
    the executable: `/usr/sbin/sshd -D`, or `sshd: /usr/sbin/sshd [listener]`
    once OpenSSH 9.8+ has set its process title."""
    lines = [line for line in docker_top_output.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header = lines[0].split()
    column = next((i for i, name in enumerate(header) if name.upper() in ("CMD", "COMMAND")), len(header) - 1)
    for row in lines[1:]:
        fields = row.split(None, column)
        if len(fields) <= column:
            continue
        words = fields[column].split()
        executable = words[0].rstrip(":").rsplit("/", 1)[-1] if words else ""
        if executable == "sshd":
            return True
    return False


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

    def pause_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def resume_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
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
        disk_mode: DiskMode = DiskMode.NONE,
    ) -> None:
        self.backend_name = backend_name
        self.disk_mode = disk_mode

    def start_pod(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        image = runtime.metadata.get("image") or workload.image
        container_name = f"greencompute-pod-{runtime.deployment_id[:12]}"

        # Clean up any stale container with this deterministic name before run.
        # If the node-agent was killed between `docker run` and persisting
        # container_id, the host container survives but the runtime record has
        # container_id=None, so _terminate_runtime can't remove it on retry —
        # and the bare `docker run --name` below would then fail forever with
        # "name already in use". `docker rm -f` is idempotent (no-op if absent).
        # Mirrors DockerInferenceBackend.start_runtime so pods self-heal too.
        try:
            subprocess.run(  # noqa: S603, S607
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # docker missing or rm hung — let the subsequent `docker run` be
            # the authoritative failure.
            pass

        cmd: list[str] = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
        ]

        # Resource limits (proportional to GPU share, RunPod-style). Zero means
        # "legacy unbounded" — skip the flag so existing workloads still work.
        if runtime.cpu_cores_allocated and runtime.cpu_cores_allocated > 0:
            cmd += ["--cpus", f"{runtime.cpu_cores_allocated:.2f}"]
        if runtime.memory_gb_allocated and runtime.memory_gb_allocated > 0:
            cmd += ["--memory", f"{runtime.memory_gb_allocated}g"]
        # Disk quota — only meaningful in STORAGE_OPT mode (caps the container
        # overlay). In LOOP_MOUNT modes the /workspace bind-mount is already
        # a size-capped ext4 loop, so no extra flag is needed.
        if self.disk_mode == DiskMode.STORAGE_OPT and runtime.volume_size_gb:
            cmd += ["--storage-opt", f"size={runtime.volume_size_gb}G"]

        # SSH port forwarding — container always uses port 22 (injected sshd),
        # host port comes from the allocator (GREENCOMPUTE_SSH_PORT_RANGE_*).
        container_ssh_port = 22
        if runtime.ssh_port:
            cmd += ["-p", f"0.0.0.0:{runtime.ssh_port}:{container_ssh_port}"]

        # Extra user-exposed ports: {container_port: host_port} pre-allocated
        # by services.py from GREENCOMPUTE_USER_PORT_RANGE_* (max 10).
        port_allocations: dict[int, int] = runtime.metadata.get("port_allocations") or {}
        for container_port_str, host_port in port_allocations.items():
            try:
                cp = int(container_port_str)
                hp = int(host_port)
            except (TypeError, ValueError):
                continue
            cmd += ["-p", f"0.0.0.0:{hp}:{cp}"]

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
        # GREENCOMPUTE_HOST_REPO_DIR tells us where the repo lives on the host.
        host_repo = os.environ.get("GREENCOMPUTE_HOST_REPO_DIR", "")
        if host_repo:
            entrypoint = f"{host_repo}/infra/docker/entrypoint.sh"
        else:
            entrypoint = str(_ENTRYPOINT_PATH)
        cmd += ["-v", f"{entrypoint}:/greencompute-entrypoint.sh:ro"]
        cmd += ["--entrypoint", "/greencompute-entrypoint.sh"]

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

        # Read back the actual host port Docker assigned (sanity check —
        # should match what we bound, but Docker may pick differently under
        # race conditions).
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

        # Verify each user-exposed port matches the allocation. If Docker
        # reports a different host port than what we requested, trust Docker.
        verified_port_mappings: dict[int, int] = {}
        for container_port_str, host_port in port_allocations.items():
            try:
                cp = int(container_port_str)
            except (TypeError, ValueError):
                continue
            actual_host_port = int(host_port) if host_port else 0
            try:
                pr = subprocess.run(  # noqa: S603
                    ["docker", "port", container_id, str(cp)],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if pr.returncode == 0 and pr.stdout.strip():
                    for line in pr.stdout.strip().splitlines():
                        if ":" in line:
                            actual_host_port = int(line.rsplit(":", 1)[-1])
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
            if actual_host_port:
                verified_port_mappings[cp] = actual_host_port

        return runtime.model_copy(
            update={
                "container_id": container_id,
                "ssh_port": actual_ssh_port,
                "port_mappings": verified_port_mappings,
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

    def ssh_status(self, runtime: UnifiedRuntimeRecord) -> tuple[str, str | None]:
        """One non-blocking look at a started pod's SSH: ("up", None) once
        sshd runs, ("pending", None) while it still might, or ("failed",
        reason) once it never can, with the reason in words for the customer.
        The node agent calls this once per reconcile tick; a blocking wait
        here (as before) stalled every other lease on the node for as long as
        apt-get took.

        Checks INSIDE the container via the docker socket, NOT by dialing the
        node's public SSH host. On our providers the node can't reach its own
        public IP (NAT hairpin), so the old `socket.create_connection(
        (ssh_host, ssh_port))` false-negatived on EVERY pod and burned the full
        timeout without ever verifying anything.
        """
        container = runtime.container_id
        if not container:
            return "up", None  # stub / no container — nothing to verify
        # A container that keeps exiting is checked first: Docker restarts it
        # (--restart unless-stopped), so sshd may well be "up" for a moment in
        # each cycle and would otherwise be reported ready at random.
        dead = self._container_problem(container)
        if dead:
            return "failed", dead
        if self._sshd_running(container):
            return "up", None
        problem = self._ssh_setup_problem(container)
        if problem:
            return "failed", problem
        return "pending", None

    @staticmethod
    def _sshd_running(container_id: str) -> bool:
        """True once the sshd daemon itself is running inside the pod. `docker
        top` reads the container's processes host-side, so it needs no tooling
        in the user's image and is immune to the node's own network/hairpin.

        Only the executable counts, not the arguments: the set-up steps that
        run before sshd exists (`mkdir -p /run/sshd`, `sed … sshd_config`, the
        installer's `adduser … sshd`) all mention sshd, and the substring match
        this replaces reported pods ready that then refused SSH for minutes
        (external API user, 2026-10)."""
        try:
            result = subprocess.run(  # noqa: S603
                ["docker", "top", container_id],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0 and sshd_in_process_list(result.stdout)

    @staticmethod
    def _container_problem(container_id: str) -> str | None:
        """The container itself is gone or keeps exiting, or None if it is up.
        The entrypoint only exits if something kills it (an OOM, say): our
        `--entrypoint` clears the image's CMD, so a pod otherwise idles in
        `sleep infinity`."""
        try:
            state = subprocess.run(  # noqa: S603
                ["docker", "inspect", "-f",  # noqa: S607
                 "{{.State.Running}} {{.State.Restarting}} {{.RestartCount}} {{.State.ExitCode}}", container_id],
                capture_output=True, text=True, timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if state.returncode != 0:
            if "No such" in (state.stderr or ""):
                return "the container is gone (removed outside the platform)"
            return None
        words = state.stdout.split()
        if len(words) < 4:
            return None
        running, restarting, restarts, code = words[0], words[1], int(words[2] or 0), words[3]
        if restarting == "true" or restarts > 0 or running == "false":
            tail = _last_log_lines(container_id)
            times = f"restarted {restarts} time(s)" if restarts else "not restarted"
            return (
                f"the container exits right after it starts (exit code {code}, {times}), so SSH can't stay up. "
                f"Last output: {tail}"
            )
        return None

    @staticmethod
    def _ssh_setup_problem(container_id: str) -> str | None:
        """Why the entrypoint could not bring sshd up in this image, in words
        for the customer, or None while it still might. Only what the
        entrypoint logs as final counts: an image with no package manager, or
        a package install that failed. The entrypoint's own "sshd failed to
        start" is NOT final (the image's start script may start sshd itself),
        so that one is left to the timeout."""
        try:
            logs = subprocess.run(  # noqa: S603
                ["docker", "logs", "--tail", "5000", container_id],  # noqa: S607
                capture_output=True, text=True, timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if logs.returncode != 0:
            return None
        output = (logs.stdout or "") + (logs.stderr or "")
        if "cannot install openssh-server" in output:
            return (
                "the SSH server could not be installed in this image (no apt-get, apk, yum or dnf). "
                "Use an image based on Ubuntu, Debian, Alpine or CentOS/Rocky, or one with openssh-server preinstalled"
            )
        if "[greencompute] SSH setup failed" in output:
            return (
                "installing the SSH server in this image failed (a package download or mirror error, usually). "
                "Try again, or use an image with openssh-server preinstalled"
            )
        return None

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

    def pause_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        """Suspend a pod by stopping its container WITHOUT removing it.

        Pod-safe: uses `docker stop` (graceful SIGTERM, then SIGKILL) so the
        container's filesystem, named volumes, SSH host keys and port mappings
        are all preserved on disk. The container moves to the 'exited' state and
        can be restarted in place via resume_pod. We deliberately keep
        container_id so resume can `docker start` the SAME container.
        NEVER `docker rm -f` here — that would destroy the tenant's pod state.
        """
        if runtime.container_id:
            try:
                subprocess.run(  # noqa: S603
                    ["docker", "stop", runtime.container_id],  # noqa: S607
                    capture_output=True,
                    timeout=60.0,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return runtime.model_copy(
            update={
                "status": "suspended",
                "current_stage": "suspended",
                # container_id intentionally PRESERVED so resume can restart it.
                "metadata": {**runtime.metadata, "suspended_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def resume_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        """Resume a previously-paused pod by starting the SAME container.

        `docker start` re-launches the exited container with its original
        filesystem, volumes, SSH keys, env and port bindings intact — no
        destroy-and-recreate, no data loss.
        """
        if runtime.container_id:
            try:
                result = subprocess.run(  # noqa: S603
                    ["docker", "start", runtime.container_id],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                )
                if result.returncode != 0:
                    raise PodError(
                        f"docker start failed: {result.stderr}",
                        failure_class="pod_resume_failure",
                        stage="resume_pod",
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise PodError(
                    f"docker not available: {exc}",
                    failure_class="pod_resume_failure",
                    stage="resume_pod",
                ) from exc
        return runtime.model_copy(
            update={
                "status": "ready",
                "current_stage": "ready",
                "metadata": {**runtime.metadata, "resumed_at": _utcnow().isoformat()},
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

    def pause_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        # Stub: preserve container_id (no real docker stop), just flip state.
        return runtime.model_copy(
            update={
                "status": "suspended",
                "current_stage": "suspended",
                "metadata": {**runtime.metadata, "suspended_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def resume_pod(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        return runtime.model_copy(
            update={
                "status": "ready",
                "current_stage": "ready",
                "metadata": {**runtime.metadata, "resumed_at": _utcnow().isoformat()},
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
