"""UnifiedNodeAgentService — single service handling inference, pod, and VM workloads."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from greenference_protocol import (
    CapacityUpdate,
    ControlPlaneHTTPClient,
    ControlPlaneHTTPError,
    DeploymentState,
    DeploymentStatusUpdate,
    Heartbeat,
    LeaseAssignment,
    MinerRegistration,
    NodeCapability,
    SSHAccessRecord,
    UnifiedRuntimeRecord,
    VolumeRecord,
    WorkloadKind,
    WorkloadSpec,
)

from greenference_node_agent.config import Settings
from greenference_node_agent.domain.inference import (
    ArtifactBundle,
    DockerInferenceBackend,
    InferenceRuntimeError,
    LocalArtifactInferenceBackend,
    ProcessInferenceBackend,
    StagedArtifactStore,
)
from greenference_node_agent.domain.gpu_allocator import GpuAllocationError, GpuAllocator
from greenference_node_agent.domain.pod import PodError, ProcessPodBackend, StubPodBackend
from greenference_node_agent.domain.ssh import SSHError, build_ssh_access, choose_free_port, generate_ssh_keypair
from greenference_node_agent.domain.vm import FirecrackerVMBackend, StubVMBackend, VMError
from greenference_node_agent.domain.volume import LocalVolumeManager, VolumeError
from greenference_node_agent.infrastructure.repository import NodeAgentRepository

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NodeAgentService:
    """Unified node agent service — dispatches workloads by kind."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = NodeAgentRepository(state_path=settings.runtime_state_path)
        self.control_plane = ControlPlaneHTTPClient(
            base_url=settings.control_plane_url,
            hotkey=settings.miner_hotkey,
            auth_secret=settings.miner_auth_secret,
            coldkey_name=settings.coldkey_name,
            hotkey_name=settings.hotkey_name,
            auth_mode=settings.auth_mode,
        )

        # Pod backend
        if settings.pod_backend == "process":
            self.pod_backend: ProcessPodBackend | StubPodBackend = ProcessPodBackend()
        else:
            self.pod_backend = StubPodBackend()
        self.volume_manager = LocalVolumeManager(settings.volume_base_dir)

        # Inference backend
        fallback = LocalArtifactInferenceBackend()
        if settings.inference_backend == "docker":
            self.inference_backend: DockerInferenceBackend | ProcessInferenceBackend | LocalArtifactInferenceBackend = DockerInferenceBackend()
        elif settings.inference_backend == "process":
            self.inference_backend = ProcessInferenceBackend(fallback_backend=fallback)
        else:
            self.inference_backend = fallback
        self.artifact_store = StagedArtifactStore(settings.artifact_cache_dir)

        # VM backend
        if settings.vm_backend == "firecracker":
            self.vm_backend: StubVMBackend | FirecrackerVMBackend = FirecrackerVMBackend()
        else:
            self.vm_backend = StubVMBackend()

        # GPU device allocator
        self.gpu_allocator = GpuAllocator(settings.gpu_count)

        # Track volume records for cleanup
        self._volume_records: dict[str, VolumeRecord] = {}

    # --- Agent lifecycle ---

    def onboard(self, registration: MinerRegistration) -> MinerRegistration:
        logger.info("registering miner %s with control-plane at %s", registration.hotkey, self.settings.control_plane_url)
        return self.control_plane.register_miner(registration)

    def publish_heartbeat(self, heartbeat: Heartbeat) -> Heartbeat:
        return self.control_plane.record_heartbeat(heartbeat)

    def publish_capacity(self, update: CapacityUpdate) -> CapacityUpdate:
        return self.control_plane.update_capacity(update)

    def build_capacity_update(self) -> CapacityUpdate:
        """Build a CapacityUpdate reflecting current GPU availability."""
        s = self.settings
        # Calculate available GPUs based on active runtimes
        reserved_split_units = 0
        for rt in self.repository.runtimes.values():
            if rt.status in ("accepted", "preparing", "starting", "ready"):
                reserved_split_units += int(rt.gpu_fraction * s.gpu_split_units)
        total_units = s.gpu_count * s.gpu_split_units
        available_fractional = max(0, (total_units - reserved_split_units)) / s.gpu_split_units

        node = NodeCapability(
            hotkey=s.miner_hotkey,
            node_id=s.node_id,
            gpu_model=s.gpu_model,
            gpu_count=s.gpu_count,
            available_gpus=int(available_fractional),
            vram_gb_per_gpu=s.vram_gb_per_gpu,
            cpu_cores=s.cpu_cores,
            memory_gb=s.memory_gb,
            performance_score=s.performance_score,
            security_tier=s.security_tier,
        )
        return CapacityUpdate(hotkey=s.miner_hotkey, nodes=[node])

    # --- Lease sync & reconciliation ---

    def sync_leases(self, hotkey: str) -> list[LeaseAssignment]:
        try:
            return self.control_plane.list_leases(hotkey)
        except ControlPlaneHTTPError:
            logger.exception("failed to sync leases for %s", hotkey)
            return []

    def reconcile_once(self, hotkey: str) -> None:
        """Main reconciliation loop: sync leases, dispatch workloads, cleanup orphans."""
        leases = self.sync_leases(hotkey)
        active_deployment_ids = {lease.deployment_id for lease in leases}

        # Start new workloads from leases
        for lease in leases:
            existing = self.repository.get_runtime(lease.deployment_id)
            if existing is not None:
                # If the runtime was saved but never reached ready/failed, retry it
                if existing.status in ("accepted", "starting"):
                    logger.info("retrying stuck runtime %s (status=%s)", lease.deployment_id, existing.status)
                    self.repository.remove_runtime(lease.deployment_id)
                else:
                    continue  # Already completed (ready/failed/terminated)
            try:
                self._reconcile_workload(lease)
            except Exception:
                logger.exception("failed to reconcile lease %s", lease.deployment_id)
                # Report failure to control plane so deployment doesn't stay SCHEDULED forever
                try:
                    self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                        deployment_id=lease.deployment_id,
                        state=DeploymentState.FAILED,
                        error="node-agent reconciliation failed (check node-agent logs)",
                    ))
                except Exception:
                    logger.exception("failed to report reconciliation failure for %s", lease.deployment_id)

        # Terminate orphaned runtimes (no longer in active leases)
        for deployment_id, runtime in list(self.repository.runtimes.items()):
            if deployment_id in active_deployment_ids:
                continue
            if runtime.status == "terminated":
                continue
            # Also clean up containers for failed runtimes whose lease is gone
            logger.info("terminating orphaned runtime %s (status=%s)", deployment_id, runtime.status)
            self._terminate_runtime(runtime, reason="lease_expired")

    def _reconcile_workload(self, lease: LeaseAssignment) -> None:
        """Dispatch a new lease to the appropriate runtime handler based on workload kind."""
        logger.info("reconciling lease %s (workload=%s)", lease.deployment_id, lease.workload_id)
        workload = self.control_plane.get_workload(lease.workload_id)
        if workload is None:
            logger.warning("workload %s not found for lease %s — reporting failure", lease.workload_id, lease.deployment_id)
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=lease.deployment_id,
                state=DeploymentState.FAILED,
                error=f"workload {lease.workload_id} not found on control plane",
            ))
            return

        runtime = UnifiedRuntimeRecord(
            deployment_id=lease.deployment_id,
            workload_id=lease.workload_id,
            hotkey=lease.hotkey,
            node_id=lease.node_id,
            workload_kind=workload.kind,
            status="accepted",
            current_stage="accepted",
        )
        self.repository.upsert_runtime(runtime)

        kind = workload.kind
        if kind == WorkloadKind.INFERENCE:
            self._start_inference_runtime(runtime, workload)
        elif kind == WorkloadKind.POD:
            self._start_pod_runtime(runtime, workload)
        elif kind == WorkloadKind.VM:
            self._start_vm_runtime(runtime, workload)
        else:
            logger.error("unsupported workload kind: %s", kind)
            self._fail_runtime(runtime, f"unsupported workload kind: {kind}")

    # --- Workload-specific handlers ---

    def _start_inference_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start an inference workload (model serving)."""
        model_id = workload.runtime.model_identifier if workload.runtime else workload.image
        gpu_count = workload.requirements.gpu_count if workload.requirements else 1
        logger.info("starting inference runtime for %s (model: %s, gpus: %d)", runtime.deployment_id, model_id, gpu_count)

        # Allocate specific GPU devices
        try:
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count)
        except GpuAllocationError as exc:
            logger.error("GPU allocation failed for %s: %s", runtime.deployment_id, exc)
            self._fail_runtime(runtime, f"GPU allocation failed: {exc}")
            return

        runtime = runtime.model_copy(update={
            "gpu_fraction": gpu_count,
            "metadata": {**runtime.metadata, "gpu_devices": gpu_devices, "gpu_count": gpu_count},
        })

        # Stage artifact
        build_id = workload.metadata.get("build_id", runtime.deployment_id)
        image = workload.image or "unknown"
        artifact_uri = workload.metadata.get("artifact_uri", f"local://{image}")
        artifact_digest = workload.metadata.get("artifact_digest", f"sha256:{runtime.deployment_id[:16]}")
        runtime_manifest = {
            "runtime_kind": workload.metadata.get("runtime_kind", "hf-causal-lm"),
            "model_identifier": model_id,
            "model_revision": workload.metadata.get("model_revision"),
            "tokenizer_identifier": workload.metadata.get("tokenizer_identifier"),
            "seed_corpus": workload.metadata.get("seed_corpus", [
                f"{image} serves greenference inference requests",
                "miners keep deployments healthy with recovery failover and streaming completions",
            ]),
        }
        payload = {
            "build_id": build_id,
            "image": image,
            "docker_image": image,
            "runtime_manifest": runtime_manifest,
        }

        try:
            artifact = self.artifact_store.stage_artifact(
                deployment_id=runtime.deployment_id,
                build_id=build_id,
                image=image,
                artifact_uri=artifact_uri,
                artifact_digest=artifact_digest,
                registry_manifest_uri=workload.metadata.get("registry_manifest_uri"),
                context_manifest_uri=workload.metadata.get("context_manifest_uri"),
                dockerfile_path=workload.metadata.get("dockerfile_path"),
                payload=payload,
            )
        except InferenceRuntimeError:
            logger.exception("artifact staging failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, "artifact staging failed")
            return

        # Prepare runtime directory
        runtime_dir = self.artifact_store.runtime_dir(runtime.deployment_id)
        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_inference_backend",
            "model_identifier": model_id,
            "image": image,
            "runtime_mode": self.settings.inference_backend,
            "staged_artifact_path": artifact.staged_artifact_path,
            "runtime_dir": runtime_dir,
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)

        # Start inference process
        try:
            runtime = self.inference_backend.start_runtime(runtime, artifact)
        except InferenceRuntimeError as exc:
            logger.exception("inference start failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, str(exc))
            return

        runtime = runtime.model_copy(update={
            "status": "ready",
            "current_stage": "ready",
            "endpoint": runtime.runtime_url or f"{self.settings.miner_api_base_url}/deployments/{runtime.deployment_id}",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_pod_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a pod workload (Docker container with SSH)."""
        s = self.settings
        template_name = workload.metadata.get("template")
        logger.info("starting pod runtime for %s (template: %s)", runtime.deployment_id, template_name)

        # Pick SSH port and generate keypair
        try:
            ssh_port = choose_free_port(s.ssh_port_range_start, s.ssh_port_range_end)
            private_key, public_key = generate_ssh_keypair()
        except SSHError:
            logger.exception("SSH setup failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, "SSH setup failed")
            return

        # Allocate specific GPU devices
        gpu_count = workload.requirements.gpu_count if workload.requirements else 1
        try:
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count)
        except GpuAllocationError as exc:
            logger.error("GPU allocation failed for %s: %s", runtime.deployment_id, exc)
            self._fail_runtime(runtime, f"GPU allocation failed: {exc}")
            return

        # Create persistent volume
        gpu_fraction = gpu_count
        volume_size_gb = int(workload.metadata.get("volume_size_gb", 50))
        try:
            volume = self.volume_manager.create_volume(
                deployment_id=runtime.deployment_id,
                hotkey=runtime.hotkey,
                node_id=runtime.node_id,
                size_gb=volume_size_gb,
            )
            self._volume_records[runtime.deployment_id] = volume
        except VolumeError:
            logger.exception("volume creation failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, "volume creation failed")
            return

        # Resolve image from template or workload
        image = workload.image
        if not image and template_name:
            from greenference_node_agent.domain.templates import get_template
            tpl = get_template(template_name)
            if tpl:
                image = tpl.image

        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_pod",
            "template": template_name,
            "gpu_fraction": gpu_fraction,
            "ssh_host": s.ssh_host,
            "ssh_port": ssh_port,
            "ssh_username": "root",
            "ssh_fingerprint": None,
            "volume_path": volume.path,
            "volume_size_gb": volume_size_gb,
            "metadata": {
                **runtime.metadata,
                "image": image,
                "ssh_public_keys": [
                    public_key,
                    # Include user-provided SSH keys from workload metadata
                    *[k for k in (workload.metadata.get("ssh_public_keys") or []) if k.strip()],
                ],
                "ssh_private_key": private_key,
                "gpu_devices": gpu_devices,
                "gpu_count": gpu_count,
            },
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)

        # Start container
        try:
            runtime = self.pod_backend.start_pod(runtime, workload)
        except PodError as exc:
            logger.exception("pod start failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, str(exc))
            return

        # Wait for SSH to be reachable before marking ready
        if hasattr(self.pod_backend, "wait_for_ready"):
            if not self.pod_backend.wait_for_ready(runtime, timeout_seconds=30.0):
                logger.warning("pod SSH not reachable for %s after 30s, marking ready anyway", runtime.deployment_id)

        # Build endpoint with SSH connection details
        ssh_endpoint = f"ssh://{runtime.ssh_username}@{runtime.ssh_host}:{runtime.ssh_port}"
        runtime = runtime.model_copy(update={
            "endpoint": ssh_endpoint,
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_vm_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a VM workload (Firecracker/stub)."""
        gpu_count = workload.requirements.gpu_count if workload.requirements else 1
        logger.info("starting VM runtime for %s (gpus: %d)", runtime.deployment_id, gpu_count)

        try:
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count)
        except GpuAllocationError as exc:
            logger.error("GPU allocation failed for %s: %s", runtime.deployment_id, exc)
            self._fail_runtime(runtime, f"GPU allocation failed: {exc}")
            return

        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_vm",
            "metadata": {**runtime.metadata, "gpu_devices": gpu_devices, "gpu_count": gpu_count},
        })
        self.repository.upsert_runtime(runtime)

        try:
            runtime = self.vm_backend.start_vm(runtime, workload)
        except VMError as exc:
            logger.exception("VM start failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, str(exc))
            return

        runtime = runtime.model_copy(update={
            "endpoint": f"{self.settings.miner_api_base_url}/vms/{runtime.deployment_id}",
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    # --- Access helpers ---

    def get_ssh_access(self, deployment_id: str, include_private_key: bool = False) -> SSHAccessRecord | None:
        runtime = self.repository.get_runtime(deployment_id)
        if runtime is None or runtime.workload_kind not in (WorkloadKind.POD, "pod"):
            return None
        private_key = runtime.metadata.get("ssh_private_key") if include_private_key else None
        return build_ssh_access(runtime, include_private_key=include_private_key, private_key=private_key)

    def invoke_inference(self, deployment_id: str, payload: Any) -> Any:
        """Invoke chat completions on an inference runtime."""
        from greenference_protocol import ChatCompletionRequest
        runtime = self.repository.get_runtime(deployment_id)
        if runtime is None:
            return None
        if runtime.workload_kind not in (WorkloadKind.INFERENCE, "inference"):
            return None
        if runtime.status != "ready":
            return None
        request = ChatCompletionRequest(**payload) if isinstance(payload, dict) else payload
        return self.inference_backend.invoke(runtime, request)

    # --- Runtime lifecycle helpers ---

    def _report_deployment_ready(self, runtime: UnifiedRuntimeRecord) -> None:
        try:
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=runtime.deployment_id,
                state=DeploymentState.READY,
                endpoint=runtime.endpoint,
                ssh_private_key=runtime.metadata.get("ssh_private_key"),
                ready_instances=1,
            ))
        except ControlPlaneHTTPError:
            logger.exception("failed to report ready for %s", runtime.deployment_id)

    def _fail_runtime(self, runtime: UnifiedRuntimeRecord, error: str) -> None:
        # Release GPU allocation on failure
        self.gpu_allocator.release(runtime.deployment_id)

        runtime = runtime.model_copy(update={
            "status": "failed",
            "last_error": error,
            "failure_class": "workload_dispatch",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        try:
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=runtime.deployment_id,
                state=DeploymentState.FAILED,
                error=error,
            ))
        except ControlPlaneHTTPError:
            logger.exception("failed to report failure for %s", runtime.deployment_id)

    def _terminate_runtime(self, runtime: UnifiedRuntimeRecord, reason: str = "terminated") -> None:
        # Release GPU allocation
        self.gpu_allocator.release(runtime.deployment_id)

        # Stop the actual backend workload
        kind = runtime.workload_kind
        try:
            if kind in (WorkloadKind.INFERENCE, "inference") and (runtime.process_id or runtime.container_id):
                self.inference_backend.stop_runtime(runtime)
                if runtime.runtime_dir:
                    self.artifact_store.delete_runtime_dir(runtime.runtime_dir)
                if runtime.staged_artifact_path:
                    self.artifact_store.delete_staged_artifact(runtime.staged_artifact_path)
            elif kind in (WorkloadKind.POD, "pod") and runtime.container_id:
                self.pod_backend.stop_pod(runtime)
                volume = self._volume_records.pop(runtime.deployment_id, None)
                if volume:
                    self.volume_manager.delete_volume(volume)
            elif kind in (WorkloadKind.VM, "vm") and runtime.vm_id:
                self.vm_backend.stop_vm(runtime)
        except Exception:
            logger.exception("backend cleanup failed for %s", runtime.deployment_id)

        runtime = runtime.model_copy(update={
            "status": "terminated",
            "current_stage": "terminated",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        try:
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=runtime.deployment_id,
                state=DeploymentState.TERMINATED,
            ))
        except ControlPlaneHTTPError:
            logger.exception("failed to report termination for %s", runtime.deployment_id)

    def terminate_deployment(self, deployment_id: str) -> dict[str, Any]:
        runtime = self.repository.get_runtime(deployment_id)
        if runtime is None:
            return {"status": "not_found"}
        self._terminate_runtime(runtime, reason="user_request")
        return {"status": "terminated", "deployment_id": deployment_id}

    # --- Recovery ---

    def recover_runtime_state(self, hotkey: str) -> dict[str, Any]:
        """On startup, check persisted runtimes and terminate stale ones."""
        resumed = 0
        terminated = 0
        for deployment_id, runtime in list(self.repository.runtimes.items()):
            if runtime.status in ("terminated", "failed"):
                continue
            # Check if deployment still exists on control plane
            try:
                deployment = self.control_plane.get_deployment(deployment_id)
                if deployment and deployment.state in (DeploymentState.READY, DeploymentState.STARTING):
                    resumed += 1
                    continue
            except ControlPlaneHTTPError:
                pass
            # Stale runtime — terminate
            self._terminate_runtime(runtime, reason="stale_recovery")
            terminated += 1

        return {
            "last_recovery_at": _utcnow().isoformat(),
            "resumed_runtimes": resumed,
            "terminated_stale_runtimes": terminated,
        }

    # --- Observability ---

    def runtime_summary(self) -> dict[str, Any]:
        return self.repository.runtime_summary()

    def fleet_status(self) -> dict[str, Any]:
        summary = self.runtime_summary()
        return {
            "runtimes": summary,
            "hotkey": self.settings.miner_hotkey,
            "node_id": self.settings.node_id,
            "supported_kinds": self.settings.supported_workload_kinds,
        }
