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
    UnifiedRuntimeRecord,
    WorkloadKind,
    WorkloadSpec,
)

from greenference_node_agent.config import Settings
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
                continue  # Already tracked
            try:
                self._reconcile_workload(lease)
            except Exception:
                logger.exception("failed to reconcile lease %s", lease.deployment_id)

        # Terminate orphaned runtimes (no longer in active leases)
        for deployment_id, runtime in list(self.repository.runtimes.items()):
            if runtime.status in ("terminated", "failed"):
                continue
            if deployment_id not in active_deployment_ids:
                logger.info("terminating orphaned runtime %s", deployment_id)
                self._terminate_runtime(runtime, reason="lease_expired")

    def _reconcile_workload(self, lease: LeaseAssignment) -> None:
        """Dispatch a new lease to the appropriate runtime handler based on workload kind."""
        workload = self.control_plane.get_workload(lease.workload_id)
        if workload is None:
            logger.warning("workload %s not found for lease %s", lease.workload_id, lease.deployment_id)
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
        logger.info("starting inference runtime for %s (model: %s)", runtime.deployment_id, workload.runtime.model_identifier)
        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_inference_backend",
            "model_identifier": workload.runtime.model_identifier,
            "image": workload.image,
            "runtime_mode": self.settings.inference_backend,
        })
        self.repository.upsert_runtime(runtime)

        # TODO: Phase 2 full implementation — pull artifact, start subprocess/K8s,
        # health check, mark ready. For now, mark ready with stub endpoint.
        runtime = runtime.model_copy(update={
            "status": "ready",
            "current_stage": "ready",
            "endpoint": f"{self.settings.miner_api_base_url}/deployments/{runtime.deployment_id}",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_pod_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a pod workload (Docker container with SSH)."""
        logger.info("starting pod runtime for %s (template: %s)", runtime.deployment_id, workload.metadata.get("template"))
        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_pod",
            "template": workload.metadata.get("template"),
            "gpu_fraction": workload.requirements.gpu_count,
        })
        self.repository.upsert_runtime(runtime)

        # TODO: Phase 2 full implementation — create volume, generate SSH keypair,
        # pick port, start Docker container. For now, mark ready with stub.
        runtime = runtime.model_copy(update={
            "status": "ready",
            "current_stage": "ready",
            "endpoint": f"{self.settings.miner_api_base_url}/deployments/{runtime.deployment_id}",
            "ssh_host": self.settings.ssh_host,
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_vm_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a VM workload (Firecracker/stub)."""
        logger.info("starting VM runtime for %s", runtime.deployment_id)
        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_vm",
        })
        self.repository.upsert_runtime(runtime)

        # TODO: Phase 2 full implementation — start Firecracker VM.
        runtime = runtime.model_copy(update={
            "status": "ready",
            "current_stage": "ready",
            "endpoint": f"{self.settings.miner_api_base_url}/vms/{runtime.deployment_id}",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    # --- Runtime lifecycle helpers ---

    def _report_deployment_ready(self, runtime: UnifiedRuntimeRecord) -> None:
        try:
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=runtime.deployment_id,
                state=DeploymentState.READY,
                endpoint=runtime.endpoint,
                ready_instances=1,
            ))
        except ControlPlaneHTTPError:
            logger.exception("failed to report ready for %s", runtime.deployment_id)

    def _fail_runtime(self, runtime: UnifiedRuntimeRecord, error: str) -> None:
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
