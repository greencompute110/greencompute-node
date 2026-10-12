"""UnifiedNodeAgentService — single service handling inference, pod, and VM workloads."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from greencompute_protocol import (
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

from greencompute_node_agent.config import Settings
from greencompute_node_agent.domain.inference import (
    ArtifactBundle,
    DockerInferenceBackend,
    InferenceRuntimeError,
    LocalArtifactInferenceBackend,
    ProcessInferenceBackend,
    StagedArtifactStore,
    resolve_tensor_parallel_size,
)
from greencompute_node_agent.domain.multinode_launch import is_worker_runtime
from greencompute_node_agent.domain.gpu_allocator import GpuAllocationError, GpuAllocator
from greencompute_node_agent.domain.disk import detect_disk_mode
from greencompute_node_agent.domain.pod import PodError, ProcessPodBackend, StubPodBackend
from greencompute_node_agent.domain.ssh import SSHError, build_ssh_access, choose_free_port, is_port_free, pod_ssh_keys
from greencompute_node_agent.domain.vm import FirecrackerVMBackend, StubVMBackend, VMError
from greencompute_node_agent.domain.volume import LocalVolumeManager, VolumeError
from greencompute_node_agent.infrastructure.repository import NodeAgentRepository

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

        # Disk enforcement — auto-detect the strongest mode this host supports.
        # Operator can override via GREENCOMPUTE_DISK_ENFORCEMENT_MODE env var.
        self.disk_mode = detect_disk_mode(settings.disk_enforcement_mode)
        if self.disk_mode.value == "none":
            logger.warning(
                "disk quotas disabled — pod /workspace is unbounded; "
                "run node-agent as root or set up sudoers/XFS-pquota to enforce"
            )
        else:
            logger.info("disk enforcement mode: %s", self.disk_mode.value)

        # Pod backend
        if settings.pod_backend == "process":
            self.pod_backend: ProcessPodBackend | StubPodBackend = ProcessPodBackend(disk_mode=self.disk_mode)
        else:
            self.pod_backend = StubPodBackend()
        self.volume_manager = LocalVolumeManager(settings.volume_base_dir, disk_mode=self.disk_mode)

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

        # Serializes the compound reconcile/recover flow. The per-operation
        # locks on GpuAllocator/NodeAgentRepository only guard single calls;
        # the snapshot→allocate→docker-run→upsert sequence must be atomic as a
        # whole, or the worker loop and the manual /reconcile + /recovery routes
        # (all run in the threadpool) can both observe a lease as unstarted and
        # each allocate GPUs for it — leaking allocations and orphaning a
        # container on the docker-name collision.
        self._reconcile_lock = threading.Lock()

        # Rebuild allocator state from persisted runtimes — prevents double-
        # allocation across node-agent restarts. Running containers from
        # before the restart still hold their GPUs via --gpus; the allocator
        # must know about them so it doesn't hand the same cards to a new
        # workload.
        self._rehydrate_gpu_allocator()

        # Track volume records for cleanup
        self._volume_records: dict[str, VolumeRecord] = {}

    def _rehydrate_gpu_allocator(self) -> None:
        active_states = {"accepted", "preparing", "starting", "ready"}
        for rt in self.repository.snapshot_runtimes():
            if rt.status not in active_states:
                continue
            devices = rt.metadata.get("gpu_devices") or []
            if not devices:
                continue
            try:
                # Force allocator to treat these devices as taken by this
                # deployment. We bypass the normal allocate() path because
                # we already know which specific devices were assigned.
                self.gpu_allocator.rehydrate(rt.deployment_id, {int(d) for d in devices})
                logger.info(
                    "rehydrated GPU allocation: %s → %s",
                    rt.deployment_id, sorted(devices),
                )
            except (TypeError, ValueError):
                logger.warning(
                    "could not rehydrate GPUs for %s (bad metadata: %r)",
                    rt.deployment_id, devices,
                )

    # --- Agent lifecycle ---

    def onboard(self, registration: MinerRegistration) -> MinerRegistration:
        logger.info("registering miner %s with control-plane at %s", registration.hotkey, self.settings.control_plane_url)
        return self.control_plane.register_miner(registration)

    def publish_heartbeat(self, heartbeat: Heartbeat) -> Heartbeat:
        return self.control_plane.record_heartbeat(heartbeat)

    def publish_capacity(self, update: CapacityUpdate) -> CapacityUpdate:
        return self.control_plane.update_capacity(update)

    def _foreign_busy_devices(self) -> set[int]:
        """GPU device indices physically in use by a NON-GreenCompute process
        (co-tenant / squatter): busy per nvidia-smi but NOT in our own
        allocation set. Best-effort — an empty set (probe unavailable) preserves
        the legacy behavior. We must neither advertise nor allocate these."""
        try:
            from greencompute_node_agent.domain.gpu_probe import busy_gpu_devices
            return busy_gpu_devices() - self.gpu_allocator.allocated_devices()
        except Exception:
            logger.debug("foreign-GPU probe failed; assuming none", exc_info=True)
            return set()

    def build_capacity_update(self) -> CapacityUpdate:
        """Build a CapacityUpdate reflecting current GPU availability."""
        s = self.settings
        # Derive available GPUs from the allocator's whole-device truth, NOT from
        # the overloaded gpu_fraction field (which the node-agent populates with a
        # whole device COUNT, not a 0..1 share — see _start_*_runtime). The
        # allocator and metadata["gpu_devices"] both count whole devices, so this
        # matches gpu_allocator.free_count exactly and is robust regardless of
        # what gpu_fraction holds.
        reserved_gpus = 0
        for rt in self.repository.snapshot_runtimes():
            if rt.status in ("accepted", "preparing", "starting", "ready"):
                devices = rt.metadata.get("gpu_devices") or []
                reserved_gpus += len(devices) or int(rt.metadata.get("gpu_count", 1) or 1)
        # Exclude GPUs a NON-GreenCompute process is physically using (co-tenant
        # / crypto squatter). Without this the node is "theft-blind" and would
        # advertise stolen cards as free → the scheduler over-commits a customer
        # onto them (the Babelbit/.24 golden-miner incident).
        foreign = self._foreign_busy_devices()
        if foreign:
            logger.warning(
                "%d GPU(s) %s on %s are in use by non-GreenCompute processes — "
                "excluding from advertised capacity",
                len(foreign), sorted(foreign), s.node_id,
            )
        available_gpus = max(0, s.gpu_count - reserved_gpus - len(foreign))

        # The scheduler's workload_kinds filter reads this label (exact
        # comma-separated tokens; absent = node accepts everything). Without
        # it a pod/vm rental could be placed on an inference-only node.
        kinds = ",".join(
            k.strip().lower() for k in s.supported_workload_kinds if k.strip()
        )
        # Operator-supplied labels (cluster_ip / interconnect_domain /
        # interconnect_gbps drive distributed-inference placement) merge with the
        # built-in workload_kinds label.
        labels: dict[str, str] = dict(s.node_labels)
        if kinds:
            labels["workload_kinds"] = kinds
        node = NodeCapability(
            hotkey=s.miner_hotkey,
            node_id=s.node_id,
            gpu_model=s.gpu_model,
            gpu_count=s.gpu_count,
            available_gpus=available_gpus,
            vram_gb_per_gpu=s.vram_gb_per_gpu,
            cpu_cores=s.cpu_cores,
            memory_gb=s.memory_gb,
            performance_score=s.performance_score,
            security_tier=s.security_tier,
            labels=labels,
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
        """Serialize the reconcile flow against the worker loop and the manual
        /reconcile + /recovery routes. If a pass is already running we skip
        rather than race into a double GPU allocation; the next tick (or a
        retry of the manual trigger) picks it up."""
        if not self._reconcile_lock.acquire(blocking=False):
            logger.info("reconcile already in progress — skipping this trigger")
            return
        try:
            self._reconcile_once_locked(hotkey)
        finally:
            self._reconcile_lock.release()

    def _reconcile_once_locked(self, hotkey: str) -> None:
        """Main reconciliation loop: sync leases, dispatch workloads, cleanup orphans."""
        all_leases = self.sync_leases(hotkey)
        # Multi-box-per-hotkey: the control-plane returns EVERY lease for the
        # hotkey, but each lease is placed on a specific node_id. Without this
        # filter, every box sharing the hotkey would run every deployment —
        # duplicating work and leaking GPU capacity across the fleet. Only act
        # on leases assigned to THIS box's node_id. A lease with no node_id is
        # treated as ours (single-box / legacy back-compat) but logged so the
        # gap is visible.
        my_node = self.settings.node_id
        leases = []
        for lease in all_leases:
            lease_node = getattr(lease, "node_id", None)
            if lease_node and lease_node != my_node:
                continue
            if not lease_node:
                logger.warning(
                    "lease %s has no node_id — running on %s by back-compat; "
                    "control-plane should pin it to a node",
                    lease.deployment_id, my_node,
                )
            leases.append(lease)
        active_deployment_ids = {lease.deployment_id for lease in leases}
        # Every lease for this hotkey across the WHOLE fleet, regardless of
        # node_id. Used by orphan-cleanup to tell "this deployment's lease is
        # gone everywhere" (safe to report TERMINATED) apart from "this is a
        # duplicate we ran for another box — its lease is alive on that box"
        # (tear down locally, but DON'T report or we'd kill the real pod).
        fleet_deployment_ids = {lease.deployment_id for lease in all_leases}

        # Start new workloads from leases
        for lease in leases:
            existing = self.repository.get_runtime(lease.deployment_id)

            # --- Pod-safe SUSPEND/RESUME (ORCH-H3) ---
            # SAFETY (load-bearing): we only EVER suspend a runtime when the
            # control-plane has EXPLICITLY marked this lease status='suspended'.
            # Any other status, a missing/unknown status, or (handled in
            # sync_leases) a control-plane that is unreachable => the lease does
            # not reach this branch and the runtime KEEPS RUNNING by default.
            # This is what guarantees the live tenant pods (lease.status='active')
            # are never stopped by deploying this change.
            lease_status = (getattr(lease, "status", None) or "").lower()
            if lease_status == "suspended":
                # Control-plane has explicitly suspended this lease. Stop the
                # runtime in place (docker stop, NOT rm) and free its GPUs.
                if existing is not None and existing.status in ("ready", "starting"):
                    self._suspend_runtime(existing)
                # If no live runtime exists (already suspended, or never started),
                # there is nothing to do — never dispatch a fresh runtime for a
                # suspended lease.
                continue
            if existing is not None and existing.status == "suspended":
                # Lease is NOT suspended (handled above) but the local runtime is
                # — the control-plane has resumed it (status back to active).
                # Restart the SAME container and re-acquire its GPU allocation.
                self._resume_runtime(existing)
                continue

            if existing is not None:
                # If the runtime was saved but never reached ready/failed, retry it
                if existing.status in ("accepted", "starting"):
                    # Tear down locally FIRST (release GPUs + stop any launched
                    # container) before remove+re-dispatch — otherwise the old
                    # allocation/container leaks (the inf-8f1f5461 orphan). Use
                    # report=False: the lease is still active and we're about to
                    # re-create the runtime, so a TERMINATED report would race the
                    # fresh record. A READY runtime never reaches this branch (it
                    # hits the else: continue below), so healthy pods are untouched.
                    logger.info(
                        "retrying stuck runtime %s (status=%s) — local teardown then re-dispatch",
                        lease.deployment_id, existing.status,
                    )
                    self._terminate_runtime(existing, reason="stuck_retry", report=False)
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

        # Clean up runtimes that are no longer ours to run.
        for deployment_id, runtime in self.repository.snapshot_runtime_items():
            if deployment_id in active_deployment_ids:
                continue
            if runtime.status == "terminated":
                continue
            if runtime.status == "suspended":
                # Pod-safe SUSPEND (ORCH-H3): a suspended runtime is an
                # INTENTIONALLY stopped-but-preserved container. Lease ABSENCE
                # alone must NEVER destroy it — a transient list_leases gap would
                # otherwise wipe preserved work. So before reaping, confirm with
                # the control-plane: tear it down ONLY if the deployment is
                # EXPLICITLY TERMINATED (the credit-exhaustion reaper deleting a
                # saved/unsaved pod, or an operator cleanup). A still-SUSPENDED
                # deployment — or an unreachable control-plane — keeps the
                # container on disk for a later `docker start` resume.
                terminated = False
                try:
                    dep = self.control_plane.get_deployment(deployment_id)
                    terminated = dep is not None and dep.state == DeploymentState.TERMINATED
                except Exception:
                    logger.warning(
                        "could not confirm deployment state for suspended runtime %s — keeping it",
                        deployment_id,
                    )
                if terminated:
                    logger.info(
                        "reaping suspended runtime %s — control-plane terminated the deployment",
                        deployment_id,
                    )
                    self._terminate_runtime(runtime, reason="lease_terminated")
                    self.repository.remove_runtime(deployment_id)
                continue
            if deployment_id in fleet_deployment_ids:
                # Lease still exists, just pinned to ANOTHER node_id. This box
                # ran it as a duplicate (pre-filter back-compat). Reclaim our
                # local GPUs/container but do NOT report TERMINATED — the
                # deployment is alive and healthy on its real node, and a
                # spurious report would mark it dead and tear down the tenant's
                # actual pod on the other box.
                logger.info(
                    "shedding duplicate runtime %s (lease pinned to another node) "
                    "— local teardown only, not reporting",
                    deployment_id,
                )
                self._terminate_runtime(runtime, reason="foreign_node_duplicate", report=False)
            else:
                # Lease is gone fleet-wide → genuine orphan. Terminate + report.
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
            # A distributed replica's rank role arrives on the LEASE (each rank
            # shares one workload but has its own role/rank/head_host). Stash it
            # so _start_inference_runtime can hand it to the launcher.
            metadata={"multi_node": lease.multi_node} if lease.multi_node else {},
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
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count, avoid=self._foreign_busy_devices())
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

        # Resolve runtime_kind from the most trustworthy source available.
        # The gateway's build_*_workload helpers set this on
        # `workload.runtime.runtime_kind` (pydantic field); legacy workloads
        # may have it in `workload.metadata["runtime_kind"]`; as a last
        # resort we fall back to image-name heuristics so a mislabeled
        # deployment still goes down the right code path.
        runtime_kind: str | None = None
        if getattr(workload, "runtime", None) is not None:
            rk = getattr(workload.runtime, "runtime_kind", None)
            if rk:
                runtime_kind = str(rk)
        if not runtime_kind:
            runtime_kind = workload.metadata.get("runtime_kind")
        if not runtime_kind and image:
            image_lower = image.lower()
            if "diffusion" in image_lower:
                runtime_kind = "diffusion"
            elif "vllm" in image_lower:
                runtime_kind = "vllm"
        if not runtime_kind:
            runtime_kind = "hf-causal-lm"

        runtime_manifest = {
            "runtime_kind": runtime_kind,
            "model_identifier": model_id,
            # vLLM is started with `--model <model_identifier>` (the HF repo
            # path like "meta-llama/Llama-3.2-1B-Instruct") so it loads
            # weights from the right place. But the gateway forwards
            # /v1/chat/completions with the workload's slug name (e.g.
            # "meta-llama-Llama-3.2-1B-Instruct"); without an alias vLLM
            # 404s on the slug. Surface workload.name here so inference.py
            # can pass it as --served-model-name and accept both.
            "serve_as_name": workload.name,
            "model_revision": workload.metadata.get("model_revision"),
            "tokenizer_identifier": workload.metadata.get("tokenizer_identifier"),
            "seed_corpus": workload.metadata.get("seed_corpus", [
                f"{image} serves greencompute inference requests",
                "miners keep deployments healthy with recovery failover and streaming completions",
            ]),
        }
        logger.info(
            "resolved runtime_kind=%s for %s (image=%s)",
            runtime_kind, runtime.deployment_id, image,
        )
        payload = {
            "build_id": build_id,
            "image": image,
            "docker_image": image,
            "runtime_manifest": runtime_manifest,
            # Shard the model across every GPU we just allocated. Without this
            # vLLM falls back to its TP=1 default and loads the whole model onto
            # a single card — the other allocated GPUs stay idle and any model
            # that needed them OOMs. Derived from the devices actually handed to
            # the container; workload metadata can override.
            "tensor_parallel_size": resolve_tensor_parallel_size(
                len(gpu_devices), workload.metadata.get("tensor_parallel_size")
            ),
        }
        # One rank of a distributed replica: the launcher reads this to decide
        # head-vs-worker, Ray coordinates, and the parallelism degrees.
        mn = (runtime.metadata or {}).get("multi_node")
        if isinstance(mn, dict) and mn:
            payload["multi_node"] = mn
        # User-supplied HuggingFace token (gated models). When present, the
        # node-agent injects it into the container env in preference to the
        # miner operator's own HF_TOKEN — lets a user deploy a gated model
        # even when the miner hasn't pre-configured creds for that repo.
        hf_token = workload.metadata.get("hf_token")
        if isinstance(hf_token, str) and hf_token.strip():
            payload["hf_token"] = hf_token.strip()

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

        # A worker rank of a distributed replica is READY (its GPUs have joined
        # the head's Ray cluster) but must never be advertised as routable — it
        # runs no API server. The gateway only considers deployments with an
        # endpoint, so leaving it None is what keeps inference off the workers;
        # only the head rank serves.
        is_worker = is_worker_runtime(runtime.metadata)
        runtime = runtime.model_copy(update={
            "status": "ready",
            "current_stage": "ready",
            "endpoint": (
                None if is_worker
                else f"{self.settings.miner_api_base_url}/inference/{runtime.deployment_id}"
            ),
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_pod_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a pod workload (Docker container with SSH)."""
        s = self.settings
        template_name = workload.metadata.get("template")
        logger.info("starting pod runtime for %s (template: %s)", runtime.deployment_id, template_name)

        # Pick an SSH port; generate a keypair only if the workload brought no
        # keys of its own.
        try:
            ssh_port = choose_free_port(s.ssh_port_range_start, s.ssh_port_range_end)
            private_key, ssh_public_keys = pod_ssh_keys(workload.metadata.get("ssh_public_keys"))
        except SSHError:
            logger.exception("SSH setup failed for %s", runtime.deployment_id)
            self._fail_runtime(runtime, "SSH setup failed")
            return

        # Allocate additional user-exposed TCP ports. User requests a list of
        # container ports; we pick free host ports from the user range.
        requested_ports_raw = workload.metadata.get("requested_ports") or []
        # Normalize + dedupe + cap at 10
        seen: set[int] = set()
        requested_ports: list[int] = []
        for p in requested_ports_raw:
            try:
                pi = int(p)
            except (TypeError, ValueError):
                continue
            if 1 <= pi <= 65535 and pi != 22 and pi not in seen:
                seen.add(pi)
                requested_ports.append(pi)
            if len(requested_ports) >= 10:
                break
        # Vast-style: prefer same-number mapping when the container port falls
        # inside our host-port range and the host port is actually free. Fall
        # back to random allocation otherwise.
        port_allocations: dict[int, int] = {}
        used_host_ports: set[int] = set()
        for container_port in requested_ports:
            in_range = s.user_port_range_start <= container_port <= s.user_port_range_end
            if in_range and container_port not in used_host_ports and is_port_free(container_port):
                host_port = container_port
            else:
                try:
                    host_port = choose_free_port(s.user_port_range_start, s.user_port_range_end)
                    while host_port in used_host_ports:
                        host_port = choose_free_port(s.user_port_range_start, s.user_port_range_end)
                except SSHError:
                    logger.warning(
                        "exhausted user port range for %s — allocated %d/%d requested",
                        runtime.deployment_id, len(port_allocations), len(requested_ports),
                    )
                    break
            port_allocations[container_port] = host_port
            used_host_ports.add(host_port)

        # Allocate specific GPU devices. Honor no-GPU templates (e.g.
        # ubuntu-ssh, gpu_fraction=0.0): a CPU-only pod must NOT reserve a whole
        # physical GPU. WorkloadRequirements.gpu_count is floored at 1 by the
        # protocol, so the template's gpu_fraction is the only signal that the
        # tenant asked for a CPU pod.
        gpu_count = workload.requirements.gpu_count if workload.requirements else 1
        if template_name:
            from greencompute_node_agent.domain.templates import get_template
            tpl = get_template(template_name)
            if tpl is not None and tpl.gpu_fraction == 0.0:
                gpu_count = 0
                logger.info(
                    "pod %s uses no-GPU template %r — skipping GPU allocation",
                    runtime.deployment_id, template_name,
                )
        try:
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count, avoid=self._foreign_busy_devices())
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
            from greencompute_node_agent.domain.templates import get_template
            tpl = get_template(template_name)
            if tpl:
                image = tpl.image

        # Compute per-pod CPU + RAM allocation proportional to GPU share
        # (RunPod-style: 1/N GPUs of host → 1/N of CPU and RAM). If the
        # workload explicitly requests a smaller slice, honor that; otherwise
        # cap at the fair share.
        host_gpus = max(s.gpu_count, 1)
        # CPU-only pods (gpu_count==0) still get one GPU-slot's worth of CPU/RAM
        # rather than the degenerate 1-core/1GB floor a 0 numerator would yield.
        sizing_gpu_units = gpu_count if gpu_count > 0 else 1
        gpu_fraction_of_host = sizing_gpu_units / host_gpus
        max_cpu = max(1.0, round(s.cpu_cores * gpu_fraction_of_host, 2))
        max_mem_gb = max(1, int(s.memory_gb * gpu_fraction_of_host))
        req_cpu = float(getattr(workload.requirements, "cpu_cores", 0) or 0) if workload.requirements else 0.0
        req_mem = int(getattr(workload.requirements, "memory_gb", 0) or 0) if workload.requirements else 0
        cpu_cores_allocated = min(max_cpu, req_cpu) if req_cpu > 0 else max_cpu
        memory_gb_allocated = min(max_mem_gb, req_mem) if req_mem > 0 else max_mem_gb
        logger.info(
            "pod %s allocated: gpus=%d/%d cpu=%.2f cores (max %.2f) memory=%dGB (max %dGB)",
            runtime.deployment_id, gpu_count, host_gpus,
            cpu_cores_allocated, max_cpu, memory_gb_allocated, max_mem_gb,
        )

        runtime = runtime.model_copy(update={
            "status": "starting",
            "current_stage": "start_pod",
            "template": template_name,
            "gpu_fraction": gpu_fraction,
            "cpu_cores_allocated": cpu_cores_allocated,
            "memory_gb_allocated": memory_gb_allocated,
            "ssh_host": s.ssh_host,
            "ssh_port": ssh_port,
            "ssh_username": "root",
            "ssh_fingerprint": None,
            "volume_path": volume.path,
            "volume_size_gb": volume_size_gb,
            "metadata": {
                **runtime.metadata,
                "image": image,
                "ssh_public_keys": ssh_public_keys,
                "ssh_private_key": private_key,
                "gpu_devices": gpu_devices,
                "gpu_count": gpu_count,
                "port_allocations": port_allocations,
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

        # Confirm sshd actually came up before reporting the pod ready, so the
        # customer isn't handed a pod they can't connect to. The check is
        # host-side (docker), immune to the node's NAT-hairpin (which made the
        # old public-IP socket check false-negative on every pod). Record the
        # result on the runtime so a genuine SSH failure is VISIBLE to ops
        # rather than silently reported ready.
        ssh_verified = True
        if hasattr(self.pod_backend, "wait_for_ready"):
            ssh_verified = self.pod_backend.wait_for_ready(runtime, timeout_seconds=90.0)
            if not ssh_verified:
                logger.warning(
                    "pod %s: sshd not confirmed up within timeout — reporting ready "
                    "but flagging ssh_verified=false (investigate the image/entrypoint)",
                    runtime.deployment_id,
                )

        # Build endpoint with SSH connection details
        ssh_endpoint = f"ssh://{runtime.ssh_username}@{runtime.ssh_host}:{runtime.ssh_port}"
        runtime = runtime.model_copy(update={
            "endpoint": ssh_endpoint,
            "metadata": {**runtime.metadata, "ssh_verified": ssh_verified},
        })
        self.repository.upsert_runtime(runtime)
        self._report_deployment_ready(runtime)

    def _start_vm_runtime(self, runtime: UnifiedRuntimeRecord, workload: WorkloadSpec) -> None:
        """Start a VM workload (Firecracker/stub)."""
        gpu_count = workload.requirements.gpu_count if workload.requirements else 1
        logger.info("starting VM runtime for %s (gpus: %d)", runtime.deployment_id, gpu_count)

        try:
            gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count, avoid=self._foreign_busy_devices())
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
        from greencompute_protocol import ChatCompletionRequest
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
                port_mappings=runtime.port_mappings or None,
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

    def _terminate_runtime(self, runtime: UnifiedRuntimeRecord, reason: str = "terminated", *, report: bool = True) -> None:
        """Tear down a runtime locally. When `report` is False we DON'T tell the
        control-plane it terminated — used for foreign-node duplicates (a lease
        that belongs to another box but ran here): we want to free our local
        GPUs/container without marking the real deployment (alive on its own
        node) as dead."""
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
        if report:
            try:
                self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                    deployment_id=runtime.deployment_id,
                    state=DeploymentState.TERMINATED,
                ))
            except ControlPlaneHTTPError:
                logger.exception("failed to report termination for %s", runtime.deployment_id)

    def _suspend_runtime(self, runtime: UnifiedRuntimeRecord) -> None:
        """Pod-safe SUSPEND (ORCH-H3). Stop the runtime in place and free GPUs,
        WITHOUT tearing it down. This is NOT a terminate:
          - POD: `docker stop` the container (pause_pod) — filesystem, volumes,
            SSH keys and port mappings are all preserved on disk so a later
            resume can `docker start` the SAME container. We NEVER `docker rm -f`.
          - INFERENCE: stop the (stateless, recreatable) serving container so the
            GPUs free up; on resume it is re-dispatched fresh from the workload.
          - VM: stop the VM; recreated on resume.
        In every case we release the GPU allocation and report SUSPENDED back to
        the control-plane. We do NOT report TERMINATED — the lease lives on.
        Only ever called for a lease the control-plane has EXPLICITLY marked
        status='suspended'."""
        logger.info(
            "suspending runtime %s (kind=%s) — docker stop / release GPUs, container preserved",
            runtime.deployment_id, runtime.workload_kind,
        )
        # Release the GPU allocation so the freed cards become schedulable.
        self.gpu_allocator.release(runtime.deployment_id)

        kind = runtime.workload_kind
        try:
            if kind in (WorkloadKind.POD, "pod"):
                # docker stop — preserves the container (NOT rm). pause_pod sets
                # status='suspended' and KEEPS container_id for resume.
                runtime = self.pod_backend.pause_pod(runtime)
            elif kind in (WorkloadKind.INFERENCE, "inference"):
                # Inference replicas are recreatable; stop_runtime tears the
                # container down (rm) but that is safe — there is no tenant state
                # to preserve. We then mark the record suspended so resume knows
                # to re-dispatch it fresh.
                if runtime.process_id or runtime.container_id:
                    runtime = self.inference_backend.stop_runtime(runtime)
                runtime = runtime.model_copy(update={
                    "status": "suspended",
                    "current_stage": "suspended",
                    "metadata": {**runtime.metadata, "suspended_at": _utcnow().isoformat()},
                    "updated_at": _utcnow(),
                })
            elif kind in (WorkloadKind.VM, "vm"):
                if runtime.vm_id:
                    self.vm_backend.stop_vm(runtime)
                runtime = runtime.model_copy(update={
                    "status": "suspended",
                    "current_stage": "suspended",
                    "metadata": {**runtime.metadata, "suspended_at": _utcnow().isoformat()},
                    "updated_at": _utcnow(),
                })
            else:
                runtime = runtime.model_copy(update={
                    "status": "suspended",
                    "current_stage": "suspended",
                    "updated_at": _utcnow(),
                })
        except Exception:
            logger.exception("backend suspend failed for %s — marking suspended anyway", runtime.deployment_id)
            runtime = runtime.model_copy(update={
                "status": "suspended",
                "current_stage": "suspended",
                "updated_at": _utcnow(),
            })

        self.repository.upsert_runtime(runtime)
        try:
            self.control_plane.update_deployment_status(DeploymentStatusUpdate(
                deployment_id=runtime.deployment_id,
                state=DeploymentState.SUSPENDED,
            ))
        except ControlPlaneHTTPError:
            logger.exception("failed to report SUSPENDED for %s", runtime.deployment_id)

    def _resume_runtime(self, runtime: UnifiedRuntimeRecord) -> None:
        """Pod-safe RESUME (ORCH-H3). Bring a suspended runtime back online:
          - re-acquire the GPU allocation (fails the resume if no capacity);
          - POD: `docker start` the SAME container (resume_pod) — original data,
            SSH keys and packages survive; report STARTING then READY;
          - INFERENCE/VM: the container/VM was removed on suspend, so re-dispatch
            it fresh from the workload spec.
        Only ever called when a previously-suspended runtime's lease has returned
        to a non-suspended (active) status."""
        gpu_count = int(runtime.metadata.get("gpu_count", runtime.gpu_fraction or 1) or 1)
        _is_pod = runtime.workload_kind in (WorkloadKind.POD, "pod")
        _original_devices = runtime.metadata.get("gpu_devices") or []
        try:
            if _is_pod and _original_devices:
                # A pod's container is pinned to its ORIGINAL --gpus devices, so it
                # MUST resume on exactly those cards — not a fresh free[:n] slice,
                # or the allocator's record would diverge from the container's
                # reality. Stays suspended if those exact cards aren't free.
                gpu_devices = self.gpu_allocator.allocate_specific(
                    runtime.deployment_id, _original_devices
                )
            else:
                gpu_devices = self.gpu_allocator.allocate(runtime.deployment_id, gpu_count, avoid=self._foreign_busy_devices())
        except GpuAllocationError as exc:
            # No capacity to resume — leave it suspended; the control-plane will
            # retry. Do NOT fail/terminate (that would destroy the preserved pod).
            logger.warning(
                "cannot resume %s — GPU re-allocation failed (%s); staying suspended",
                runtime.deployment_id, exc,
            )
            return

        kind = runtime.workload_kind
        if kind in (WorkloadKind.POD, "pod"):
            logger.info("resuming pod runtime %s — docker start (same container)", runtime.deployment_id)
            runtime = runtime.model_copy(update={
                "metadata": {**runtime.metadata, "gpu_devices": gpu_devices, "gpu_count": gpu_count},
            })
            try:
                runtime = self.pod_backend.resume_pod(runtime)
            except PodError as exc:
                # docker start failed — release the just-acquired GPUs and leave
                # the runtime suspended so a later reconcile retries. The
                # container itself is untouched (still exists on disk).
                logger.exception("pod resume failed for %s — staying suspended", runtime.deployment_id)
                self.gpu_allocator.release(runtime.deployment_id)
                runtime = runtime.model_copy(update={
                    "status": "suspended",
                    "current_stage": "suspended",
                    "last_error": f"resume failed: {exc}",
                    "updated_at": _utcnow(),
                })
                self.repository.upsert_runtime(runtime)
                return
            self.repository.upsert_runtime(runtime)
            self._report_deployment_ready(runtime)
            return

        # INFERENCE / VM: recreate from the workload spec (container/VM was
        # removed on suspend). Release the placeholder allocation we just took so
        # the start handler can allocate cleanly, then re-dispatch.
        self.gpu_allocator.release(runtime.deployment_id)
        workload = None
        try:
            workload = self.control_plane.get_workload(runtime.workload_id)
        except ControlPlaneHTTPError:
            logger.exception("failed to fetch workload for resume of %s — staying suspended", runtime.deployment_id)
        if workload is None:
            logger.warning(
                "workload %s not found for resume of %s — staying suspended",
                runtime.workload_id, runtime.deployment_id,
            )
            return
        # Reset to 'accepted' so the start handler treats it as a fresh dispatch.
        runtime = runtime.model_copy(update={
            "status": "accepted",
            "current_stage": "accepted",
            "updated_at": _utcnow(),
        })
        self.repository.upsert_runtime(runtime)
        if kind in (WorkloadKind.INFERENCE, "inference"):
            self._start_inference_runtime(runtime, workload)
        elif kind in (WorkloadKind.VM, "vm"):
            self._start_vm_runtime(runtime, workload)

    def terminate_deployment(self, deployment_id: str) -> dict[str, Any]:
        runtime = self.repository.get_runtime(deployment_id)
        if runtime is None:
            return {"status": "not_found"}
        self._terminate_runtime(runtime, reason="user_request")
        return {"status": "terminated", "deployment_id": deployment_id}

    # --- Recovery ---

    # Deployment states for which a persisted runtime should be KEPT (not torn
    # down) at startup recovery. Keep-by-default: anything legitimately in
    # flight (queued/pulling/starting), serving (ready), draining, or
    # intentionally suspended must survive a node-agent restart. Only
    # PENDING/FAILED/TERMINATED fall through to local teardown.
    _RECOVERY_KEEP_STATES = frozenset({
        DeploymentState.READY,
        DeploymentState.STARTING,
        DeploymentState.SCHEDULED,
        DeploymentState.PULLING,
        DeploymentState.DRAINING,
        DeploymentState.SUSPENDED,
    })

    def recover_runtime_state(self, hotkey: str) -> dict[str, Any]:
        """Recover persisted runtimes under the reconcile lock so it can't race
        a concurrent reconcile pass into double GPU allocation. Skips (returns
        busy) rather than blocking if a pass is already running."""
        if not self._reconcile_lock.acquire(blocking=False):
            logger.info("reconcile in progress — deferring recovery")
            return {"status": "busy", "resumed": 0, "terminated": 0}
        try:
            return self._recover_runtime_state_locked(hotkey)
        finally:
            self._reconcile_lock.release()

    def _recover_runtime_state_locked(self, hotkey: str) -> dict[str, Any]:
        """On startup, check persisted runtimes and tear down only the ones the
        control-plane confirms are gone/dead. Keep-by-default so a node-agent
        restart never kills in-flight or running pods; a control-plane blip is
        treated as "keep, reconcile later" rather than "terminate"."""
        resumed = 0
        terminated = 0
        for deployment_id, runtime in self.repository.snapshot_runtime_items():
            if runtime.status in ("terminated", "failed"):
                continue
            # Check if deployment still exists on control plane.
            try:
                deployment = self.control_plane.get_deployment(deployment_id)
            except ControlPlaneHTTPError:
                # Control-plane unreachable — KEEP the runtime and reconcile
                # later. A restart during a CP outage must NOT tear down live
                # pods; reconcile_once is the authority for genuine orphans
                # once the CP is reachable again.
                logger.warning(
                    "control-plane unreachable during recovery for %s — keeping runtime, will reconcile later",
                    deployment_id,
                )
                resumed += 1
                continue
            if deployment is None:
                # Explicit 404 → deployment is gone fleet-wide. Reclaim our
                # local GPUs/container but DON'T report TERMINATED (reconcile_once
                # is the authority for reporting), so we never spuriously mark a
                # deployment dead.
                logger.info(
                    "deployment %s not found on control-plane during recovery — local teardown",
                    deployment_id,
                )
                self._terminate_runtime(runtime, reason="recovery_missing", report=False)
                terminated += 1
                continue
            if deployment.state in self._RECOVERY_KEEP_STATES:
                resumed += 1
                continue
            # PENDING/FAILED/TERMINATED → tear down. Only report when the
            # control-plane already considers it dead, so we don't fight it.
            self._terminate_runtime(
                runtime,
                reason="stale_recovery",
                report=deployment.state in (DeploymentState.TERMINATED, DeploymentState.FAILED),
            )
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
