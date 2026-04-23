"""VM lifecycle management — StubVMBackend and FirecrackerVMBackend."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any

from greencompute_protocol import UnifiedRuntimeRecord, WorkloadSpec


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VMError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str, stage: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.stage = stage


class VMBackend:
    """Abstract VM backend."""

    def start_vm(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def stop_vm(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        raise NotImplementedError

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        raise NotImplementedError


class StubVMBackend(VMBackend):
    """Stub VM backend for development/testing — returns realistic-looking VM state."""

    def __init__(self, *, backend_name: str = "stub-vm-backend") -> None:
        self.backend_name = backend_name

    def start_vm(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        vm_id = f"stub-vm-{runtime.deployment_id[:8]}"
        return runtime.model_copy(
            update={
                "vm_id": vm_id,
                "status": "ready",
                "current_stage": "ready",
                "ssh_host": runtime.ssh_host or "127.0.0.1",
                "ssh_port": runtime.ssh_port or 22,
                "ssh_username": "root",
                "metadata": {
                    **runtime.metadata,
                    "backend": self.backend_name,
                    "vm_id": vm_id,
                    "console_url": f"ws://127.0.0.1:8007/vms/{runtime.deployment_id}/console",
                    "started_at": _utcnow().isoformat(),
                    "stub": True,
                },
                "updated_at": _utcnow(),
            }
        )

    def stop_vm(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        return runtime.model_copy(
            update={
                "status": "terminated",
                "current_stage": "terminated",
                "vm_id": None,
                "metadata": {**runtime.metadata, "terminated_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        return {"status": "running", "healthy": True, "stub": True, "vm_id": runtime.vm_id}


class FirecrackerVMBackend(VMBackend):
    """Firecracker microVM backend (requires firecracker binary on the host)."""

    def __init__(
        self,
        *,
        backend_name: str = "firecracker-vm-backend",
        kernel_path: str = "/opt/greencompute/vmlinux",
        rootfs_path: str = "/opt/greencompute/rootfs.ext4",
    ) -> None:
        self.backend_name = backend_name
        self.kernel_path = kernel_path
        self.rootfs_path = rootfs_path

    def start_vm(
        self,
        runtime: UnifiedRuntimeRecord,
        workload: WorkloadSpec,
    ) -> UnifiedRuntimeRecord:
        try:
            result = subprocess.run(  # noqa: S603
                ["firecracker", "--version"],  # noqa: S607
                capture_output=True,
                timeout=5.0,
            )
            if result.returncode != 0:
                raise VMError(
                    "firecracker binary not functional",
                    failure_class="vm_start_failure",
                    stage="start_vm",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise VMError(
                f"firecracker not available: {exc}",
                failure_class="vm_start_failure",
                stage="start_vm",
            ) from exc

        vm_id = f"fc-{runtime.deployment_id[:8]}"
        return runtime.model_copy(
            update={
                "vm_id": vm_id,
                "status": "ready",
                "current_stage": "ready",
                "metadata": {
                    **runtime.metadata,
                    "backend": self.backend_name,
                    "vm_id": vm_id,
                    "kernel_path": self.kernel_path,
                    "rootfs_path": self.rootfs_path,
                    "started_at": _utcnow().isoformat(),
                },
                "updated_at": _utcnow(),
            }
        )

    def stop_vm(self, runtime: UnifiedRuntimeRecord) -> UnifiedRuntimeRecord:
        return runtime.model_copy(
            update={
                "status": "terminated",
                "current_stage": "terminated",
                "vm_id": None,
                "metadata": {**runtime.metadata, "terminated_at": _utcnow().isoformat()},
                "updated_at": _utcnow(),
            }
        )

    def health(self, runtime: UnifiedRuntimeRecord) -> dict[str, Any]:
        if not runtime.vm_id:
            return {"status": "not_started", "healthy": False}
        return {"status": "running", "healthy": True, "vm_id": runtime.vm_id}
