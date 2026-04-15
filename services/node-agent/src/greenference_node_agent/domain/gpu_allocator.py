"""GPU device allocator — tracks which physical GPUs are in use."""

from __future__ import annotations

import logging
from typing import Any

from greenference_node_agent.domain.gpu_docker import gpu_docker_flags

logger = logging.getLogger(__name__)


class GpuAllocationError(RuntimeError):
    pass


class GpuAllocator:
    """Tracks per-device GPU allocation on a single node.

    Given a node with N GPUs (device IDs 0..N-1), allocates specific
    devices to workloads and prevents double-allocation.
    """

    def __init__(self, total_gpus: int) -> None:
        self.total_gpus = total_gpus
        self.device_ids = set(range(total_gpus))
        # deployment_id → set of allocated device IDs
        self._allocations: dict[str, set[int]] = {}

    @property
    def free_devices(self) -> set[int]:
        used = set()
        for devices in self._allocations.values():
            used |= devices
        return self.device_ids - used

    @property
    def free_count(self) -> int:
        return len(self.free_devices)

    @property
    def used_count(self) -> int:
        return self.total_gpus - self.free_count

    def allocate(self, deployment_id: str, gpu_count: int) -> list[int]:
        """Allocate gpu_count devices. Returns sorted list of device IDs.

        Raises GpuAllocationError if not enough GPUs are free.
        """
        if gpu_count <= 0:
            return []
        if gpu_count > self.total_gpus:
            raise GpuAllocationError(
                f"requested {gpu_count} GPUs but node only has {self.total_gpus}"
            )
        free = sorted(self.free_devices)
        if len(free) < gpu_count:
            raise GpuAllocationError(
                f"requested {gpu_count} GPUs but only {len(free)} free "
                f"(total={self.total_gpus}, used by {len(self._allocations)} workloads)"
            )
        allocated = free[:gpu_count]
        self._allocations[deployment_id] = set(allocated)
        logger.info(
            "allocated GPUs %s to %s (%d/%d now used)",
            allocated, deployment_id, self.used_count, self.total_gpus,
        )
        return allocated

    def release(self, deployment_id: str) -> list[int]:
        """Release GPUs for a deployment. Returns the freed device IDs."""
        devices = self._allocations.pop(deployment_id, set())
        if devices:
            logger.info(
                "released GPUs %s from %s (%d/%d now used)",
                sorted(devices), deployment_id, self.used_count, self.total_gpus,
            )
        return sorted(devices)

    def get_allocation(self, deployment_id: str) -> list[int]:
        """Get device IDs allocated to a deployment."""
        return sorted(self._allocations.get(deployment_id, set()))

    def status(self) -> dict[str, Any]:
        return {
            "total_gpus": self.total_gpus,
            "free_gpus": self.free_count,
            "used_gpus": self.used_count,
            "allocations": {
                dep_id: sorted(devices)
                for dep_id, devices in self._allocations.items()
            },
        }

    def docker_gpu_flag(self, deployment_id: str) -> list[str]:
        """Return Docker flags for GPU passthrough (auto-detected method)."""
        devices = self.get_allocation(deployment_id)
        return gpu_docker_flags(devices if devices else None)
