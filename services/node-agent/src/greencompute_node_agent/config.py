"""Unified node agent settings — superset of miner + compute configs."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    service_name: str = "greencompute-node-agent"
    enable_background_workers: bool = False
    worker_poll_interval_seconds: float = Field(default=1.0, ge=0.1)
    bootstrap_miner: bool = False

    # Persistence
    runtime_state_path: str = "/tmp/greencompute-node-runtime-state.json"
    artifact_cache_dir: str = "/tmp/greencompute-node-artifacts"
    volume_base_dir: str = "/tmp/greencompute-node-volumes"

    # Remote control-plane URL (Phase 1: HTTP client, no in-process import)
    control_plane_url: str = "http://127.0.0.1:8001"

    # Miner identity
    miner_hotkey: str = "node-local"
    miner_payout_address: str = "5FnodeLocal"
    miner_auth_secret: str = "greencompute-node-local-secret"
    miner_api_base_url: str = "http://127.0.0.1:8007"
    miner_validator_url: str = "http://127.0.0.1:8002"

    # Hardware
    node_id: str = "node-local"
    gpu_model: str = "a100"
    gpu_count: int = Field(default=1, ge=1)
    available_gpus: int = Field(default=1, ge=0)
    vram_gb_per_gpu: int = Field(default=80, ge=1)
    cpu_cores: int = Field(default=32, ge=1)
    memory_gb: int = Field(default=128, ge=1)
    performance_score: float = Field(default=1.0, ge=0.0)
    gpu_split_units: int = Field(default=100, ge=1)

    # Security
    security_tier: str = "standard"
    attestation_enabled: bool = False

    # Backends
    pod_backend: str = "process"  # process | stub | k8s
    vm_backend: str = "stub"  # stub | firecracker
    inference_backend: str = "docker"  # docker | process | fallback
    allow_runtime_fallback: bool = False

    # Workload kinds this node supports
    supported_workload_kinds: list[str] = Field(
        default_factory=lambda: ["inference", "pod", "vm"]
    )

    # SSH access (for pod/VM workloads)
    ssh_host: str = "127.0.0.1"
    ssh_port_range_start: int = Field(default=30000, ge=1024)
    ssh_port_range_end: int = Field(default=31000, ge=1025)

    # Extra TCP ports the user can expose from their pod (apps, databases, APIs).
    # Host-side bindings drawn from this range. Keep separate from SSH range.
    # Hard cap of 10 ports per pod is enforced in pod.py.
    user_port_range_start: int = Field(default=31001, ge=1024)
    user_port_range_end: int = Field(default=32000, ge=1025)

    # Per-pod disk quota enforcement. None = auto-detect. Valid values:
    # "loop_mount" | "loop_mount_sudo" | "storage_opt" | "none".
    disk_enforcement_mode: str | None = None

    # Auth mode: "hotkey" (ed25519, production) or "hmac" (shared secret, dev)
    auth_mode: str = "hmac"
    # Bittensor wallet names for ed25519 signing (reads from ~/.bittensor/wallets/)
    coldkey_name: str | None = None
    hotkey_name: str = "default"

    # Auth (optional, for route protection)
    agent_auth_secret: str | None = None
    inference_auth_secret: str | None = None
    compute_auth_secret: str | None = None


def load_settings() -> Settings:
    miner_hotkey = os.getenv("GREENCOMPUTE_MINER_HOTKEY", "node-local")
    # node_id is the PRIMARY KEY of the control-plane's node_inventory table,
    # so two miners sharing the same `node_id` will silently overwrite each
    # other's capacity updates. If the operator hasn't set
    # GREENCOMPUTE_MINER_NODE_ID, derive a unique default from the hotkey
    # so fresh installs can't collide out-of-the-box.
    default_node_id = f"node-{miner_hotkey[:12]}" if miner_hotkey else "node-local"
    return Settings(
        enable_background_workers=_env_bool("GREENCOMPUTE_ENABLE_BACKGROUND_WORKERS", False),
        worker_poll_interval_seconds=float(os.getenv("GREENCOMPUTE_WORKER_POLL_INTERVAL_SECONDS", "1.0")),
        bootstrap_miner=_env_bool("GREENCOMPUTE_BOOTSTRAP_MINER", False),
        runtime_state_path=os.getenv("GREENCOMPUTE_RUNTIME_STATE_PATH", "/tmp/greencompute-node-runtime-state.json"),
        artifact_cache_dir=os.getenv("GREENCOMPUTE_ARTIFACT_CACHE_DIR", "/tmp/greencompute-node-artifacts"),
        volume_base_dir=os.getenv("GREENCOMPUTE_VOLUME_BASE_DIR", "/tmp/greencompute-node-volumes"),
        control_plane_url=os.getenv("GREENCOMPUTE_CONTROL_PLANE_URL", "http://127.0.0.1:8001"),
        miner_hotkey=miner_hotkey,
        miner_payout_address=os.getenv("GREENCOMPUTE_MINER_PAYOUT_ADDRESS", "5FnodeLocal"),
        miner_auth_secret=os.getenv("GREENCOMPUTE_MINER_AUTH_SECRET", "greencompute-node-local-secret"),
        miner_api_base_url=os.getenv("GREENCOMPUTE_MINER_API_BASE_URL", "http://127.0.0.1:8007"),
        miner_validator_url=os.getenv("GREENCOMPUTE_MINER_VALIDATOR_URL", "http://127.0.0.1:8002"),
        node_id=os.getenv("GREENCOMPUTE_MINER_NODE_ID", default_node_id),
        gpu_model=os.getenv("GREENCOMPUTE_GPU_MODEL", "a100"),
        gpu_count=int(os.getenv("GREENCOMPUTE_GPU_COUNT", "1")),
        available_gpus=int(os.getenv("GREENCOMPUTE_GPU_COUNT", "1")),
        vram_gb_per_gpu=int(os.getenv("GREENCOMPUTE_VRAM_GB_PER_GPU", "80")),
        cpu_cores=int(os.getenv("GREENCOMPUTE_CPU_CORES", "32")),
        memory_gb=int(os.getenv("GREENCOMPUTE_MEMORY_GB", "128")),
        performance_score=float(os.getenv("GREENCOMPUTE_PERFORMANCE_SCORE", "1.0")),
        gpu_split_units=int(os.getenv("GREENCOMPUTE_GPU_SPLIT_UNITS", "100")),
        security_tier=os.getenv("GREENCOMPUTE_SECURITY_TIER", "standard"),
        attestation_enabled=_env_bool("GREENCOMPUTE_ATTESTATION_ENABLED", False),
        pod_backend=os.getenv("GREENCOMPUTE_POD_BACKEND", "process"),
        vm_backend=os.getenv("GREENCOMPUTE_VM_BACKEND", "stub"),
        inference_backend=os.getenv("GREENCOMPUTE_INFERENCE_BACKEND", "process"),
        allow_runtime_fallback=_env_bool("GREENCOMPUTE_ALLOW_RUNTIME_FALLBACK", False),
        supported_workload_kinds=os.getenv("GREENCOMPUTE_SUPPORTED_WORKLOAD_KINDS", "inference,pod,vm").split(","),
        ssh_host=os.getenv("GREENCOMPUTE_SSH_HOST", "127.0.0.1"),
        ssh_port_range_start=int(os.getenv("GREENCOMPUTE_SSH_PORT_RANGE_START", "30000")),
        ssh_port_range_end=int(os.getenv("GREENCOMPUTE_SSH_PORT_RANGE_END", "31000")),
        user_port_range_start=int(os.getenv("GREENCOMPUTE_USER_PORT_RANGE_START", "31001")),
        user_port_range_end=int(os.getenv("GREENCOMPUTE_USER_PORT_RANGE_END", "32000")),
        disk_enforcement_mode=os.getenv("GREENCOMPUTE_DISK_ENFORCEMENT_MODE") or None,
        auth_mode=os.getenv("GREENCOMPUTE_AUTH_MODE", "hmac"),
        coldkey_name=os.getenv("GREENCOMPUTE_COLDKEY_NAME") or None,
        hotkey_name=os.getenv("GREENCOMPUTE_HOTKEY_NAME", "default"),
        agent_auth_secret=os.getenv("GREENCOMPUTE_AGENT_AUTH_SECRET") or None,
        inference_auth_secret=os.getenv("GREENCOMPUTE_INFERENCE_AUTH_SECRET") or None,
        compute_auth_secret=os.getenv("GREENCOMPUTE_COMPUTE_AUTH_SECRET") or None,
    )
