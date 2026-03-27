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
    service_name: str = "greenference-node-agent"
    enable_background_workers: bool = False
    worker_poll_interval_seconds: float = Field(default=1.0, ge=0.1)
    bootstrap_miner: bool = False

    # Persistence
    runtime_state_path: str = "/tmp/greenference-node-runtime-state.json"
    artifact_cache_dir: str = "/tmp/greenference-node-artifacts"
    volume_base_dir: str = "/tmp/greenference-node-volumes"

    # Remote control-plane URL (Phase 1: HTTP client, no in-process import)
    control_plane_url: str = "http://127.0.0.1:8001"

    # Miner identity
    miner_hotkey: str = "node-local"
    miner_payout_address: str = "5FnodeLocal"
    miner_auth_secret: str = "greenference-node-local-secret"
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
    inference_backend: str = "process"  # process | k8s
    allow_runtime_fallback: bool = False

    # Workload kinds this node supports
    supported_workload_kinds: list[str] = Field(
        default_factory=lambda: ["inference", "pod", "vm"]
    )

    # SSH access (for pod/VM workloads)
    ssh_host: str = "127.0.0.1"
    ssh_port_range_start: int = Field(default=30000, ge=1024)
    ssh_port_range_end: int = Field(default=31000, ge=1025)

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
    return Settings(
        enable_background_workers=_env_bool("GREENFERENCE_ENABLE_BACKGROUND_WORKERS", False),
        worker_poll_interval_seconds=float(os.getenv("GREENFERENCE_WORKER_POLL_INTERVAL_SECONDS", "1.0")),
        bootstrap_miner=_env_bool("GREENFERENCE_BOOTSTRAP_MINER", False),
        runtime_state_path=os.getenv("GREENFERENCE_RUNTIME_STATE_PATH", "/tmp/greenference-node-runtime-state.json"),
        artifact_cache_dir=os.getenv("GREENFERENCE_ARTIFACT_CACHE_DIR", "/tmp/greenference-node-artifacts"),
        volume_base_dir=os.getenv("GREENFERENCE_VOLUME_BASE_DIR", "/tmp/greenference-node-volumes"),
        control_plane_url=os.getenv("GREENFERENCE_CONTROL_PLANE_URL", "http://127.0.0.1:8001"),
        miner_hotkey=os.getenv("GREENFERENCE_MINER_HOTKEY", "node-local"),
        miner_payout_address=os.getenv("GREENFERENCE_MINER_PAYOUT_ADDRESS", "5FnodeLocal"),
        miner_auth_secret=os.getenv("GREENFERENCE_MINER_AUTH_SECRET", "greenference-node-local-secret"),
        miner_api_base_url=os.getenv("GREENFERENCE_MINER_API_BASE_URL", "http://127.0.0.1:8007"),
        miner_validator_url=os.getenv("GREENFERENCE_MINER_VALIDATOR_URL", "http://127.0.0.1:8002"),
        node_id=os.getenv("GREENFERENCE_MINER_NODE_ID", "node-local"),
        gpu_model=os.getenv("GREENFERENCE_GPU_MODEL", "a100"),
        gpu_count=int(os.getenv("GREENFERENCE_GPU_COUNT", "1")),
        available_gpus=int(os.getenv("GREENFERENCE_GPU_COUNT", "1")),
        vram_gb_per_gpu=int(os.getenv("GREENFERENCE_VRAM_GB_PER_GPU", "80")),
        cpu_cores=int(os.getenv("GREENFERENCE_CPU_CORES", "32")),
        memory_gb=int(os.getenv("GREENFERENCE_MEMORY_GB", "128")),
        performance_score=float(os.getenv("GREENFERENCE_PERFORMANCE_SCORE", "1.0")),
        gpu_split_units=int(os.getenv("GREENFERENCE_GPU_SPLIT_UNITS", "100")),
        security_tier=os.getenv("GREENFERENCE_SECURITY_TIER", "standard"),
        attestation_enabled=_env_bool("GREENFERENCE_ATTESTATION_ENABLED", False),
        pod_backend=os.getenv("GREENFERENCE_POD_BACKEND", "process"),
        vm_backend=os.getenv("GREENFERENCE_VM_BACKEND", "stub"),
        inference_backend=os.getenv("GREENFERENCE_INFERENCE_BACKEND", "process"),
        allow_runtime_fallback=_env_bool("GREENFERENCE_ALLOW_RUNTIME_FALLBACK", False),
        supported_workload_kinds=os.getenv("GREENFERENCE_SUPPORTED_WORKLOAD_KINDS", "inference,pod,vm").split(","),
        ssh_host=os.getenv("GREENFERENCE_SSH_HOST", "127.0.0.1"),
        ssh_port_range_start=int(os.getenv("GREENFERENCE_SSH_PORT_RANGE_START", "30000")),
        ssh_port_range_end=int(os.getenv("GREENFERENCE_SSH_PORT_RANGE_END", "31000")),
        auth_mode=os.getenv("GREENFERENCE_AUTH_MODE", "hmac"),
        coldkey_name=os.getenv("GREENFERENCE_COLDKEY_NAME") or None,
        hotkey_name=os.getenv("GREENFERENCE_HOTKEY_NAME", "default"),
        agent_auth_secret=os.getenv("GREENFERENCE_AGENT_AUTH_SECRET") or None,
        inference_auth_secret=os.getenv("GREENFERENCE_INFERENCE_AUTH_SECRET") or None,
        compute_auth_secret=os.getenv("GREENFERENCE_COMPUTE_AUTH_SECRET") or None,
    )
