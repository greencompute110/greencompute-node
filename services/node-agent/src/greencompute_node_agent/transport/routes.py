"""Unified node agent routes — agent lifecycle + inference + compute endpoints."""

from __future__ import annotations

from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from greencompute_protocol import (
    WorkloadKind,
    CapacityUpdate,
    Heartbeat,
    MinerRegistration,
)

from greencompute_node_agent.transport.redaction import redact
from greencompute_node_agent.transport.security import validate_auth

router = APIRouter()

# Service instance — set by main.py during app startup
_service = None
_settings = None


def set_service(svc: Any, settings: Any) -> None:
    global _service, _settings
    _service = svc
    _settings = settings


def _svc():
    if _service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return _service


def _cfg():
    if _settings is None:
        raise HTTPException(status_code=503, detail="settings not initialized")
    return _settings


def _runtime_public_dump(runtime: Any) -> dict:
    """Serialize a runtime record with secret metadata fields redacted."""
    return redact(runtime.model_dump(mode="json"))


# --- Agent lifecycle endpoints ---


@router.post("/agent/v1/register")
def register(
    payload: MinerRegistration,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    result = _svc().onboard(payload)
    return result.model_dump(mode="json")


@router.post("/agent/v1/capacity")
def publish_capacity(
    payload: CapacityUpdate,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    result = _svc().publish_capacity(payload)
    return result.model_dump(mode="json")


@router.post("/agent/v1/heartbeat")
def publish_heartbeat(
    payload: Heartbeat,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    result = _svc().publish_heartbeat(payload)
    return result.model_dump(mode="json")


@router.get("/agent/v1/leases/{hotkey}")
def list_leases(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> list[dict]:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    leases = _svc().sync_leases(hotkey)
    if leases is None:
        raise HTTPException(status_code=502, detail="the control plane didn't return a lease list")
    return [lease.model_dump(mode="json") for lease in leases]


@router.post("/agent/v1/reconcile/{hotkey}")
def reconcile(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    _svc().reconcile_once(hotkey)
    return {"status": "ok"}


@router.post("/agent/v1/recovery/{hotkey}")
def recovery(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return _svc().recover_runtime_state(hotkey)


# --- Runtime observability ---


@router.get("/agent/v1/runtimes")
def list_runtimes(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> list[dict]:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return [_runtime_public_dump(rt) for rt in _svc().repository.snapshot_runtimes()]


@router.get("/agent/v1/runtimes/summary")
def runtime_summary(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return _svc().runtime_summary()


@router.get("/agent/v1/runtimes/{deployment_id}")
def get_runtime(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="runtime not found")
    return _runtime_public_dump(runtime)


@router.post("/agent/v1/deployments/{deployment_id}/chat/completions")
async def chat_completions(
    deployment_id: str,
    payload: dict,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    # Fail-closed: the gateway reaches inference via /inference/{id}/v1/chat/completions,
    # not this /agent/v1 alias, so closing it removes a second unauthenticated entrypoint.
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    result = _svc().invoke_inference(deployment_id, payload)
    if result is None:
        raise HTTPException(status_code=404, detail="inference deployment not found or not ready")
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


@router.get("/inference/{deployment_id}/healthz")
def inference_healthz(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    """Health check for a specific inference runtime."""
    validate_auth(x_agent_auth, _cfg(), sensitive=False)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or runtime.status != "ready" or not runtime.runtime_url:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok", "deployment_id": deployment_id}


@router.post("/pods/{deployment_id}/exec")
def pod_exec(
    deployment_id: str,
    body: dict,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    """Run a command in the pod, as root, from /workspace: {command,
    timeout_seconds (1-120)}. Only with the node secret (sensitive)."""
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    from greencompute_node_agent.domain.pod_exec import MAX_COMMAND_CHARS, MAX_TIMEOUT_SECONDS, run_in_pod

    command = body.get("command") if isinstance(body, dict) else None
    if not isinstance(command, str) or not command.strip() or len(command) > MAX_COMMAND_CHARS or "\x00" in command:
        raise HTTPException(status_code=400, detail=f"command must be 1-{MAX_COMMAND_CHARS} characters")
    timeout = body.get("timeout_seconds", 60)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise HTTPException(status_code=400, detail=f"timeout_seconds must be 1-{MAX_TIMEOUT_SECONDS}")
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or not runtime.container_id or runtime.status != "ready":
        raise HTTPException(status_code=409, detail="the pod isn't running")
    if runtime.workload_kind not in ("pod", WorkloadKind.POD):
        raise HTTPException(status_code=409, detail="commands run only in rentals (pods)")
    from greencompute_node_agent.domain.pod_exec import BusyError

    try:
        return run_in_pod(runtime.container_id, command, timeout_seconds=timeout)
    except BusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc


@router.get("/pods/{deployment_id}/logs")
def pod_logs(
    deployment_id: str,
    lines: int = 200,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    """The pod's recent output: its start command's log and its container
    log, the last `lines` lines of each (at most 2000). A job's output can
    hold secrets, so only with the node secret (sensitive)."""
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or not runtime.container_id:
        raise HTTPException(status_code=404, detail="runtime not found")
    from greencompute_node_agent.domain.pod_logs import collect_pod_logs

    env = runtime.metadata.get("env_vars") or {}
    return collect_pod_logs(runtime.container_id, lines=lines,
                            has_command=isinstance(env, dict) and bool(env.get("GC_START_COMMAND_B64")))


@router.get("/pods/{deployment_id}/stats")
def pod_stats(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    """Live resource stats for a running pod/inference runtime.

    Best-effort — any probe that fails is omitted from the response so the UI
    can show partial data. Returns at most a few dozen bytes of JSON.
    """
    validate_auth(x_agent_auth, _cfg(), sensitive=False)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or not runtime.container_id:
        raise HTTPException(status_code=404, detail="runtime not found")
    if runtime.status != "ready":
        # Runtime exists but not running — return empty stats rather than 404
        # so the UI can keep polling without flipping into an error state.
        return {}
    from greencompute_node_agent.domain.pod_stats import collect_pod_stats
    gpu_devices = runtime.metadata.get("gpu_devices") or []
    try:
        gpu_device_ids = [int(d) for d in gpu_devices]
    except (TypeError, ValueError):
        gpu_device_ids = []
    return collect_pod_stats(
        container_id=runtime.container_id,
        gpu_device_ids=gpu_device_ids,
        workspace_path="/workspace",
    )


@router.post("/inference/{deployment_id}/v1/chat/completions")
async def inference_proxy(
    deployment_id: str,
    req: Request,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> StreamingResponse:
    """Proxy /v1/chat/completions to the runtime's vLLM container.

    Requires the same `X-Agent-Auth` header all other node-agent routes
    expect. Without this check, anyone reachable on port 8007 (especially
    over a public IP) could bypass the gateway's balance / rate-limit
    enforcement and invoke models directly via the deployment_id.
    """
    validate_auth(x_agent_auth, _cfg(), sensitive=False)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or runtime.status != "ready" or not runtime.runtime_url:
        raise HTTPException(status_code=404, detail="inference runtime not found or not ready")
    body = await req.body()
    upstream_url = f"{runtime.runtime_url}/v1/chat/completions"
    upstream_req = urllib_request.Request(upstream_url, data=body, method="POST")
    upstream_req.add_header("content-type", "application/json")
    try:
        resp = urllib_request.urlopen(upstream_req, timeout=120.0)  # noqa: S310
    except (HTTPError, URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    def _stream():
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                yield line
        finally:
            resp.close()

    content_type = resp.headers.get("content-type", "application/json")
    return StreamingResponse(_stream(), media_type=content_type)


@router.get("/agent/v1/deployments/{deployment_id}/ssh")
def get_ssh_access(
    deployment_id: str,
    include_private_key: bool = False,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    access = _svc().get_ssh_access(deployment_id, include_private_key=include_private_key)
    if access is None:
        raise HTTPException(status_code=404, detail="SSH access not available for this deployment")
    return access.model_dump(mode="json")


@router.delete("/agent/v1/deployments/{deployment_id}/terminate")
def terminate_deployment(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return _svc().terminate_deployment(deployment_id)


@router.get("/agent/v1/gpu-status")
def gpu_status(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return _svc().gpu_allocator.status()


@router.get("/agent/v1/fleet")
def fleet_status(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_auth(x_agent_auth, _cfg(), sensitive=True)
    return _svc().fleet_status()
