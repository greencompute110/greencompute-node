"""Unified node agent routes — agent lifecycle + inference + compute endpoints."""

from __future__ import annotations

from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from greenference_protocol import (
    CapacityUpdate,
    Heartbeat,
    MinerRegistration,
)

from greenference_node_agent.transport.security import validate_optional_auth

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


# --- Agent lifecycle endpoints ---


@router.post("/agent/v1/register")
def register(
    payload: MinerRegistration,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    result = _svc().onboard(payload)
    return result.model_dump(mode="json")


@router.post("/agent/v1/capacity")
def publish_capacity(
    payload: CapacityUpdate,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    result = _svc().publish_capacity(payload)
    return result.model_dump(mode="json")


@router.post("/agent/v1/heartbeat")
def publish_heartbeat(
    payload: Heartbeat,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    result = _svc().publish_heartbeat(payload)
    return result.model_dump(mode="json")


@router.get("/agent/v1/leases/{hotkey}")
def list_leases(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> list[dict]:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    leases = _svc().sync_leases(hotkey)
    return [lease.model_dump(mode="json") for lease in leases]


@router.post("/agent/v1/reconcile/{hotkey}")
def reconcile(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    _svc().reconcile_once(hotkey)
    return {"status": "ok"}


@router.post("/agent/v1/recovery/{hotkey}")
def recovery(
    hotkey: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return _svc().recover_runtime_state(hotkey)


# --- Runtime observability ---


@router.get("/agent/v1/runtimes")
def list_runtimes(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> list[dict]:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return [rt.model_dump(mode="json") for rt in _svc().repository.runtimes.values()]


@router.get("/agent/v1/runtimes/summary")
def runtime_summary(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return _svc().runtime_summary()


@router.get("/agent/v1/runtimes/{deployment_id}")
def get_runtime(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="runtime not found")
    return runtime.model_dump(mode="json")


@router.post("/agent/v1/deployments/{deployment_id}/chat/completions")
async def chat_completions(
    deployment_id: str,
    payload: dict,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    result = _svc().invoke_inference(deployment_id, payload)
    if result is None:
        raise HTTPException(status_code=404, detail="inference deployment not found or not ready")
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


@router.get("/inference/{deployment_id}/healthz")
def inference_healthz(deployment_id: str) -> dict:
    """Health check for a specific inference runtime."""
    runtime = _svc().repository.get_runtime(deployment_id)
    if runtime is None or runtime.status != "ready" or not runtime.runtime_url:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok", "deployment_id": deployment_id}


@router.post("/inference/{deployment_id}/v1/chat/completions")
async def inference_proxy(deployment_id: str, req: Request) -> StreamingResponse:
    """Proxy /v1/chat/completions to the runtime's vLLM container."""
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
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    access = _svc().get_ssh_access(deployment_id, include_private_key=include_private_key)
    if access is None:
        raise HTTPException(status_code=404, detail="SSH access not available for this deployment")
    return access.model_dump(mode="json")


@router.delete("/agent/v1/deployments/{deployment_id}/terminate")
def terminate_deployment(
    deployment_id: str,
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return _svc().terminate_deployment(deployment_id)


@router.get("/agent/v1/gpu-status")
def gpu_status(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return _svc().gpu_allocator.status()


@router.get("/agent/v1/fleet")
def fleet_status(
    x_agent_auth: str | None = Header(default=None, alias="X-Agent-Auth"),
) -> dict:
    validate_optional_auth(x_agent_auth, _cfg().agent_auth_secret)
    return _svc().fleet_status()
