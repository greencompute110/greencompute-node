"""FastAPI application entry point for the unified greencompute node agent."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, status

from greencompute_protocol import Heartbeat, MinerRegistration

from greencompute_node_agent.application.services import NodeAgentService
from greencompute_node_agent.config import load_settings
from greencompute_node_agent.transport.routes import router, set_service

logger = logging.getLogger(__name__)

settings = load_settings()
service = NodeAgentService(settings)

_worker_state: dict[str, object | None] = {
    "running": False,
    "last_iteration": None,
    "bootstrapped": False,
    "last_recovery_at": None,
    "resumed_runtimes": 0,
    "terminated_stale_runtimes": 0,
}


def _bootstrap() -> None:
    registration = MinerRegistration(
        hotkey=settings.miner_hotkey,
        payout_address=settings.miner_payout_address,
        auth_secret=settings.miner_auth_secret,
        api_base_url=settings.miner_api_base_url,
        validator_url=settings.miner_validator_url,
    )
    service.onboard(registration)
    service.publish_heartbeat(Heartbeat(hotkey=settings.miner_hotkey, healthy=True))
    capacity = service.build_capacity_update()
    service.publish_capacity(capacity)
    recovery = service.recover_runtime_state(settings.miner_hotkey)
    _worker_state["last_recovery_at"] = recovery["last_recovery_at"]
    _worker_state["resumed_runtimes"] = recovery["resumed_runtimes"]
    _worker_state["terminated_stale_runtimes"] = recovery["terminated_stale_runtimes"]
    _worker_state["bootstrapped"] = True


def _heartbeat_tick() -> None:
    """Send heartbeat + capacity update. Must stay fast — no heavy I/O."""
    service.publish_heartbeat(Heartbeat(hotkey=settings.miner_hotkey, healthy=True))
    capacity = service.build_capacity_update()
    service.publish_capacity(capacity)


def _reconcile_tick() -> None:
    """Reconcile workloads — may take minutes (docker pull, health checks)."""
    service.reconcile_once(settings.miner_hotkey)
    _worker_state["last_iteration"] = datetime.now(UTC).isoformat()


async def _heartbeat_loop() -> None:
    """Dedicated heartbeat loop — keeps the miner alive regardless of reconciliation."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            if settings.bootstrap_miner and _worker_state["bootstrapped"]:
                await loop.run_in_executor(None, _heartbeat_tick)
        except Exception:
            logger.exception("heartbeat loop error")
        await asyncio.sleep(settings.worker_poll_interval_seconds)


async def _worker_loop() -> None:
    _worker_state["running"] = True
    loop = asyncio.get_running_loop()
    while True:
        try:
            if settings.bootstrap_miner and not _worker_state["bootstrapped"]:
                await loop.run_in_executor(None, _bootstrap)
            if settings.bootstrap_miner:
                await loop.run_in_executor(None, _reconcile_tick)
        except Exception:
            logger.exception("worker loop error")
        await asyncio.sleep(settings.worker_poll_interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    set_service(service, settings)
    tasks: list[asyncio.Task] = []
    if settings.enable_background_workers:
        tasks.append(asyncio.create_task(_worker_loop()))
        tasks.append(asyncio.create_task(_heartbeat_loop()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(
    title="GreenCompute Node Agent",
    version="0.1.0",
    description="Unified GPU node agent — inference, pod, and VM workloads.",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/healthz")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        "workers_enabled": settings.enable_background_workers,
    }


@app.get("/livez")
def liveness() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict:
    payload: dict = {
        "status": "ok",
        "service": settings.service_name,
        "bootstrapped": bool(_worker_state["bootstrapped"]) if settings.bootstrap_miner else True,
    }
    if settings.enable_background_workers:
        payload["workers_enabled"] = True
        payload["worker_running"] = bool(_worker_state["running"])
        payload["worker_last_iteration"] = _worker_state["last_iteration"]
        summary = service.runtime_summary()
        payload["runtime_count"] = summary["total"]
        payload["runtime_status_breakdown"] = summary["by_status"]
        payload["runtime_kind_breakdown"] = summary["by_kind"]
        payload["failed_runtime_count"] = summary["failed"]
        payload["last_recovery_at"] = _worker_state["last_recovery_at"]
        payload["resumed_runtimes"] = _worker_state["resumed_runtimes"]
        payload["terminated_stale_runtimes"] = _worker_state["terminated_stale_runtimes"]
    return payload
