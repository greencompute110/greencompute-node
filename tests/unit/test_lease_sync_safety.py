"""A control-plane error is never "no leases" (2026-10-29/30): sync_leases
used to return [] on any HTTP error, and the orphan clean-up then removed
every pod and model on the box. A customer's running pod went that way when
.88 failed, and both catalog models during a deploy."""
from types import SimpleNamespace

import pytest

from greencompute_node_agent.application.services import ControlPlaneUnavailable, NodeAgentService, _NodeControlPlaneClient
from greencompute_protocol import ControlPlaneHTTPError


def _agent(list_leases):
    svc = object.__new__(NodeAgentService)
    svc.control_plane = SimpleNamespace(list_leases=list_leases)
    svc.settings = SimpleNamespace(node_id="node-1")
    touched = []

    def forbidden(*a, **k):
        touched.append(a)
        raise AssertionError("nothing may be judged or changed without a lease list")
    svc.repository = SimpleNamespace(snapshot_runtime_items=forbidden, get_runtime=forbidden)
    svc._terminate_runtime = forbidden
    return svc, touched


@pytest.mark.parametrize("status", [500, 502, 503, 401, 404])
def test_an_error_from_the_control_plane_skips_the_pass_and_removes_nothing(status):
    def fails(hotkey):
        raise ControlPlaneHTTPError(status, "Internal Server Error")
    svc, touched = _agent(fails)
    assert svc.sync_leases("hk") is None
    svc._reconcile_once_locked("hk")  # returns at once
    assert touched == []


def test_a_real_empty_list_is_still_an_answer():
    svc, _ = _agent(lambda hotkey: [])
    assert svc.sync_leases("hk") == []


# --- the rest of the review (2026-10-30): no verdict without an answer ---------

from urllib.error import URLError  # noqa: E402

from greencompute_protocol import DeploymentRecord, DeploymentState, LeaseAssignment, UnifiedRuntimeRecord, WorkloadKind  # noqa: E402


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("timed out"), ValueError("not JSON")])
def test_no_answer_of_any_kind_is_no_lease_list(error):
    def fails(hotkey):
        raise error
    svc, touched = _agent(fails)
    assert svc.sync_leases("hk") is None
    svc._reconcile_once_locked("hk")
    assert touched == []


def _running(dep_id="d1"):
    return UnifiedRuntimeRecord(deployment_id=dep_id, workload_id="w1", hotkey="hk", node_id="node-1",
                                workload_kind=WorkloadKind.POD, container_id="c1", status="ready")


class World:
    """A node with one running pod, a control plane whose lease list comes back empty, and
    a get_deployment answer chosen per test."""

    def __init__(self, answer):
        self.svc = object.__new__(NodeAgentService)
        self.terminated = []
        self.reports = []
        runtime = _running()
        self.svc.settings = SimpleNamespace(node_id="node-1")
        self.saved = {runtime.deployment_id: runtime}
        self.asks = 0
        self.svc.repository = SimpleNamespace(
            snapshot_runtime_items=lambda: list(self.saved.items()),
            get_runtime=lambda dep_id: None,
            remove_runtime=lambda dep_id: None,
            upsert_runtime=lambda rt: self.saved.__setitem__(rt.deployment_id, rt) or rt,
        )

        def get_deployment(dep_id):
            self.asks += 1
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.svc.control_plane = SimpleNamespace(list_leases=lambda hk: [], get_deployment=get_deployment,
                                                 update_deployment_status=self.reports.append)
        self.svc._terminate_runtime = lambda rt, reason="", report=True: self.terminated.append((rt.deployment_id, reason))


def _dep(state):
    return DeploymentRecord(deployment_id="d1", workload_id="w1", state=state)


@pytest.mark.parametrize("answer", [_dep(DeploymentState.READY), _dep(DeploymentState.STARTING),
                                    ControlPlaneUnavailable("down"), ControlPlaneHTTPError(500, "boom")])
def test_an_empty_lease_list_never_removes_a_pod_the_control_plane_still_runs(answer):
    world = World(answer)
    world.svc._reconcile_once_locked("hk")
    assert world.terminated == []
    # Asked again only after a few minutes, not on every tick.
    world.svc._reconcile_once_locked("hk")
    assert world.asks == 1


def test_the_orphan_clean_up_never_reports_so_it_cannot_race_a_new_placement():
    world = World(None)  # 404: deleted, or moved to another machine
    calls = []
    world.svc._terminate_runtime = lambda rt, reason="", report=True: calls.append(report)
    world.svc._reconcile_once_locked("hk")
    assert calls == [False]


@pytest.mark.parametrize("answer", [_dep(DeploymentState.TERMINATED), _dep(DeploymentState.FAILED), None])
def test_a_pod_whose_deployment_ended_or_is_gone_is_removed(answer):
    world = World(answer)
    world.svc._reconcile_once_locked("hk")
    assert world.terminated == [("d1", "lease_expired")]


def _dispatching(error):
    svc = object.__new__(NodeAgentService)
    reports = []
    lease = LeaseAssignment(deployment_id="d9", workload_id="w9", hotkey="hk", node_id="node-1", status="active")
    svc.settings = SimpleNamespace(node_id="node-1")
    svc.repository = SimpleNamespace(snapshot_runtime_items=lambda: [], get_runtime=lambda dep_id: None)
    svc.control_plane = SimpleNamespace(list_leases=lambda hk: [lease], update_deployment_status=reports.append)

    def dispatch(lease):
        raise error
    svc._reconcile_workload = dispatch
    return svc, reports


@pytest.mark.parametrize("error", [ControlPlaneUnavailable("refused"), ControlPlaneHTTPError(502, "bad gateway"),
                                   ControlPlaneHTTPError(429, "slow down")])
def test_a_control_plane_hiccup_while_starting_never_fails_the_deployment(error):
    svc, reports = _dispatching(error)
    svc._reconcile_once_locked("hk")
    assert reports == []  # tried again next tick, not FAILED (which is final)


@pytest.mark.parametrize("error", [RuntimeError("image not found"), URLError("registry refused"),
                                   TimeoutError("docker pull timed out")])
def test_a_real_error_while_starting_still_fails_it(error):
    # A timeout on the node itself (a registry, docker) is not the control
    # plane being down: retrying it forever would never end.
    svc, reports = _dispatching(error)
    svc._reconcile_once_locked("hk")
    assert [r.state for r in reports] == [DeploymentState.FAILED]


def test_the_node_client_turns_transport_failures_into_control_plane_unavailable(monkeypatch):
    from greencompute_protocol import ControlPlaneHTTPClient

    client = object.__new__(_NodeControlPlaneClient)

    def refused(self, method, path, body=None):
        raise URLError("connection refused")
    monkeypatch.setattr(ControlPlaneHTTPClient, "_request", refused)
    with pytest.raises(ControlPlaneUnavailable):
        client._request("GET", "/miner/v1/leases/hk")

    def server_error(self, method, path, body=None):
        raise ControlPlaneHTTPError(500, "boom")
    monkeypatch.setattr(ControlPlaneHTTPClient, "_request", server_error)
    with pytest.raises(ControlPlaneHTTPError):
        client._request("GET", "/miner/v1/leases/hk")


def _reporting(fail_times, error):
    svc = object.__new__(NodeAgentService)
    saved = {}
    svc.repository = SimpleNamespace(upsert_runtime=lambda rt: saved.__setitem__(rt.deployment_id, rt) or rt,
                                     snapshot_runtime_items=lambda: list(saved.items()),
                                     get_runtime=lambda dep_id: saved.get(dep_id))
    attempts = []

    def report(update):
        attempts.append(update.state)
        if len(attempts) <= fail_times:
            raise error
    svc.settings = SimpleNamespace(node_id="node-1")
    svc.control_plane = SimpleNamespace(update_deployment_status=report, list_leases=lambda hk: [
        LeaseAssignment(deployment_id="d1", workload_id="w1", hotkey="hk", node_id="node-1", status="active")])
    return svc, saved, attempts


@pytest.mark.parametrize("state", [DeploymentState.READY, DeploymentState.FAILED])
def test_a_report_that_doesnt_land_is_sent_again_until_it_does(state, monkeypatch):
    from datetime import timedelta

    monkeypatch.setattr(NodeAgentService, "_RESEND_EVERY", timedelta(0))
    svc, saved, attempts = _reporting(2, ControlPlaneUnavailable("timed out"))
    rt = _running()
    saved["d1"] = rt
    if state == DeploymentState.READY:
        svc._report_deployment_ready(rt)
    else:
        svc.gpu_allocator = SimpleNamespace(release=lambda dep: None)
        svc._fail_runtime(rt, "image not found")
    assert saved["d1"].metadata["pending_report"]["state"] == state.value
    svc._reconcile_once_locked("hk")  # fails again: still waiting
    assert saved["d1"].metadata["pending_report"]["state"] == state.value
    svc._reconcile_once_locked("hk")  # lands
    assert attempts == [state] * 3 and "pending_report" not in saved["d1"].metadata
    svc._reconcile_once_locked("hk")
    assert len(attempts) == 3  # nothing more


def test_a_report_the_control_plane_refuses_is_not_retried_forever():
    svc, saved, attempts = _reporting(99, ControlPlaneHTTPError(409, "cannot transition"))
    saved["d1"] = _running()
    svc._report_deployment_ready(saved["d1"])
    svc._reconcile_once_locked("hk")
    assert attempts == [DeploymentState.READY] and "pending_report" not in saved["d1"].metadata


# --- an undelivered report is sent again only while it still fits -------------

def _pending(state, runtime_status, lease):
    """A runtime with an undelivered `state` report; `lease` is this node's lease
    status for it, "other" (it runs on another machine now) or None (no lease)."""
    svc, saved, attempts = _reporting(0, None)
    update = __import__("greencompute_protocol").DeploymentStatusUpdate(deployment_id="d1", state=state)
    rt = _running().model_copy(update={"status": runtime_status,
                                       "metadata": {"pending_report": update.model_dump(mode="json")}})
    saved["d1"] = rt
    mine = [] if lease in (None, "other") else [
        LeaseAssignment(deployment_id="d1", workload_id="w1", hotkey="hk", node_id="node-1", status=lease)]
    fleet = {"d1"} if lease is not None else set()
    svc._resend_pending_reports(mine, fleet)
    return attempts, saved["d1"]


@pytest.mark.parametrize("state, status, lease, sent", [
    (DeploymentState.READY, "ready", "activating", True),
    (DeploymentState.READY, "ready", "other", False),          # re-placed on another machine
    (DeploymentState.READY, "terminated", "active", False),    # terminated since: stale
    (DeploymentState.FAILED, "failed", "active", True),
    (DeploymentState.FAILED, "failed", None, False),           # the control plane moved on
    (DeploymentState.SUSPENDED, "suspended", "suspended", True),
    (DeploymentState.SUSPENDED, "suspended", "active", False),  # resumed since: never undo it
    (DeploymentState.TERMINATED, "terminated", "active", True),  # the ghost-billing case: it must land
    (DeploymentState.TERMINATED, "terminated", None, True),
    (DeploymentState.TERMINATED, "terminated", "other", False),  # would kill the new placement
])
def test_an_undelivered_report_goes_again_only_while_it_still_fits(state, status, lease, sent):
    attempts, rt = _pending(state, status, lease)
    assert (attempts == [state]) is sent
    if not sent:
        assert "pending_report" not in rt.metadata


def test_resends_are_paced():
    from datetime import timedelta

    svc, saved, attempts = _reporting(99, ControlPlaneUnavailable("down"))
    lease = [LeaseAssignment(deployment_id="d1", workload_id="w1", hotkey="hk", node_id="node-1", status="active")]
    saved["d1"] = _running()
    svc._report_deployment_ready(saved["d1"])
    t0 = __import__("datetime").datetime(2026, 10, 30, 16, 0, tzinfo=__import__("datetime").UTC)
    svc._resend_pending_reports(lease, {"d1"}, now=t0)
    svc._resend_pending_reports(lease, {"d1"}, now=t0 + timedelta(seconds=10))
    assert len(attempts) == 2  # the first send, one resend; not one every tick
    svc._resend_pending_reports(lease, {"d1"}, now=t0 + timedelta(seconds=31))
    assert len(attempts) == 3
