"""A control-plane error is never "no leases" (2026-10-29/30): sync_leases
used to return [] on any HTTP error, and the orphan clean-up then removed
every pod and model on the box. A customer's running pod went that way when
.88 failed, and both catalog models during a deploy."""
from types import SimpleNamespace

import pytest

from greencompute_node_agent.application.services import NodeAgentService
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
