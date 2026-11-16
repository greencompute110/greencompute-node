"""The terminal route on the node: behind the node secret (fails closed),
only for running pods, nothing but a lookup for the deployment id, and the
session's messages (ready, output, resize, ping, exit). `docker exec` is
swapped for a local shell."""
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # the test client

from fastapi.testclient import TestClient  # noqa: E402
from greencompute_node_agent.domain import pod_terminal  # noqa: E402
from greencompute_node_agent.transport import routes  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


def wait_gone(pid: int, seconds: float = 3.0) -> bool:
    """Exited (or a zombie waiting to be reaped by whoever adopted it)."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except FileNotFoundError:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


@pytest.fixture
def local(monkeypatch):
    """Terminals run an interactive sh here instead of docker exec; hang-ups are recorded."""
    hang_ups = []
    monkeypatch.setattr(pod_terminal, "exec_argv", lambda cid, marker, shell: ["sh", "-i"])
    monkeypatch.setattr(pod_terminal, "hang_up_argv", lambda cid, marker: hang_ups.append((cid, marker)) or ["true"])
    monkeypatch.setattr(pod_terminal, "KILL_GRACE_SECONDS", 0.5)
    return hang_ups


SECRET = "node-secret-for-tests"


@pytest.fixture
def node(monkeypatch, local):
    runtimes = {
        "pod-1": SimpleNamespace(workload_kind="pod", status="ready", container_id="c1", metadata={}),
        "model-1": SimpleNamespace(workload_kind="inference", status="ready", container_id="c2", metadata={}),
        "pod-starting": SimpleNamespace(workload_kind="pod", status="starting", container_id=None, metadata={}),
    }
    looked_up = []

    def get_runtime(deployment_id):
        looked_up.append(deployment_id)
        return runtimes.get(deployment_id)
    settings = SimpleNamespace(agent_auth_secret=SECRET, inference_auth_secret=None, compute_auth_secret=None)
    routes.set_service(SimpleNamespace(repository=SimpleNamespace(get_runtime=get_runtime)), settings)
    started = []
    real_start = pod_terminal.Terminal.start

    def start(self):
        started.append(self)
        real_start(self)
    monkeypatch.setattr(pod_terminal.Terminal, "start", start)
    app = fastapi.FastAPI()
    app.include_router(routes.router)
    yield TestClient(app), settings, started, looked_up
    routes.set_service(None, None)


def _refused_handshake(client, path, headers):
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(path, headers=headers):
        pass
    return exc.value.code


def _error(client, path):
    with client.websocket_connect(path, headers={"X-Agent-Auth": SECRET}) as ws:
        msg = json.loads(ws.receive_text())
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    return msg, exc.value.code


def test_the_route_refuses_calls_without_the_node_secret(node):
    client, settings, started, _ = node
    assert _refused_handshake(client, "/pods/pod-1/terminal", {}) == 4401
    assert _refused_handshake(client, "/pods/pod-1/terminal", {"X-Agent-Auth": "wrong"}) == 4401
    settings.agent_auth_secret = None  # no secret configured on the machine: off, not open
    assert _refused_handshake(client, "/pods/pod-1/terminal", {"X-Agent-Auth": ""}) == 4401
    assert started == []


@needs_sh
def test_only_running_pods_get_a_terminal(node):
    client, _, started, _ = node
    msg, code = _error(client, "/pods/model-1/terminal")
    assert (msg["code"], code) == ("conflict", 4409) and "pods" in msg["message"]
    msg, code = _error(client, "/pods/pod-starting/terminal")
    assert (msg["code"], code) == ("conflict", 4409)
    msg, code = _error(client, "/pods/nope/terminal")
    assert (msg["code"], code) == ("not_found", 4404)
    assert started == []


@pytest.mark.parametrize("path", [
    "/pods/pod-1;touch%20pwned/terminal", "/pods/$(id)/terminal", "/pods/pod-1%20--privileged/terminal",
])
def test_a_deployment_id_is_only_ever_looked_up(node, path):
    client, _, started, looked_up = node
    msg, code = _error(client, path)
    assert (msg["code"], code) == ("not_found", 4404)
    assert started == [] and looked_up and looked_up[-1] != "pod-1"


def test_a_path_in_the_deployment_id_reaches_no_route(node):
    client, _, started, looked_up = node
    with pytest.raises(WebSocketDisconnect), \
            client.websocket_connect("/pods/..%2F..%2Fagent%2Fv1/terminal", headers={"X-Agent-Auth": SECRET}):
        pass
    assert started == [] and looked_up == []


def test_an_unknown_shell_is_refused_before_anything_runs(node):
    client, _, started, _ = node
    msg, code = _error(client, "/pods/pod-1/terminal?shell=bash%3B%20touch%20/tmp/pwned")
    assert (msg["code"], code) == ("invalid_request", 4400)
    assert started == []


@needs_sh
def test_a_session_end_to_end(node):
    client, _, started, _ = node
    with client.websocket_connect("/pods/pod-1/terminal?cols=100&rows=30&shell=bash",
                                  headers={"X-Agent-Auth": SECRET}) as ws:
        assert json.loads(ws.receive_text()) == {"type": "ready"}
        assert (started[0].cols, started[0].rows) == (100, 30)
        ws.send_bytes(b"echo $((6*7))\n")
        out = b""
        while b"42" not in out:
            out += ws.receive_bytes()
        ws.send_text(json.dumps({"type": "resize", "cols": 120, "rows": 40}))
        ws.send_text(json.dumps({"type": "ping"}))
        msg = ws.receive()
        while msg.get("text") is None:
            msg = ws.receive()
        assert json.loads(msg["text"]) == {"type": "pong"}
        assert (started[0].cols, started[0].rows) == (120, 40)
        ws.send_text(json.dumps({"type": "input", "data": "exit 3\n"}))
        while True:
            msg = ws.receive()
            if msg.get("text"):
                break
        assert json.loads(msg["text"]) == {"type": "exit", "code": 3}
        assert ws.receive()["code"] == 1000
    assert pod_terminal._SLOTS._value == pod_terminal.MAX_TERMINALS


@needs_sh
def test_the_gateway_hanging_up_ends_the_shell(node, local):
    client, _, started, _ = node
    hang_ups = local
    with client.websocket_connect("/pods/pod-1/terminal", headers={"X-Agent-Auth": SECRET}) as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"
        pid = started[0].pid
    deadline = time.monotonic() + 10
    while (pod_terminal._SLOTS._value != pod_terminal.MAX_TERMINALS or not hang_ups) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pod_terminal._SLOTS._value == pod_terminal.MAX_TERMINALS
    assert wait_gone(pid) and hang_ups == [("c1", started[0].marker)]


def test_a_busy_machine_refuses_with_a_code(node):
    client, *_ = node
    taken = [pod_terminal._SLOTS.acquire(blocking=False) for _ in range(pod_terminal.MAX_TERMINALS)]
    try:
        msg, code = _error(client, "/pods/pod-1/terminal")
        assert (msg["code"], code) == ("terminal_limit", 4429)
    finally:
        for _ in taken:
            pod_terminal._SLOTS.release()
