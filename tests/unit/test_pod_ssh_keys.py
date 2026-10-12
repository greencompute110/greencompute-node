"""A pod whose workload brings its own SSH keys must accept exactly those.
The node agent used to add a platform-generated key as well and report its
private half to the control plane, so a second root key to the pod existed
that the user never asked for."""
import pytest

from greencompute_node_agent.domain import ssh


def _no_keygen():
    raise AssertionError("must not generate a key when the user supplied one")


def test_own_keys_only_and_no_generated_key(monkeypatch):
    monkeypatch.setattr(ssh, "generate_ssh_keypair", _no_keygen)
    private_key, keys = ssh.pod_ssh_keys(["ssh-ed25519 AAAA user@laptop", "  ", "ssh-rsa BBBB ci\n"])
    assert private_key is None
    assert keys == ["ssh-ed25519 AAAA user@laptop", "ssh-rsa BBBB ci"]


@pytest.mark.parametrize("supplied", [None, [], ["", "   "]])
def test_generated_key_only_when_none_supplied(monkeypatch, supplied):
    monkeypatch.setattr(ssh, "generate_ssh_keypair", lambda: ("PRIVATE", "ssh-ed25519 PLATFORM"))
    assert ssh.pod_ssh_keys(supplied) == ("PRIVATE", ["ssh-ed25519 PLATFORM"])
