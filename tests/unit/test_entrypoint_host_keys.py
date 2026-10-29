"""Each new pod makes SSH host keys of its own. An image with openssh-server
preinstalled carries keys made when the image was built, the same in every
pod of that image, so another renter of the image held this pod's private
host key; the published fingerprint would then prove nothing."""
import os
import pathlib
import shutil
import subprocess

import pytest

ENTRYPOINT = pathlib.Path(__file__).resolve().parents[2] / "infra" / "docker" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("ssh-keygen") is None, reason="needs bash and ssh-keygen")


def _function_source() -> str:
    text = ENTRYPOINT.read_text()
    start = text.index("fresh_host_keys() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _run(root: pathlib.Path, container_id: str, path: str | None = None, **env) -> int:
    return subprocess.run(
        ["bash", "-c", _function_source() + f'\nfresh_host_keys "{root}" "{container_id}"\n'],
        env={"PATH": path or os.environ["PATH"], **env}, timeout=60,
    ).returncode


def _key(root: pathlib.Path) -> str:
    return (root / "etc/ssh/ssh_host_ed25519_key.pub").read_text()


def _baked_image(root: pathlib.Path) -> str:
    """/etc/ssh as an image with openssh-server preinstalled ships it."""
    (root / "etc/ssh").mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-keygen", "-A", "-f", str(root)], check=True, capture_output=True)
    return _key(root)


def test_a_new_pod_replaces_the_images_keys_once(tmp_path):
    baked = _baked_image(tmp_path)
    assert _run(tmp_path, "c0ffee01", GC_NEW_HOST_KEYS="1") == 0
    first = _key(tmp_path)
    assert first != baked
    assert (tmp_path / "etc/ssh/ssh_host_ed25519_key").exists()
    assert (tmp_path / "etc/ssh/.greencompute-host-keys").read_text().strip() == "c0ffee01"
    # A restart or resume of the same container keeps them.
    assert _run(tmp_path, "c0ffee01", GC_NEW_HOST_KEYS="1") == 0
    assert _key(tmp_path) == first


def test_a_pod_saved_as_an_image_and_started_again_gets_new_keys(tmp_path):
    _baked_image(tmp_path)
    assert _run(tmp_path, "c0ffee01", GC_NEW_HOST_KEYS="1") == 0
    first = _key(tmp_path)
    # Same files (marker included), new container.
    assert _run(tmp_path, "beef0002", GC_NEW_HOST_KEYS="1") == 0
    assert _key(tmp_path) != first


def test_a_pod_started_before_keeps_its_keys(tmp_path):
    baked = _baked_image(tmp_path)
    assert _run(tmp_path, "c0ffee01") == 0  # no GC_NEW_HOST_KEYS: a container from before
    assert _key(tmp_path) == baked


def test_an_image_without_keys_gets_them(tmp_path):
    assert _run(tmp_path, "c0ffee01", GC_NEW_HOST_KEYS="1") == 0
    assert _key(tmp_path)


def test_if_new_keys_cant_be_made_the_old_ones_stay(tmp_path):
    """ssh-keygen -A exits 0 even when it saves nothing; SSH must still work."""
    baked = _baked_image(tmp_path)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ssh-keygen").write_text("#!/bin/sh\nexit 0\n")
    (fake / "ssh-keygen").chmod(0o755)
    assert _run(tmp_path, "c0ffee01", path=f"{fake}:{os.environ['PATH']}", GC_NEW_HOST_KEYS="1") == 1
    assert _key(tmp_path) == baked
    assert (tmp_path / "etc/ssh/ssh_host_ed25519_key").exists()
    assert not (tmp_path / "etc/ssh/.greencompute-host-keys").exists()  # tried again next start
