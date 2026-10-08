"""The pod entrypoint must hand the image's environment to SSH sessions.

Reported by an external tester: on the PyTorch/conda image, `/opt/conda` existed
but was not on PATH after SSHing in, so the pod looked like bare Ubuntu and they
assumed the image was broken. sshd builds each session's environment from
scratch; `persist_image_env` snapshots the container's start-up environment into
the two files sshd sessions do read.

These run the real function out of the real script, with its output paths and
input environ redirected into a temp dir so no root is needed.
"""
import os
import pathlib
import shutil
import subprocess

import pytest

ENTRYPOINT = pathlib.Path(__file__).resolve().parents[2] / "infra" / "docker" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _function_source() -> str:
    text = ENTRYPOINT.read_text()
    start = text.index("persist_image_env() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _run(tmp_path, environ: dict[str, str], etc_environment: str | None = None):
    src = tmp_path / "environ"
    etcenv = tmp_path / "etc" / "environment"
    etcenv.parent.mkdir(parents=True, exist_ok=True)
    if etc_environment is not None:
        etcenv.write_text(etc_environment)
    src.write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in environ.items()))
    profile = tmp_path / "profile.d" / "00-image-env.sh"
    sshenv = tmp_path / "ssh" / "environment"
    script = _function_source() + "\npersist_image_env\n"
    subprocess.run(
        ["bash", "-c", script],
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "GC_ENVIRON_SOURCE": str(src),
            "GC_PROFILE_D_FILE": str(profile),
            "GC_SSH_ENV_FILE": str(sshenv),
            "GC_ETC_ENV_FILE": str(etcenv),
        },
    )
    _run.etcenv = etcenv
    return profile, sshenv


def test_conda_path_reaches_login_shells_and_one_off_commands(tmp_path):
    image_path = "/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    profile, sshenv = _run(tmp_path, {
        "PATH": image_path,
        "CONDA_PREFIX": "/opt/conda",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
    })
    # Login shell: sourcing the profile.d file restores the image PATH even
    # after a distro /etc/profile reset it.
    out = subprocess.run(
        ["bash", "-c", f'PATH=/usr/bin:/bin; . "{profile}"; echo "$PATH|$CONDA_PREFIX"'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == f"{image_path}|/opt/conda"
    # One-off `ssh host cmd`: sshd reads ~/.ssh/environment literally.
    lines = sshenv.read_text().splitlines()
    assert f"PATH={image_path}" in lines
    assert "LD_LIBRARY_PATH=/usr/local/nvidia/lib64" in lines


def test_session_specific_and_bootstrap_variables_are_not_frozen(tmp_path):
    profile, sshenv = _run(tmp_path, {
        "PATH": "/usr/bin",
        "HOSTNAME": "abc123",
        "HOME": "/root",
        "PWD": "/",
        "AUTHORIZED_KEYS": "ssh-ed25519 AAAA test",
    })
    body = profile.read_text() + sshenv.read_text()
    for name in ("HOSTNAME", "HOME", "PWD", "AUTHORIZED_KEYS"):
        assert f"{name}=" not in body


def test_values_with_spaces_and_quotes_survive_shell_quoting(tmp_path):
    tricky = "a b \"c\" 'd' $e `f`"
    profile, _ = _run(tmp_path, {"PATH": "/usr/bin", "TRICKY": tricky})
    out = subprocess.run(
        ["bash", "-c", f'. "{profile}"; printf %s "$TRICKY"'],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out == tricky


def test_multiline_values_are_skipped_rather_than_corrupting_the_file(tmp_path):
    _, sshenv = _run(tmp_path, {"PATH": "/usr/bin", "CERT": "line1\nline2", "AFTER": "ok"})
    lines = sshenv.read_text().splitlines()
    assert "AFTER=ok" in lines
    assert not any(line.startswith("CERT=") or line == "line2" for line in lines)


def test_files_get_safe_permissions(tmp_path):
    profile, sshenv = _run(tmp_path, {"PATH": "/usr/bin"})
    assert oct(profile.stat().st_mode & 0o777) == "0o644"
    assert oct(sshenv.stat().st_mode & 0o777) == "0o600"


def test_entrypoint_enables_user_environment_and_calls_the_function():
    text = ENTRYPOINT.read_text()
    assert "PermitUserEnvironment yes" in text
    # Called after configure_ssh and before sshd starts, or it has no effect.
    assert text.index("persist_image_env ||") < text.index("/usr/sbin/sshd 2>/dev/null")


def test_etc_environment_overrides_the_stock_path_that_pam_applies_last(tmp_path):
    # The bug that survived the first fix: with UsePAM, OpenSSH applies pam_env
    # AFTER ~/.ssh/environment, and Ubuntu images ship a generic PATH in
    # /etc/environment. Reproduced on pytorch/pytorch:2.7.0-cuda12.8.
    stock = 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games"\nLANG=C.UTF-8\n'
    image_path = "/opt/conda/bin:/usr/bin:/bin"
    _run(tmp_path, {"PATH": image_path}, etc_environment=stock)
    lines = _run.etcenv.read_text().splitlines()
    assert [l for l in lines if l.startswith("PATH=")] == [f'PATH="{image_path}"']
    # Unrelated stock entries are preserved.
    assert "LANG=C.UTF-8" in lines


def test_etc_environment_is_created_when_absent(tmp_path):
    _run(tmp_path, {"PATH": "/opt/conda/bin:/usr/bin"})
    assert 'PATH="/opt/conda/bin:/usr/bin"' in _run.etcenv.read_text().splitlines()
    assert oct(_run.etcenv.stat().st_mode & 0o777) == "0o644"


def test_values_pam_env_cannot_represent_are_left_out_of_etc_environment(tmp_path):
    _run(tmp_path, {"PATH": "/usr/bin", "QUOTED": 'say "hi"'})
    assert "QUOTED" not in _run.etcenv.read_text()
