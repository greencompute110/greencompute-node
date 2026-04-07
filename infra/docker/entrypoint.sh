#!/bin/bash
# Greenference pod entrypoint — injected into any container image.
# Installs SSH server, writes authorized keys, starts sshd, then
# runs the original CMD so the container works as expected.
# Modeled after Vast.ai / RunPod pod bootstrap.
set -e

# --- SSH setup ---
install_ssh() {
    if command -v sshd >/dev/null 2>&1; then
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache openssh-server >/dev/null 2>&1
    elif command -v yum >/dev/null 2>&1; then
        yum install -y openssh-server >/dev/null 2>&1
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y openssh-server >/dev/null 2>&1
    else
        echo "[greenference] WARN: cannot install openssh-server — unknown package manager" >&2
        return 1
    fi
}

configure_ssh() {
    mkdir -p /run/sshd /root/.ssh
    chmod 700 /root/.ssh

    # Write authorized keys
    if [ -n "$AUTHORIZED_KEYS" ]; then
        echo "$AUTHORIZED_KEYS" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi

    # Generate host keys if missing
    ssh-keygen -A 2>/dev/null || true

    # Configure sshd: root login, key auth only, port 22
    local cfg=/etc/ssh/sshd_config
    if [ -f "$cfg" ]; then
        sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' "$cfg"
        sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' "$cfg"
        sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' "$cfg"
        sed -i 's/^#\?Port .*/Port 22/' "$cfg"
        # Ensure these are set even if not present
        grep -q "^PermitRootLogin" "$cfg" || echo "PermitRootLogin yes" >> "$cfg"
        grep -q "^PubkeyAuthentication" "$cfg" || echo "PubkeyAuthentication yes" >> "$cfg"
    fi
}

echo "[greenference] setting up SSH..."
if install_ssh; then
    configure_ssh
    /usr/sbin/sshd 2>/dev/null || echo "[greenference] WARN: sshd failed to start" >&2
    echo "[greenference] SSH ready on port 22"
else
    echo "[greenference] SSH setup failed — container will run without SSH" >&2
fi

# --- Run original command or sleep forever ---
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "[greenference] no CMD specified — keeping container alive"
    exec sleep infinity
fi
