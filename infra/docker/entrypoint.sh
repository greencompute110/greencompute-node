#!/bin/bash
# Green Compute pod entrypoint — injected into any container image.
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
        echo "[greencompute] WARN: cannot install openssh-server — unknown package manager" >&2
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

    # Configure sshd: root login, KEY-ONLY auth (no passwords), port 22.
    # Password auth is disabled on purpose: no root password is ever set, so it
    # could never succeed anyway, and leaving it on turns a failed key auth into
    # a misleading password PROMPT instead of a clear "Permission denied
    # (publickey)" — which sends users debugging the wrong thing.
    local cfg=/etc/ssh/sshd_config
    if [ -f "$cfg" ]; then
        sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' "$cfg"
        sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' "$cfg"
        sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' "$cfg"
        sed -i 's/^#\?Port .*/Port 22/' "$cfg"
        # Ensure these are set even if not present
        grep -q "^PermitRootLogin" "$cfg" || echo "PermitRootLogin yes" >> "$cfg"
        grep -q "^PubkeyAuthentication" "$cfg" || echo "PubkeyAuthentication yes" >> "$cfg"
        grep -q "^PasswordAuthentication" "$cfg" || echo "PasswordAuthentication no" >> "$cfg"
        sed -i 's/^#\?PermitUserEnvironment.*/PermitUserEnvironment yes/' "$cfg"
        grep -q "^PermitUserEnvironment" "$cfg" || echo "PermitUserEnvironment yes" >> "$cfg"
    fi
    # Many base images (Ubuntu/Debian) Include /etc/ssh/sshd_config.d/*.conf
    # FIRST, so a shipped drop-in (e.g. 50-cloud-init.conf: "PasswordAuthentication
    # yes") would override the main config above. A 00- drop-in sorts first and
    # wins, so key-only auth holds regardless of the user's chosen image.
    if [ -d /etc/ssh/sshd_config.d ]; then
        cat > /etc/ssh/sshd_config.d/00-greencompute.conf <<'EOF'
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
PermitUserEnvironment yes
EOF
    fi
}

# --- Carry the image's environment into SSH sessions ---
# sshd builds a fresh environment for every session, so ENV set by the image
# (PATH=/opt/conda/bin:..., CUDA_HOME, LD_LIBRARY_PATH, CONDA_*) and by
# `docker run -e` never reached a user who SSHed in. A PyTorch/conda image then
# looked like plain Ubuntu -- `python` missing, conda absent -- and users (and
# agents) concluded the image was broken. Snapshot this process's start-up
# environment, which is exactly image ENV + docker -e, into:
#   * /etc/profile.d -- login shells (`ssh host`), which source /etc/profile;
#     it runs after Debian/Ubuntu's /etc/profile resets PATH, so it wins;
#   * ~/.ssh/environment (+ PermitUserEnvironment) -- one-off commands
#     (`ssh host python train.py`) and scp, which never read /etc/profile;
#   * /etc/environment -- because with UsePAM (Ubuntu/Debian default) OpenSSH
#     applies pam_env AFTER ~/.ssh/environment, and stock images ship a generic
#     PATH there that silently overwrote ours. Verified on the real
#     pytorch/pytorch:2.7.0-cuda12.8 image: without this, `ssh host python`
#     still failed even with ~/.ssh/environment correct.
# Paths are overridable only so the test suite can run this unprivileged.
persist_image_env() {
    local profile="${GC_PROFILE_D_FILE:-/etc/profile.d/00-greencompute-image-env.sh}"
    local sshenv="${GC_SSH_ENV_FILE:-/root/.ssh/environment}"
    local etcenv="${GC_ETC_ENV_FILE:-/etc/environment}"
    local etcnew="${etcenv}.greencompute.$$"
    local names=" "
    local skip='^(HOSTNAME|HOME|PWD|OLDPWD|SHLVL|TERM|_|SHELL|USER|LOGNAME|MAIL|AUTHORIZED_KEYS)$'
    mkdir -p "$(dirname "$profile")" "$(dirname "$sshenv")"
    : > "$profile"
    : > "$sshenv"
    : > "$etcnew"
    local kv name value
    while IFS= read -r -d '' kv; do
        name=${kv%%=*}
        value=${kv#*=}
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ "$name" =~ $skip ]] && continue
        # ~/.ssh/environment is one NAME=value per line with no escaping, so a
        # multi-line value would corrupt every entry after it. Skip those.
        [[ "$value" == *$'\n'* ]] && continue
        printf 'export %s=%q\n' "$name" "$value" >> "$profile"
        printf '%s=%s\n' "$name" "$value" >> "$sshenv"
        # pam_env strips surrounding quotes and does no escaping, so a value
        # containing a double quote can't be represented -- leave it out.
        if [[ "$value" != *'"'* ]]; then
            printf '%s="%s"\n' "$name" "$value" >> "$etcnew"
            names+="$name "
        fi
    done < "${GC_ENVIRON_SOURCE:-/proc/$$/environ}"
    # Keep any existing /etc/environment entries we are not overriding.
    if [ -f "$etcenv" ]; then
        local line key
        while IFS= read -r line || [ -n "$line" ]; do
            key=${line%%=*}
            key=${key#export }
            [[ "$names" == *" $key "* ]] && continue
            printf '%s\n' "$line" >> "$etcnew"
        done < "$etcenv"
    fi
    mv -f "$etcnew" "$etcenv"
    chmod 644 "$etcenv"
    chmod 644 "$profile"
    chmod 600 "$sshenv"
}

echo "[greencompute] setting up SSH..."
if install_ssh; then
    configure_ssh
    persist_image_env || echo "[greencompute] WARN: could not persist image env for SSH sessions" >&2
    /usr/sbin/sshd 2>/dev/null || echo "[greencompute] WARN: sshd failed to start" >&2
    echo "[greencompute] SSH ready on port 22"
else
    echo "[greencompute] SSH setup failed — container will run without SSH" >&2
fi

# --- Run original command or sleep forever ---
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "[greencompute] no CMD specified — keeping container alive"
    exec sleep infinity
fi
