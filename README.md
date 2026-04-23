# greencompute-node

Unified miner daemon for the Green Compute Bittensor subnet — **netuid 110 on mainnet**, **netuid 16 on testnet**. One process handles inference (vLLM + diffusion), GPU rental pods, and private VMs. No separate miner repos to run.

## what you earn for

Your node-agent receives two kinds of inference work automatically:

1. **Subnet catalog models** — once you're registered + whitelisted, the validator's Flux orchestrator assigns catalog models (currently `qwen2.5-7b-instruct`, `qwen2-vl-7b-instruct`, etc.) to your GPUs based on demand. You pull the vLLM image, serve the model, and get scored on probe results (latency, nonce-echo correctness, throughput).
2. **Private endpoints** — users can deploy their own HuggingFace model onto your GPUs for hourly-billed dedicated capacity.

Plus:

3. **GPU pods** — SSH-accessible Docker containers users rent by the hour.
4. **VMs** — Firecracker microVMs (roadmap).

## quick start

### prerequisites

- Linux with NVIDIA GPU (RTX 4090, **5090**, A100, H100, L40S, etc.)
- **NVIDIA driver ≥ 545** — required for CUDA 13 runtime (the vLLM 0.19.1 image used for catalog + private inference). RTX 5090 (Blackwell/sm_120) requires this or newer.
- Docker Engine 20.10+
- NVIDIA Container Toolkit

```bash
# install nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify driver version (must be ≥545 for CUDA 13)
nvidia-smi | grep "Driver Version"
```

### clone

```bash
mkdir greencompute-ai && cd greencompute-ai
git clone https://github.com/greenference/greencompute.git        # protocol
git clone https://github.com/greenference/greencompute-node.git   # this repo
cd greencompute-node
```

### configure

```bash
cp .env.example .env
```

Edit `.env` — fill in validator IP, hotkey, hardware specs, HF token. See `.env.example` for the full list with comments.

### run

```bash
docker compose up -d
```

### verify

```bash
curl http://localhost:8007/readyz
```

### firewall

```
8007/tcp         - node-agent API (control-plane connects here)
30000-31000/tcp  - SSH port range for pod tenants
```

## how it works

1. Agent registers with the control plane (GPU model, count, VRAM, supported workload kinds).
2. Sends heartbeats + capacity updates every ~1s.
3. Polls `GET /miner/v1/leases/{hotkey}` for assigned work.
4. On a new lease, dispatches by workload kind:
   - **Inference (catalog or private)** — pulls vLLM image, spawns container with `--model <hf_repo>` + model-specific flags (vision caps, `max_model_len`, `--limit-mm-per-prompt '{"image":4}'` for VL models), proxies requests from gateway.
   - **Pod** — Docker container with SSH port.
   - **VM** — Firecracker microVM (stub by default).
5. Reports state transitions back to control plane (`pulling` → `starting` → `ready`).
6. Validator fires periodic probe canaries; your responses must include the random nonce the validator generates to pass fraud detection.

## inference runtimes

| Template | Docker image (default) | Example models |
|---|---|---|
| `vllm` | `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` | Qwen 2.5, Llama 3, Mistral, Phi |
| `vllm-vision` | `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` | Qwen2-VL, LLaVA, Phi-3.5-vision |
| `diffusion` | `greenference/diffusion-server:latest` | SDXL, FLUX, SD3 |

All inference images are CUDA 13 → **NVIDIA driver 545+ required**.

Images pull automatically on first use. Model weights cache in `~/.cache/huggingface` (shared between catalog + private deploys of the same repo).

## updating

```bash
git pull
docker compose restart node-agent
```

Node-agent code is volume-mounted — the restart picks up the new Python without pulling new images. Catalog containers keep running through the restart (they're `docker run`'d outside compose) and the node-agent reattaches to them on startup.

## environment variables

See `.env.example` for the full list. Key ones:

| Variable | What to set |
|---|---|
| `GREENCOMPUTE_CONTROL_PLANE_URL` | Validator IP + port 28001 |
| `GREENCOMPUTE_MINER_VALIDATOR_URL` | Validator IP + port 28002 |
| `GREENCOMPUTE_MINER_HOTKEY` | Your Bittensor hotkey (SS58) |
| `GREENCOMPUTE_COLDKEY_NAME` + `_HOTKEY_NAME` | Names in `~/.bittensor/wallets/` for ed25519 signing |
| `GREENCOMPUTE_GPU_MODEL` / `GPU_COUNT` / `VRAM_GB_PER_GPU` | Your actual hardware |
| `GREENCOMPUTE_INFERENCE_BACKEND` | `docker` (production) |
| `HF_TOKEN` | HuggingFace token for gated models (Llama, etc.) |
| `GREENCOMPUTE_SSH_HOST` | Your public IP (for pod SSH) |
| `GREENCOMPUTE_VLLM_IMAGE` | Override the default vLLM image tag if needed |

## HuggingFace token

Required for gated models. Get a read token at https://huggingface.co/settings/tokens, accept the model license on HuggingFace, set `HF_TOKEN=hf_xxx` in your `.env`. The token is passed into vLLM containers automatically.

## directory structure

```
greencompute-node/
├── services/node-agent/          # agent source (Python FastAPI + worker loop)
│   └── src/greencompute_node_agent/
│       ├── application/services.py  # reconcile loop
│       ├── domain/
│       │   ├── inference.py         # vLLM / diffusion container orchestration
│       │   ├── pod.py               # pod backend
│       │   ├── vm.py                # firecracker stub
│       │   └── gpu_allocator.py     # device assignment
│       └── transport/routes.py      # /inference/<id>/v1/chat/completions proxy + /pods/<id>/stats
├── images/diffusion/             # diffusion server image Dockerfile
└── docker-compose.yml            # production compose for the node-agent
```

## troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| vLLM exits with `Error 804: forward compatibility was attempted on non supported HW` | Your NVIDIA driver is too old for CUDA 13 | `apt install nvidia-driver-550 && reboot` |
| `limit-mm-per-prompt: Value image=4 cannot be converted to loads` | Old vLLM arg syntax | Update node-agent code (`git pull` this repo) — new code passes `'{"image": 4}'` JSON |
| Deployment stuck `scheduled` forever | Node-agent can't reach control-plane | Check `GREENCOMPUTE_CONTROL_PLANE_URL` + firewall |
| Miner keeps respawning same failed catalog model | Validator applied a 15-min cooldown | Wait; if permanent, fix host (driver/VRAM/etc) then validator auto-retries |
| Probe fraud penalty keeps triggering | Your miner returned a cached/proxied response missing the validator's random nonce | Ensure your vLLM container is actually running the declared model + isn't being proxied |

See full subnet docs in [../README.md](../README.md).
