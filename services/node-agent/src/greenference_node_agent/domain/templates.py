"""Lium-style pod template catalog."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemplateSpec:
    image: str
    port: int
    gpu_fraction: float = 1.0
    env_vars: dict[str, str] = field(default_factory=dict)
    description: str = ""


BUILTIN_TEMPLATES: dict[str, TemplateSpec] = {
    "jupyter": TemplateSpec(
        image="jupyter/scipy-notebook:latest",
        port=8888,
        gpu_fraction=1.0,
        description="Jupyter Notebook with scipy stack",
    ),
    "vscode": TemplateSpec(
        image="codercom/code-server:latest",
        port=8080,
        gpu_fraction=0.5,
        description="VS Code Server in the browser",
    ),
    "pytorch": TemplateSpec(
        image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        port=8888,
        gpu_fraction=1.0,
        description="PyTorch with CUDA 12.1",
    ),
    "vllm": TemplateSpec(
        image="vllm/vllm-openai:v0.7.3",
        port=8000,
        gpu_fraction=1.0,
        description="vLLM OpenAI-compatible inference server",
    ),
    "comfyui": TemplateSpec(
        image="ghcr.io/ai-dock/comfyui:latest",
        port=8188,
        gpu_fraction=1.0,
        description="ComfyUI diffusion UI",
    ),
    "ubuntu-ssh": TemplateSpec(
        image="greenference/gpu-pod:latest",
        port=22,
        gpu_fraction=0.0,
        description="Ubuntu 22.04 with SSH access (no GPU)",
    ),
    "gpu-pod": TemplateSpec(
        image="greenference/gpu-pod:latest",
        port=22,
        gpu_fraction=1.0,
        description="GPU pod with SSH access (CUDA 12.4)",
    ),
    "pytorch-jupyter": TemplateSpec(
        image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        port=8888,
        gpu_fraction=1.0,
        env_vars={"JUPYTER_ENABLE_LAB": "yes"},
        description="PyTorch + JupyterLab",
    ),
}


def get_template(name: str) -> TemplateSpec | None:
    return BUILTIN_TEMPLATES.get(name)


def list_templates() -> dict[str, dict]:
    return {
        name: {
            "image": spec.image,
            "port": spec.port,
            "gpu_fraction": spec.gpu_fraction,
            "description": spec.description,
        }
        for name, spec in BUILTIN_TEMPLATES.items()
    }
