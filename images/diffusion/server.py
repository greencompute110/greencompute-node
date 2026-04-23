"""GreenCompute Diffusion Inference Server.

A lightweight FastAPI server that wraps HuggingFace diffusers to expose
an OpenAI-compatible /v1/chat/completions endpoint for image generation.

The server extracts the text prompt from chat messages, runs the diffusion
pipeline, and returns the generated image as a base64 data URI embedded
in markdown so it renders in the GreenCompute playground.

Usage:
    python server.py --model stabilityai/stable-diffusion-xl-base-1.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
import uuid
from typing import Any

import torch
import uvicorn
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("greencompute-diffusion")

app = FastAPI()

# Global state — set during startup
pipeline: DiffusionPipeline | None = None
model_id: str = ""
device: str = "cuda"


def extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the image generation prompt from chat messages.

    Takes the last user message content as the prompt. If the content
    includes known prefixes like "generate", "create", "draw", etc.
    they are kept as-is since diffusion models handle them fine.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content blocks — extract text parts
                parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return " ".join(parts).strip()
            return str(content).strip()
    return ""


def extract_negative_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract negative prompt if the user specifies one with 'negative:' prefix."""
    prompt = extract_prompt(messages)
    if "\nnegative:" in prompt.lower():
        parts = prompt.split("\n")
        for part in parts:
            if part.lower().startswith("negative:"):
                return part[len("negative:"):].strip()
    return ""


def generate_image(prompt: str, negative_prompt: str = "", num_steps: int = 30, guidance_scale: float = 7.5) -> str:
    """Run the diffusion pipeline and return a base64 data URI."""
    assert pipeline is not None, "Pipeline not loaded"

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    with torch.inference_mode():
        result = pipeline(**kwargs)

    image = result.images[0]

    # Encode to PNG base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/health")
async def health():
    return {"status": "ok" if pipeline is not None else "loading"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok" if pipeline is not None else "loading"}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model_name = body.get("model", model_id)

    prompt = extract_prompt(messages)
    if not prompt:
        return JSONResponse({"error": "No prompt found in messages"}, status_code=400)

    negative_prompt = extract_negative_prompt(messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Parse optional parameters from the last message or body
    num_steps = body.get("num_steps", 30)
    guidance_scale = body.get("guidance_scale", 7.5)

    logger.info("Generating image: prompt=%r steps=%d guidance=%.1f", prompt[:80], num_steps, guidance_scale)
    t0 = time.time()

    try:
        data_uri = generate_image(prompt, negative_prompt, num_steps, guidance_scale)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        content = f"[Image generation failed: {exc}]"
        if stream:
            return _stream_text(completion_id, model_name, content)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        })

    elapsed = time.time() - t0
    logger.info("Generated image in %.1fs", elapsed)

    # Return the image as markdown so the UI renders it
    content = f"![Generated image]({data_uri})\n\n*Generated in {elapsed:.1f}s*"

    if stream:
        return _stream_text(completion_id, model_name, content)

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 1, "total_tokens": len(prompt.split()) + 1},
    })


def _stream_text(completion_id: str, model: str, content: str):
    """Return a streaming response compatible with OpenAI SSE format."""

    def event_stream():
        # Send the entire content as one chunk (images can't be meaningfully streamed)
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Send finish
        done_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def load_pipeline(model: str, dtype: str = "float16") -> DiffusionPipeline:
    """Load the diffusion pipeline with optimal settings."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    logger.info("Loading diffusion model: %s (dtype=%s)", model, dtype)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "float16" else None,
            use_safetensors=True,
        )
    except Exception:
        logger.info("AutoPipeline failed, falling back to DiffusionPipeline.from_pretrained")
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    pipe = pipe.to(device)

    # Enable memory optimizations
    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    logger.info("Model loaded successfully on %s", device)
    return pipe


def main():
    global pipeline, model_id, device

    parser = argparse.ArgumentParser(description="GreenCompute Diffusion Server")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    model_id = args.model
    device = args.device
    pipeline = load_pipeline(args.model, args.dtype)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
