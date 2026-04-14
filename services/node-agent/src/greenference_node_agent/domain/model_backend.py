"""Model backends — HuggingFace, diffusion, CPU fallback, bigram fallback."""

from __future__ import annotations

import hashlib
import importlib
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from greenference_protocol import ChatCompletionRequest


class ModelBackendError(RuntimeError):
    pass


class TextGenerationModelBackend:
    backend_name: str
    model_identifier: str

    def health(self) -> dict[str, Any]:
        raise NotImplementedError

    def generate_text(self, payload: ChatCompletionRequest) -> str:
        raise NotImplementedError

    def stream_tokens(self, payload: ChatCompletionRequest) -> Iterator[str]:
        for token in self.generate_text(payload).split():
            yield token


class ManifestFallbackBackend(TextGenerationModelBackend):
    def __init__(self, manifest: dict[str, Any], *, image: str, backend_name: str, fallback_reason: str) -> None:
        self.manifest = manifest
        self.backend_name = backend_name
        self.fallback_reason = fallback_reason
        self.model_identifier = str(manifest.get("model_identifier") or image)
        self.tokenizer_identifier = str(manifest.get("tokenizer_identifier") or self.model_identifier)
        self.seed_corpus = self._normalize_corpus(manifest.get("seed_corpus"))
        if not self.seed_corpus:
            raise ModelBackendError("runtime manifest missing seed_corpus")
        self._graph = self._build_bigram_graph(self.seed_corpus)
        self._keys = sorted(self._graph.keys())
        if not self._keys:
            raise ModelBackendError("runtime manifest produced an empty language graph")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.backend_name,
            "model_identifier": self.model_identifier,
            "tokenizer_identifier": self.tokenizer_identifier,
            "fallback_reason": self.fallback_reason,
        }

    def generate_text(self, payload: ChatCompletionRequest) -> str:
        prompt = self._prompt_text(payload)
        token_budget = max(8, min(int(payload.max_tokens or 24), 24))
        rng = random.Random(self._seed_for(prompt, payload))
        generated = self._generate_tokens(prompt, token_budget, rng)
        return " ".join(generated)

    def _generate_tokens(self, prompt: str, token_budget: int, rng: random.Random) -> list[str]:
        prompt_tokens = self._tokenize(prompt)
        current = prompt_tokens[-1] if prompt_tokens and prompt_tokens[-1] in self._graph else rng.choice(self._keys)
        generated: list[str] = []
        for _ in range(token_budget):
            options = self._graph.get(current) or self._graph[rng.choice(self._keys)]
            next_token = rng.choice(options)
            generated.append(next_token)
            current = next_token
        return generated

    def _seed_for(self, prompt: str, payload: ChatCompletionRequest) -> int:
        material = (
            f"{self.model_identifier}:{self.tokenizer_identifier}:{prompt}:{payload.max_tokens}:{payload.temperature}"
        ).encode()
        return int(hashlib.sha256(material).hexdigest()[:16], 16)

    def _prompt_text(self, payload: ChatCompletionRequest) -> str:
        return " ".join(message.content for message in payload.messages if message.content)

    def _normalize_corpus(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return []

    def _build_bigram_graph(self, corpus: list[str]) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = defaultdict(list)
        for segment in corpus:
            tokens = self._tokenize(segment)
            if len(tokens) < 2:
                continue
            for current, next_token in zip(tokens, tokens[1:]):
                graph[current].append(next_token)
        return dict(graph)

    def _tokenize(self, text: str) -> list[str]:
        return [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()]


class LocalCPUTextGenerationBackend(ManifestFallbackBackend):
    def __init__(self, manifest: dict[str, Any], *, image: str) -> None:
        super().__init__(
            manifest,
            image=image,
            backend_name="local-cpu-textgen",
            fallback_reason="cpu-text-runtime",
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["runtime_kind"] = "local-cpu-textgen"
        payload["fallback_reason"] = None
        return payload


class HuggingFaceCausalLMBackend(TextGenerationModelBackend):
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.backend_name = "hf-causal-lm"
        self.model_identifier = str(manifest.get("model_identifier") or "")
        self.model_revision = manifest.get("model_revision")
        self.tokenizer_identifier = str(manifest.get("tokenizer_identifier") or self.model_identifier)
        if not self.model_identifier:
            raise ModelBackendError("runtime manifest missing model_identifier")
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ModelBackendError(f"hf runtime dependency missing: {exc.name}") from exc
        self._torch = torch
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_identifier,
                revision=self.model_revision,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_identifier,
                revision=self.model_revision,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelBackendError(f"failed to load HF model runtime: {exc}") from exc
        self.model.eval()
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.backend_name,
            "model_identifier": self.model_identifier,
            "tokenizer_identifier": self.tokenizer_identifier,
        }

    def generate_text(self, payload: ChatCompletionRequest) -> str:
        prompt = self._prompt_text(payload)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        max_new_tokens = max(8, min(int(payload.max_tokens or 24), 32))
        temperature = float(payload.temperature if payload.temperature is not None else 0.7)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            )
        generated_ids = output_ids[0][input_ids.shape[-1] :]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return generated or self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _prompt_text(self, payload: ChatCompletionRequest) -> str:
        lines = []
        for message in payload.messages:
            lines.append(f"{message.role}: {message.content}")
        lines.append("assistant:")
        return "\n".join(lines)


class DiffusionModelBackend(TextGenerationModelBackend):
    """In-process diffusion backend using HuggingFace diffusers."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.backend_name = "diffusion"
        self.model_identifier = str(manifest.get("model_identifier") or "")
        self.model_revision = manifest.get("model_revision")
        if not self.model_identifier:
            raise ModelBackendError("runtime manifest missing model_identifier")
        try:
            self._torch = importlib.import_module("torch")
            self._diffusers = importlib.import_module("diffusers")
            self._base64 = importlib.import_module("base64")
            self._io = importlib.import_module("io")
        except ModuleNotFoundError as exc:
            raise ModelBackendError(f"diffusion runtime dependency missing: {exc.name}") from exc
        try:
            self.pipe = self._diffusers.AutoPipelineForText2Image.from_pretrained(
                self.model_identifier,
                revision=self.model_revision,
                torch_dtype=self._torch.float16,
                use_safetensors=True,
            ).to("cuda")
        except Exception as exc:  # noqa: BLE001
            raise ModelBackendError(f"failed to load diffusion model: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.backend_name,
            "model_identifier": self.model_identifier,
            "runtime_kind": "diffusion",
        }

    def generate_text(self, payload: ChatCompletionRequest) -> str:
        prompt = " ".join(m.content for m in payload.messages if m.content and m.role == "user")
        if not prompt:
            return "[No prompt provided]"
        with self._torch.inference_mode():
            result = self.pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5)
        image = result.images[0]
        buf = self._io.BytesIO()
        image.save(buf, format="PNG")
        b64 = self._base64.b64encode(buf.getvalue()).decode("ascii")
        return f"![Generated image](data:image/png;base64,{b64})"

    def stream_tokens(self, payload: ChatCompletionRequest) -> Iterator[str]:
        # Diffusion can't stream — yield the full result at once
        yield self.generate_text(payload)


def _fallback_backend(manifest: dict[str, Any], *, image: str, reason: str) -> TextGenerationModelBackend:
    return ManifestFallbackBackend(
        manifest,
        image=image,
        backend_name="hf-causal-lm-fallback",
        fallback_reason=reason,
    )


def create_text_generation_backend(
    manifest: dict[str, Any],
    *,
    image: str,
    allow_fallback: bool = False,
) -> TextGenerationModelBackend:
    runtime_kind = str(manifest.get("runtime_kind") or "hf-causal-lm")
    if runtime_kind == "local-cpu-textgen":
        return LocalCPUTextGenerationBackend(manifest, image=image)
    if runtime_kind == "diffusion":
        try:
            return DiffusionModelBackend(manifest)
        except ModelBackendError as exc:
            if not allow_fallback:
                raise
            return _fallback_backend(manifest, image=image, reason=str(exc))
    if runtime_kind not in ("hf-causal-lm", "vllm"):
        raise ModelBackendError(f"unsupported runtime_kind: {runtime_kind}")
    try:
        return HuggingFaceCausalLMBackend(manifest)
    except ModelBackendError as exc:
        if not allow_fallback:
            raise
        return _fallback_backend(manifest, image=image, reason=str(exc))
