"""Real model backends. Missing libraries become unsupported, never simulated scores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from research.env import select_device
from research.memory import reset_gpu_peak, snapshot


DEFAULT_PROMPT = "The future of efficient language model inference is"


@dataclass
class LoadedModel:
    backend: str
    method: str
    model_id: str
    device: str
    precision: str
    load_time_s: float
    extras: Dict[str, Any] = field(default_factory=dict)
    _impl: Any = None
    _tokenizer: Any = None

    def close(self) -> None:
        impl = self._impl
        self._impl = None
        self._tokenizer = None
        try:
            import gc
            import torch

            if impl is not None:
                closer = getattr(impl, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
                try:
                    impl.to("cpu")
                except Exception:
                    pass
                del impl
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass


@dataclass
class GenerationSample:
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    inter_token_ms: List[float]
    tokens_per_sec: Optional[float]
    text: str
    memory: Dict[str, Any]


def _dtype_for(precision: str, device: str):
    import torch

    precision = precision.lower()
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _to_device_inputs(encoded, device):
    return {k: v.to(device) for k, v in encoded.items()}


def estimate_weight_mb(model: Any) -> Optional[float]:
    try:
        total = 0
        for param in model.parameters():
            total += param.numel() * param.element_size()
        for buf in model.buffers():
            total += buf.numel() * buf.element_size()
        return round(total / (1024**2), 3)
    except Exception:
        return None


def load_transformers(
    model_id: str,
    *,
    method: str = "fp32",
    device: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> LoadedModel:
    """Load a Hugging Face causal LM. `model`/`tokenizer` inject in-memory test models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or select_device()
    method = method.lower()
    t0 = time.perf_counter()
    extras: Dict[str, Any] = {}

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    if model is not None:
        model.eval()
        if device != "cpu":
            model.to(device)
        load_time = time.perf_counter() - t0
        precision = str(next(model.parameters()).dtype)
        return LoadedModel(
            backend="transformers",
            method=method,
            model_id=model_id,
            device=device,
            precision=precision,
            load_time_s=load_time,
            extras={"injected": True},
            _impl=model,
            _tokenizer=tokenizer,
        )

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    precision = method

    if method in {"fp16", "bf16", "fp32"}:
        dtype = _dtype_for(method, device)
        load_kwargs["torch_dtype"] = dtype
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if "device_map" not in load_kwargs:
            model.to(device)
        precision = method
    elif method == "dynamic_int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        device = "cpu"
        precision = "dynamic_int8"
        extras["note"] = "PyTorch dynamic quantization targets Linear layers on CPU"
    elif method in {"int8_bnb", "bitsandbytes_int8"}:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        device = "cuda"
        precision = "int8_bnb"
    elif method in {"int4_bnb", "bitsandbytes_int4", "nf4"}:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        device = "cuda"
        precision = "int4_nf4"
    elif method == "gptq":
        gptq_id = kwargs.get("quantized_model_id")
        if not gptq_id:
            raise ValueError(
                "GPTQ is not timed unless gptq_model_id points at a GPTQ checkpoint; "
                "refusing to load dense Hugging Face weights and label them GPTQ"
            )
        model = AutoModelForCausalLM.from_pretrained(
            gptq_id,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        extras["loaded_from"] = gptq_id
        device = "cuda"
        precision = "gptq"
        model_id = gptq_id
    elif method == "awq":
        awq_id = kwargs.get("quantized_model_id") or model_id
        try:
            from awq import AutoAWQForCausalLM

            model = AutoAWQForCausalLM.from_quantized(
                awq_id, fuse_layers=False, trust_remote_code=True
            )
            extras["loader"] = "autoawq"
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                awq_id,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            extras["loader"] = "transformers"
        extras["loaded_from"] = awq_id
        device = "cuda"
        precision = "awq"
        model_id = awq_id
    else:
        raise ValueError(f"unknown transformers method: {method}")

    model.eval()
    load_time = time.perf_counter() - t0
    extras["weight_mb"] = estimate_weight_mb(model)
    return LoadedModel(
        backend="transformers",
        method=method,
        model_id=model_id,
        device=device,
        precision=precision,
        load_time_s=load_time,
        extras=extras,
        _impl=model,
        _tokenizer=tokenizer,
    )


def _next_token_from_logits(logits, temperature: float = 0.0):
    import torch

    last = logits[:, -1, :]
    if temperature and temperature > 0:
        probs = torch.softmax(last / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(last, dim=-1, keepdim=True)


def generate_transformers(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    use_cache: bool = True,
    temperature: float = 0.0,
) -> GenerationSample:
    """Prefill + decode loop so TTFT and inter-token latency are real wall-clock times."""
    import torch

    model = loaded._impl
    tokenizer = loaded._tokenizer
    device = loaded.device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = _to_device_inputs(encoded, device)
    input_ids = encoded["input_ids"]
    prompt_tokens = int(input_ids.shape[1])
    eos_id = getattr(tokenizer, "eos_token_id", None)

    reset_gpu_peak()
    mem_before = snapshot()
    generated: List[int] = []
    inter_token: List[float] = []
    past = None
    ttft_ms = 0.0

    with torch.no_grad():
        t0 = time.perf_counter()
        if hasattr(model, "generate") and device == "meta":
            raise RuntimeError("model is on meta device")
        out = model(input_ids=input_ids, use_cache=use_cache)
        first = _next_token_from_logits(out.logits, temperature)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        token_id = int(first.item())
        generated.append(token_id)
        past = out.past_key_values if use_cache else None
        cur = first

        for _ in range(max_new_tokens - 1):
            if eos_id is not None and token_id == eos_id:
                break
            t1 = time.perf_counter()
            if use_cache:
                out = model(input_ids=cur, past_key_values=past, use_cache=True)
            else:
                all_ids = torch.cat(
                    [input_ids, torch.tensor([generated], device=input_ids.device)],
                    dim=1,
                )
                out = model(input_ids=all_ids, use_cache=False)
            nxt = _next_token_from_logits(out.logits, temperature)
            inter_token.append((time.perf_counter() - t1) * 1000.0)
            token_id = int(nxt.item())
            generated.append(token_id)
            past = out.past_key_values if use_cache else None
            cur = nxt

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    mem_after = snapshot()
    n_new = len(generated)
    tps = (n_new / (e2e_ms / 1000.0)) if e2e_ms > 0 else None
    try:
        text = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception:
        text = ""

    return GenerationSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=n_new,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        inter_token_ms=inter_token,
        tokens_per_sec=tps,
        text=text,
        memory={"before": mem_before, "after": mem_after, "peak": snapshot()},
    )


def load_gguf(
    model_id: str,
    *,
    gguf_file: Optional[str] = None,
    n_ctx: int = 512,
    n_gpu_layers: Optional[int] = None,
    **kwargs: Any,
) -> LoadedModel:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    t0 = time.perf_counter()
    device = select_device()
    gpu_compiled = None
    try:
        from llama_cpp import llama_supports_gpu_offload

        gpu_compiled = bool(llama_supports_gpu_offload())
    except Exception:
        gpu_compiled = None
    if n_gpu_layers is None:
        if device == "cuda" and gpu_compiled is not False:
            n_gpu_layers = -1
        else:
            n_gpu_layers = 0

    path = kwargs.get("model_path")
    if path is None:
        filename = gguf_file or kwargs.get("filename")
        if not filename:
            raise ValueError("GGUF load requires gguf_file or model_path")
        path = hf_hub_download(repo_id=model_id, filename=filename)

    llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        logits_all=False,
    )
    load_time = time.perf_counter() - t0
    return LoadedModel(
        backend="llama.cpp",
        method="gguf",
        model_id=model_id,
        device=device if n_gpu_layers != 0 else "cpu",
        precision=kwargs.get("precision", "gguf_q4_k_m"),
        load_time_s=load_time,
        extras={
            "path": path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "gpu_offload_compiled": gpu_compiled,
            "llama_cpp_version": getattr(__import__("llama_cpp"), "__version__", None),
        },
        _impl=llm,
    )


def generate_gguf(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> GenerationSample:
    llm = loaded._impl
    reset_gpu_peak()
    mem_before = snapshot()
    tokens_out: List[str] = []
    inter_token: List[float] = []
    ttft_ms = 0.0
    t0 = time.perf_counter()
    first = True
    last_t = t0
    usage_completion = None
    usage_prompt = None

    for chunk in llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.0),
        stream=True,
        echo=False,
    ):
        now = time.perf_counter()
        piece = chunk["choices"][0].get("text") or ""
        if first:
            ttft_ms = (now - t0) * 1000.0
            first = False
        else:
            inter_token.append((now - last_t) * 1000.0)
        last_t = now
        tokens_out.append(piece)
        usage = chunk.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            usage_completion = int(usage["completion_tokens"])
        if usage.get("prompt_tokens") is not None:
            usage_prompt = int(usage["prompt_tokens"])

    text = "".join(tokens_out)
    prompt_tokens = usage_prompt
    if prompt_tokens is None:
        try:
            prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
        except Exception:
            prompt_tokens = 0
    completion_tokens = usage_completion
    if completion_tokens is None:
        try:
            completion_tokens = len(llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            completion_tokens = len([p for p in tokens_out if p])

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    tps = (completion_tokens / (e2e_ms / 1000.0)) if e2e_ms > 0 and completion_tokens else None
    return GenerationSample(
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        inter_token_ms=inter_token,
        tokens_per_sec=tps,
        text="".join(tokens_out),
        memory={"before": mem_before, "after": snapshot()},
    )


def try_load(
    method: str,
    model_id: str,
    **kwargs: Any,
) -> Tuple[Optional[LoadedModel], Optional[str]]:
    """Load or return a human-readable unsupported/error reason. Never fakes a model."""
    method = method.lower()
    try:
        if method in {
            "fp32",
            "fp16",
            "bf16",
            "dynamic_int8",
            "int8_bnb",
            "int4_bnb",
            "bitsandbytes_int8",
            "bitsandbytes_int4",
            "nf4",
            "gptq",
            "awq",
        }:
            return load_transformers(model_id, method=method, **kwargs), None
        if method in {"gguf", "gguf_q4_k_m", "llama.cpp"}:
            return load_gguf(model_id, **kwargs), None
        if method == "vllm":
            return None, "vLLM backend is not enabled in the local engine; install vLLM on CUDA and use a dedicated config"
        if method in {"tensorrt_llm", "smooth_quant", "smoothquant", "squeeze_llm", "squeezellm"}:
            return None, f"{method} is not implemented on this platform"
        return None, f"unknown method: {method}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
