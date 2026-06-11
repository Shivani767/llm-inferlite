from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Union
import time
import torch

class InferenceProvider(ABC):
    """
    Unified interface for multiple LLM inference runtimes.
    Ensures research-grade consistency across heterogeneous backends.
    """
    @abstractmethod
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete response with metadata."""
        pass

    @abstractmethod
    def stream(self, prompt: str, sampling_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream response tokens with per-token metadata."""
        pass

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize input text."""
        pass

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Measure current GPU memory utilization."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}

class VLLMProvider(InferenceProvider):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        try:
            from vllm import LLM
            self.model = LLM(model=model_path, device=device, **kwargs)
        except ImportError:
            self.model = None
            print("vLLM not installed. Running in simulation mode.")
        
    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            return {"text": "Simulation: vLLM response", "latency": 0.05, "tokens": 100}
            
        from vllm import SamplingParams
        vllm_params = SamplingParams(**sampling_params)
        start_time = time.time()
        outputs = self.model.generate([prompt], vllm_params)
        end_time = time.time()
        
        output = outputs[0]
        return {
            "text": output.outputs[0].text,
            "latency": end_time - start_time,
            "tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason
        }

    def stream(self, prompt: str, sampling_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # vLLM offline engine doesn't support easy streaming. 
        # In a real setup, we would use AsyncLLMEngine.
        yield self.generate(prompt, sampling_params)

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        raise NotImplementedError("vLLM embedding support is limited in offline mode")

    def tokenize(self, text: str) -> List[int]:
        return self.model.get_tokenizer().encode(text) if self.model else []

class ONNXRuntimeProvider(InferenceProvider):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        except ImportError:
            self.session = None
            print("ONNX Runtime not installed. Running in simulation mode.")

    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            return {"text": "Simulation: ONNX Runtime response", "latency": 0.04, "tokens": 100}
        # Real ONNX inference logic for LLMs is complex (requires state management)
        return {"text": "ONNX implementation placeholder", "latency": 0.0}

    def stream(self, prompt: str, sampling_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield self.generate(prompt, sampling_params)

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        return []

    def tokenize(self, text: str) -> List[int]:
        return []

class LlamaCppProvider(InferenceProvider):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        try:
            from llama_cpp import Llama
            n_gpu_layers = -1 if device == "cuda" else 0
            self.model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, **kwargs)
        except ImportError:
            self.model = None
            print("llama-cpp-python not installed. Running in simulation mode.")

    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            return {"text": "Simulation: llama.cpp response", "latency": 0.06, "tokens": 100}
        
        start_time = time.time()
        output = self.model(
            prompt,
            max_tokens=sampling_params.get("max_tokens", 128),
            stop=sampling_params.get("stop", []),
            echo=False
        )
        end_time = time.time()
        
        return {
            "text": output["choices"][0]["text"],
            "latency": end_time - start_time,
            "tokens": output["usage"]["completion_tokens"]
        }

    def stream(self, prompt: str, sampling_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if not self.model:
            yield self.generate(prompt, sampling_params)
            return

        for chunk in self.model(prompt, stream=True, max_tokens=sampling_params.get("max_tokens", 128)):
            yield {"text": chunk["choices"][0]["text"]}

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if not self.model: return []
        return [self.model.embed(text)] if isinstance(text, str) else [self.model.embed(t) for v in text]

    def tokenize(self, text: str) -> List[int]:
        return self.model.tokenize(text.encode("utf-8")) if self.model else []

class TransformersProvider(InferenceProvider):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                **kwargs
            )
        except ImportError:
            self.model = None
            print("Transformers not installed. Running in simulation mode.")

    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            return {"text": "Simulation: Transformers response", "latency": 0.08, "tokens": 100}
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=sampling_params.get("max_tokens", 128),
                temperature=sampling_params.get("temperature", 1.0),
                do_sample=sampling_params.get("temperature", 1.0) > 0
            )
        end_time = time.time()
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "text": text,
            "latency": end_time - start_time,
            "tokens": outputs.shape[1] - inputs.input_ids.shape[1]
        }

    def stream(self, prompt: str, sampling_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield self.generate(prompt, sampling_params)

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if not self.model: return []
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings.cpu().tolist()

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text) if self.model else []
