"""Optional real perplexity on a short built-in passage. Never uses mocked MMLU/GSM8K."""

from __future__ import annotations

from typing import Any, Optional

WIKI_SNIPPET = (
    "Language models predict the next token in a sequence. "
    "Quantization reduces weight precision to save memory and often improves "
    "throughput on compatible kernels. Time to first token is dominated by prefill, "
    "while decode is typically memory-bandwidth bound. KV cache size grows with "
    "sequence length, number of layers, and number of KV heads. "
    "Reproducible inference research records hardware, software versions, and seeds."
)


def compute_perplexity(model: Any, tokenizer: Any, device: str, text: str = WIKI_SNIPPET) -> Optional[float]:
    try:
        import torch
        import math
    except Exception:
        return None

    try:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            return None
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is None:
                return None
            ppl = float(math.exp(min(float(loss.item()), 20.0)))
            return ppl
    except Exception:
        return None
