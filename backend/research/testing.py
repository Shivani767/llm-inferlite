"""Offline tiny causal LM for tests and the local demo (no Hub download)."""

from __future__ import annotations

from typing import List

import torch
from transformers import GPT2Config, GPT2LMHeadModel


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<pad>"
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 128

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        if isinstance(text, list):
            ids = [self._encode(t) for t in text]
            max_len = max(len(x) for x in ids)
            if padding:
                ids = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
            input_ids = torch.tensor(ids, dtype=torch.long)
            mask = (input_ids != self.pad_token_id).long()
            return {"input_ids": input_ids, "attention_mask": mask}
        ids = self._encode(text)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def _encode(self, text: str) -> List[int]:
        if not text:
            return [1]
        return [min(ord(c), 127) for c in text[:48]] or [1]

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return "".join(chr(int(i)) if 32 <= int(i) < 127 else "?" for i in ids)

    def encode(self, text: str) -> List[int]:
        return self._encode(text)


def make_tiny_lm(n_layer: int = 2, n_embd: int = 32, n_head: int = 2, seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model
