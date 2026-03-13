from typing import Dict

import torch
import torch.nn.functional as F

from .model import KStackModel
from .runtime import DEVICE, _unwrap_model


@torch.no_grad()
def sample_text(
    model: KStackModel,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    max_new_tokens: int,
    window: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 256,
    prompt_lock_chars: int = 0,
) -> str:
    model.eval()
    core_model = _unwrap_model(model)
    context = [stoi[ch] for ch in prompt if ch in stoi]
    if not context:
        context = [0]

    x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        if prompt_lock_chars > 0 and x.size(1) > window:
            lock_len = min(int(prompt_lock_chars), max(window - 1, 0), x.size(1))
            tail_len = window - lock_len
            if tail_len > 0:
                x_cond = torch.cat([x[:, :lock_len], x[:, -tail_len:]], dim=1)
            else:
                x_cond = x[:, :window]
        else:
            x_cond = x[:, -window:]
        logits = core_model(x_cond)[:, -1, :] / max(temperature, 1e-6)
        if repetition_penalty > 1.0:
            if repetition_window > 0:
                seen = x[:, -min(int(repetition_window), x.size(1)):]
            else:
                seen = x
            for b in range(logits.size(0)):
                seen_ids = torch.unique(seen[b])
                seen_logits = logits[b, seen_ids]
                seen_logits = torch.where(
                    seen_logits > 0,
                    seen_logits / float(repetition_penalty),
                    seen_logits * float(repetition_penalty),
                )
                logits[b, seen_ids] = seen_logits
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            top_vals, _ = torch.topk(logits, k, dim=-1)
            kth = top_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove_mask = cumulative_probs > float(top_p)
            sorted_remove_mask[:, 0] = False
            remove_mask = torch.zeros_like(sorted_remove_mask, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove_mask)
            logits = logits.masked_fill(remove_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return "".join(itos[int(i)] for i in x[0].tolist())
