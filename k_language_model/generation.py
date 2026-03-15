import torch
import torch.nn.functional as F

from .data import TextTokenizer
from .model import KStackModel
from .runtime import DEVICE, _unwrap_model


@torch.no_grad()
def sample_text(
    model: KStackModel,
    tokenizer: TextTokenizer,
    prompt: str,
    max_new_tokens: int,
    window: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 256,
    prompt_lock_tokens: int = 0,
) -> str:
    model.eval()
    core_model = _unwrap_model(model)
    context = tokenizer.encode(prompt)
    if not context:
        context = [0]

    x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        if prompt_lock_tokens > 0 and x.size(1) > window:
            lock_len = min(int(prompt_lock_tokens), max(window - 1, 0), x.size(1))
            tail_len = window - lock_len
            if tail_len > 0:
                x_cond = torch.cat([x[:, :lock_len], x[:, -tail_len:]], dim=1)
            else:
                x_cond = x[:, :window]
        else:
            x_cond = x[:, -window:]
        scores = core_model(x_cond)[:, -1, :] / max(temperature, 1e-6)
        if repetition_penalty > 1.0 and scores.size(-1) > 0:
            if repetition_window > 0:
                seen = x[:, -min(int(repetition_window), x.size(1)):]
            else:
                seen = x
            for b in range(scores.size(0)):
                seen_ids = torch.unique(seen[b])
                seen_scores = scores[b, seen_ids]
                seen_scores = torch.where(
                    seen_scores > 0,
                    seen_scores / float(repetition_penalty),
                    seen_scores * float(repetition_penalty),
                )
                scores[b, seen_ids] = seen_scores
        if top_k is not None and top_k > 0:
            k = min(int(top_k), scores.size(-1))
            top_vals, _ = torch.topk(scores, k, dim=-1)
            kth = top_vals[:, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove_mask = cumulative_probs > float(top_p)
            sorted_remove_mask[:, 0] = False
            remove_mask = torch.zeros_like(sorted_remove_mask, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove_mask)
            scores = scores.masked_fill(remove_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())
