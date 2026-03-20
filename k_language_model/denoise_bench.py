import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import K2Layer, MLP, RMSNorm
from .runtime import (
    DEVICE,
    LOG,
    _autocast_context,
    configure_reproducibility,
    log_runtime_metadata,
    maybe_enable_compile,
    setup_logging,
)


@dataclass(frozen=True)
class SyntheticDenoiseConfig:
    seq_len: int
    vocab_size: int
    period_min: int
    period_max: int
    copy_spans: int
    copy_min_len: int
    copy_max_len: int
    mutation_prob: float
    iid_mask_prob: float
    iid_random_prob: float
    burst_count: int
    burst_min_len: int
    burst_max_len: int


def _randint_scalar(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _build_clean_sequences(num_samples: int, cfg: SyntheticDenoiseConfig, generator: torch.Generator) -> torch.Tensor:
    seq_len = cfg.seq_len
    clean = torch.empty(num_samples, seq_len, dtype=torch.long)
    positions = torch.arange(seq_len, dtype=torch.long)

    for row in range(num_samples):
        period = _randint_scalar(generator, cfg.period_min, cfg.period_max + 1)
        motif = torch.randint(0, cfg.vocab_size, (period,), generator=generator)
        phase = _randint_scalar(generator, 0, period)
        seq = motif[(positions + phase) % period].clone()

        for _ in range(cfg.copy_spans):
            span_len = _randint_scalar(generator, cfg.copy_min_len, cfg.copy_max_len + 1)
            if span_len >= seq_len:
                continue
            src_limit = seq_len - (2 * span_len) + 1
            if src_limit <= 0:
                continue
            src_start = _randint_scalar(generator, 0, src_limit)
            dst_start = _randint_scalar(generator, src_start + span_len, seq_len - span_len + 1)
            seq[dst_start: dst_start + span_len] = seq[src_start: src_start + span_len]

        mutation_mask = torch.rand(seq_len, generator=generator) < cfg.mutation_prob
        num_mutations = int(mutation_mask.sum().item())
        if num_mutations > 0:
            seq[mutation_mask] = torch.randint(0, cfg.vocab_size, (num_mutations,), generator=generator)

        clean[row] = seq

    return clean


def _corrupt_sequences(
    clean: torch.Tensor,
    cfg: SyntheticDenoiseConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = clean.clone()
    corrupt = torch.zeros_like(clean, dtype=torch.bool)
    mask_id = cfg.vocab_size

    draw = torch.rand(clean.shape, generator=generator)
    mask_sel = draw < cfg.iid_mask_prob
    rand_sel = (draw >= cfg.iid_mask_prob) & (draw < (cfg.iid_mask_prob + cfg.iid_random_prob))
    noisy[mask_sel] = mask_id
    rand_count = int(rand_sel.sum().item())
    if rand_count > 0:
        noisy[rand_sel] = torch.randint(0, cfg.vocab_size, (rand_count,), generator=generator)
    corrupt |= mask_sel | rand_sel

    for row in range(clean.size(0)):
        for _ in range(cfg.burst_count):
            span_len = _randint_scalar(generator, cfg.burst_min_len, cfg.burst_max_len + 1)
            if span_len >= cfg.seq_len:
                continue
            start = _randint_scalar(generator, 0, cfg.seq_len - span_len + 1)
            corrupt[row, start: start + span_len] = True
            if bool(torch.rand(1, generator=generator).item() < 0.7):
                noisy[row, start: start + span_len] = mask_id
            else:
                noisy[row, start: start + span_len] = torch.randint(
                    0, cfg.vocab_size, (span_len,), generator=generator
                )

    return noisy, corrupt


def build_synthetic_dataset(
    num_samples: int,
    cfg: SyntheticDenoiseConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    clean = _build_clean_sequences(num_samples, cfg, generator)
    noisy, corrupt = _corrupt_sequences(clean, cfg, generator)
    return noisy, clean, corrupt


class Conv1DBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.norm1 = RMSNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout=0.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        x = self.norm1(h).transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x).transpose(1, 2)
        h = residual + x
        h = h + self.mlp(self.norm2(h))
        return h


class SingleBlockDenoiser(nn.Module):
    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        rank: int,
        conv_kernel: int,
        decay_impl: str,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_vocab_size = vocab_size + 1
        self.emb = nn.Embedding(self.input_vocab_size, d_model)
        if backbone == "conv1d":
            self.block = Conv1DBlock(d_model=d_model, kernel_size=conv_kernel)
        elif backbone == "k2":
            self.block = K2Layer(
                window=seq_len,
                d=d_model,
                rank=rank,
                mlp_dropout=0.0,
                residual_dropout=0.0,
                alpha_cap=1.0,
                decay_impl=decay_impl,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.block(h)
        h = self.norm(h)
        return self.head(h)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    corrupt_mask: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    flat_mask = corrupt_mask.reshape(-1)
    flat_targets = targets.reshape(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    selected_logits = flat_logits[flat_mask]
    selected_targets = flat_targets[flat_mask]
    if selected_targets.numel() == 0:
        zero = flat_logits.sum() * 0.0
        return zero, 0, 0
    loss = F.cross_entropy(selected_logits.float(), selected_targets, reduction="sum")
    correct = int((selected_logits.argmax(dim=-1) == selected_targets).sum().item())
    return loss, int(selected_targets.numel()), correct


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for start in range(0, noisy.size(0), batch_size):
        x = noisy[start: start + batch_size].to(DEVICE, non_blocking=True)
        y = clean[start: start + batch_size].to(DEVICE, non_blocking=True)
        m = corrupt[start: start + batch_size].to(DEVICE, non_blocking=True)
        with _autocast_context():
            logits = model(x)
            loss_sum, count, correct = _masked_cross_entropy(logits, y, m)
        total_loss += float(loss_sum.item())
        total_tokens += count
        total_correct += correct

    ce = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    return {"ce": ce, "acc": acc, "tokens": float(total_tokens)}


def _sample_train_batch(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, noisy.size(0), (batch_size,), generator=generator)
    return noisy[indices], clean[indices], corrupt[indices]


def run_single_benchmark(args: argparse.Namespace, backbone: str) -> dict[str, float]:
    synth_cfg = SyntheticDenoiseConfig(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        period_min=args.period_min,
        period_max=args.period_max,
        copy_spans=args.copy_spans,
        copy_min_len=args.copy_min_len,
        copy_max_len=args.copy_max_len,
        mutation_prob=args.mutation_prob,
        iid_mask_prob=args.iid_mask_prob,
        iid_random_prob=args.iid_random_prob,
        burst_count=args.burst_count,
        burst_min_len=args.burst_min_len,
        burst_max_len=args.burst_max_len,
    )
    train_noisy, train_clean, train_corrupt = build_synthetic_dataset(
        num_samples=args.train_samples,
        cfg=synth_cfg,
        seed=args.seed + 101,
    )
    val_noisy, val_clean, val_corrupt = build_synthetic_dataset(
        num_samples=args.val_samples,
        cfg=synth_cfg,
        seed=args.seed + 202,
    )

    torch.manual_seed(args.seed + (11 if backbone == "conv1d" else 29))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + (11 if backbone == "conv1d" else 29))

    model = SingleBlockDenoiser(
        backbone=backbone,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        rank=args.rank,
        conv_kernel=args.conv_kernel,
        decay_impl=args.decay_impl,
    ).to(DEVICE)
    param_count = model.count_params()
    model = maybe_enable_compile(model, enabled=args.compile, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_generator = torch.Generator().manual_seed(args.seed + 303)
    best_val_ce = float("inf")
    best_val_acc = 0.0
    last_eval_time = time.perf_counter()
    last_eval_step = 0

    LOG.info(
        "Denoise benchmark start | backbone=%s | params=%d | seq_len=%d | vocab=%d | train_samples=%d | val_samples=%d",
        backbone,
        param_count,
        args.seq_len,
        args.vocab_size,
        args.train_samples,
        args.val_samples,
    )

    for step in range(1, args.steps + 1):
        model.train()
        x, y, m = _sample_train_batch(
            train_noisy,
            train_clean,
            train_corrupt,
            batch_size=args.batch_size,
            generator=train_generator,
        )
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context():
            logits = model(x)
            loss_sum, token_count, correct = _masked_cross_entropy(logits, y, m)
            loss = loss_sum / max(token_count, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            elapsed = time.perf_counter() - last_eval_time
            step_delta = step - last_eval_step
            train_tokens = step_delta * args.batch_size * args.seq_len
            train_tok_s = train_tokens / max(elapsed, 1e-6)
            ms_per_step = 1000.0 * elapsed / max(step_delta, 1)
            train_acc = correct / max(token_count, 1)
            metrics = evaluate_model(model, val_noisy, val_clean, val_corrupt, batch_size=args.batch_size)
            best_val_ce = min(best_val_ce, metrics["ce"])
            best_val_acc = max(best_val_acc, metrics["acc"])
            LOG.info(
                "backbone=%s | step=%d | train_ce=%.4f | train_acc=%.4f | val_ce=%.4f | val_acc=%.4f | best_val_ce=%.4f | best_val_acc=%.4f | %.1f ms/step | %.0f tok/s",
                backbone,
                step,
                float(loss.item()),
                train_acc,
                metrics["ce"],
                metrics["acc"],
                best_val_ce,
                best_val_acc,
                ms_per_step,
                train_tok_s,
            )
            last_eval_time = time.perf_counter()
            last_eval_step = step

    return {
        "params": float(param_count),
        "best_val_ce": best_val_ce,
        "best_val_acc": best_val_acc,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic denoising benchmark: Conv1d vs single K2 layer.")
    p.add_argument("--backbone", choices=["conv1d", "k2", "both"], default="both")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--conv-kernel", type=int, default=7)
    p.add_argument("--decay-impl", choices=["mask", "block", "kernel"], default="mask")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--period-min", type=int, default=4)
    p.add_argument("--period-max", type=int, default=16)
    p.add_argument("--copy-spans", type=int, default=1)
    p.add_argument("--copy-min-len", type=int, default=4)
    p.add_argument("--copy-max-len", type=int, default=10)
    p.add_argument("--mutation-prob", type=float, default=0.0)
    p.add_argument("--iid-mask-prob", type=float, default=0.10)
    p.add_argument("--iid-random-prob", type=float, default=0.05)
    p.add_argument("--burst-count", type=int, default=1)
    p.add_argument("--burst-min-len", type=int, default=4)
    p.add_argument("--burst-max-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--deterministic-warn-only", action="store_true")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)
    configure_reproducibility(
        seed=args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        allow_tf32=not args.no_tf32,
    )
    log_runtime_metadata()

    backbones = ["conv1d", "k2"] if args.backbone == "both" else [args.backbone]
    results = []
    for backbone in backbones:
        results.append((backbone, run_single_benchmark(args, backbone)))

    if len(results) > 1:
        summary = " | ".join(
            f"{name}: params={int(metrics['params'])} best_val_ce={metrics['best_val_ce']:.4f} best_val_acc={metrics['best_val_acc']:.4f}"
            for name, metrics in results
        )
        LOG.info("Benchmark summary | %s", summary)


if __name__ == "__main__":
    main()
