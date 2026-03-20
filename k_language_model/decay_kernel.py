from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_MAX_KERNEL_RANK = 64


def is_decay_kernel_available(device: torch.device, rank: int) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if device.type != "cuda":
        return False
    if rank <= 0:
        return False
    return rank <= _MAX_KERNEL_RANK


def _next_power_of_two(x: int) -> int:
    x = max(int(x), 1)
    return 1 << (x - 1).bit_length()


if _TRITON_AVAILABLE:
    _KERNEL_CONFIGS = [
        triton.Config({"BLOCK_D": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["R", "D"])
    @triton.jit
    def _decay_forward_kernel(
        q_ptr,
        k_ptr,
        h_ptr,
        gamma_ptr,
        out_ptr,
        final_state_ptr,
        B,
        W,
        R,
        D,
        q_stride_b,
        q_stride_w,
        q_stride_r,
        k_stride_b,
        k_stride_w,
        k_stride_r,
        h_stride_b,
        h_stride_w,
        h_stride_d,
        out_stride_b,
        out_stride_w,
        out_stride_d,
        state_stride_b,
        state_stride_r,
        state_stride_d,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d_blk = tl.program_id(1)

        offs_d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        gamma = tl.load(gamma_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        state = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)

        q_b_ptr = q_ptr + pid_b * q_stride_b
        k_b_ptr = k_ptr + pid_b * k_stride_b
        h_b_ptr = h_ptr + pid_b * h_stride_b
        out_b_ptr = out_ptr + pid_b * out_stride_b
        state_b_ptr = final_state_ptr + pid_b * state_stride_b

        for t in range(0, W):
            h_t = tl.load(h_b_ptr + t * h_stride_w + offs_d * h_stride_d, mask=mask_d, other=0.0).to(tl.float32)
            k_t = tl.load(k_b_ptr + t * k_stride_w + offs_r * k_stride_r, mask=mask_r, other=0.0).to(tl.float32)
            q_t = tl.load(q_b_ptr + t * q_stride_w + offs_r * q_stride_r, mask=mask_r, other=0.0).to(tl.float32)

            state = gamma[:, None] * state + k_t[:, None] * h_t[None, :]
            out_t = tl.sum(q_t[:, None] * state, axis=0)
            tl.store(out_b_ptr + t * out_stride_w + offs_d * out_stride_d, out_t, mask=mask_d)

        state_ptrs = state_b_ptr + offs_r[:, None] * state_stride_r + offs_d[None, :] * state_stride_d
        tl.store(state_ptrs, state, mask=mask_r[:, None] & mask_d[None, :])

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["R", "D"])
    @triton.jit
    def _decay_backward_kernel(
        q_ptr,
        k_ptr,
        h_ptr,
        gamma_ptr,
        final_state_ptr,
        grad_out_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_h_ptr,
        grad_gamma_ptr,
        B,
        W,
        R,
        D,
        q_stride_b,
        q_stride_w,
        q_stride_r,
        k_stride_b,
        k_stride_w,
        k_stride_r,
        h_stride_b,
        h_stride_w,
        h_stride_d,
        state_stride_b,
        state_stride_r,
        state_stride_d,
        go_stride_b,
        go_stride_w,
        go_stride_d,
        gq_stride_b,
        gq_stride_w,
        gq_stride_r,
        gk_stride_b,
        gk_stride_w,
        gk_stride_r,
        gh_stride_b,
        gh_stride_w,
        gh_stride_d,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d_blk = tl.program_id(1)

        offs_d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        gamma = tl.load(gamma_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        inv_gamma = tl.where(mask_r, 1.0 / tl.maximum(gamma, 1e-8), 0.0)

        q_b_ptr = q_ptr + pid_b * q_stride_b
        k_b_ptr = k_ptr + pid_b * k_stride_b
        h_b_ptr = h_ptr + pid_b * h_stride_b
        go_b_ptr = grad_out_ptr + pid_b * go_stride_b
        gq_b_ptr = grad_q_ptr + pid_b * gq_stride_b
        gk_b_ptr = grad_k_ptr + pid_b * gk_stride_b
        gh_b_ptr = grad_h_ptr + pid_b * gh_stride_b
        state_b_ptr = final_state_ptr + pid_b * state_stride_b

        state_ptrs = state_b_ptr + offs_r[:, None] * state_stride_r + offs_d[None, :] * state_stride_d
        state_t = tl.load(state_ptrs, mask=mask_r[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        grad_state_next = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        grad_gamma_acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        for rev in range(0, W):
            t = W - 1 - rev

            h_t = tl.load(h_b_ptr + t * h_stride_w + offs_d * h_stride_d, mask=mask_d, other=0.0).to(tl.float32)
            go_t = tl.load(go_b_ptr + t * go_stride_w + offs_d * go_stride_d, mask=mask_d, other=0.0).to(tl.float32)
            k_t = tl.load(k_b_ptr + t * k_stride_w + offs_r * k_stride_r, mask=mask_r, other=0.0).to(tl.float32)
            q_t = tl.load(q_b_ptr + t * q_stride_w + offs_r * q_stride_r, mask=mask_r, other=0.0).to(tl.float32)

            s_prev = (state_t - k_t[:, None] * h_t[None, :]) * inv_gamma[:, None]
            grad_state = grad_state_next + q_t[:, None] * go_t[None, :]

            grad_h_t = tl.sum(grad_state * k_t[:, None], axis=0)
            tl.store(gh_b_ptr + t * gh_stride_w + offs_d * gh_stride_d, grad_h_t, mask=mask_d)

            grad_q_t = tl.sum(go_t[None, :] * state_t, axis=1)
            grad_k_t = tl.sum(grad_state * h_t[None, :], axis=1)
            grad_gamma_acc += tl.sum(grad_state * s_prev, axis=1)

            tl.atomic_add(gq_b_ptr + t * gq_stride_w + offs_r * gq_stride_r, grad_q_t, mask=mask_r)
            tl.atomic_add(gk_b_ptr + t * gk_stride_w + offs_r * gk_stride_r, grad_k_t, mask=mask_r)

            grad_state_next = grad_state * gamma[:, None]
            state_t = s_prev

        tl.atomic_add(grad_gamma_ptr + offs_r, grad_gamma_acc, mask=mask_r)


def _decay_forward_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, window, rank = q.shape
    d_model = h.size(-1)
    block_r = _next_power_of_two(rank)

    out = torch.empty((batch, window, d_model), device=h.device, dtype=h.dtype)
    final_state = torch.empty((batch, rank, d_model), device=h.device, dtype=torch.float32)

    grid = lambda meta: (batch, triton.cdiv(d_model, meta["BLOCK_D"]))
    _decay_forward_kernel[grid](
        q,
        k,
        h,
        gamma,
        out,
        final_state,
        batch,
        window,
        rank,
        d_model,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        h.stride(0),
        h.stride(1),
        h.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        BLOCK_R=block_r,
    )
    return out, final_state


def _decay_backward_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gamma: torch.Tensor,
    final_state: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, window, rank = q.shape
    d_model = h.size(-1)
    block_r = _next_power_of_two(rank)

    grad_q_acc = torch.zeros((batch, window, rank), device=q.device, dtype=torch.float32)
    grad_k_acc = torch.zeros((batch, window, rank), device=q.device, dtype=torch.float32)
    grad_h_acc = torch.empty((batch, window, d_model), device=h.device, dtype=torch.float32)
    grad_gamma_acc = torch.zeros((rank,), device=gamma.device, dtype=torch.float32)

    grid = lambda meta: (batch, triton.cdiv(d_model, meta["BLOCK_D"]))
    _decay_backward_kernel[grid](
        q,
        k,
        h,
        gamma,
        final_state,
        grad_out,
        grad_q_acc,
        grad_k_acc,
        grad_h_acc,
        grad_gamma_acc,
        batch,
        window,
        rank,
        d_model,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        h.stride(0),
        h.stride(1),
        h.stride(2),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_q_acc.stride(0),
        grad_q_acc.stride(1),
        grad_q_acc.stride(2),
        grad_k_acc.stride(0),
        grad_k_acc.stride(1),
        grad_k_acc.stride(2),
        grad_h_acc.stride(0),
        grad_h_acc.stride(1),
        grad_h_acc.stride(2),
        BLOCK_R=block_r,
    )
    return (
        grad_q_acc.to(dtype=q.dtype),
        grad_k_acc.to(dtype=k.dtype),
        grad_h_acc.to(dtype=h.dtype),
        grad_gamma_acc.to(dtype=gamma.dtype),
    )


def _decay_backward_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gamma: torch.Tensor,
    final_state: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, window, rank = q.shape
    d_model = h.size(-1)

    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_h = torch.zeros_like(h)
    grad_gamma = torch.zeros_like(gamma)

    gamma_f = gamma.to(dtype=torch.float32).view(1, rank, 1)
    inv_gamma = torch.reciprocal(gamma_f.clamp(min=1e-8))

    state_t = final_state.to(dtype=torch.float32)
    grad_state_next = torch.zeros((batch, rank, d_model), device=h.device, dtype=torch.float32)

    for t in range(window - 1, -1, -1):
        h_t = h[:, t, :].to(dtype=torch.float32)
        k_t = k[:, t, :].to(dtype=torch.float32)
        q_t = q[:, t, :].to(dtype=torch.float32)
        go_t = grad_out[:, t, :].to(dtype=torch.float32)

        s_prev = (state_t - k_t.unsqueeze(-1) * h_t.unsqueeze(1)) * inv_gamma
        grad_state = grad_state_next + q_t.unsqueeze(-1) * go_t.unsqueeze(1)

        grad_q[:, t, :] = (go_t.unsqueeze(1) * state_t).sum(dim=-1).to(dtype=q.dtype)
        grad_k[:, t, :] = (grad_state * h_t.unsqueeze(1)).sum(dim=-1).to(dtype=k.dtype)
        grad_h[:, t, :] = (grad_state * k_t.unsqueeze(-1)).sum(dim=1).to(dtype=h.dtype)
        grad_gamma.add_((grad_state * s_prev).sum(dim=(0, 2)).to(dtype=gamma.dtype))

        grad_state_next = grad_state * gamma_f
        state_t = s_prev

    return grad_q, grad_k, grad_h, grad_gamma


class _DecayKernelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, h: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        out, final_state = _decay_forward_cuda(q, k, h, gamma)
        ctx.save_for_backward(q, k, h, gamma, final_state)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, h, gamma, final_state = ctx.saved_tensors

        need_q, need_k, need_h, need_gamma = ctx.needs_input_grad
        if not (need_q or need_k or need_h or need_gamma):
            return None, None, None, None

        grad_out = grad_out.contiguous()
        use_cuda_kernel = is_decay_kernel_available(q.device, q.size(-1))
        if use_cuda_kernel:
            try:
                grad_q, grad_k, grad_h, grad_gamma = _decay_backward_cuda(
                    q=q,
                    k=k,
                    h=h,
                    gamma=gamma,
                    final_state=final_state,
                    grad_out=grad_out,
                )
            except Exception:
                grad_q, grad_k, grad_h, grad_gamma = _decay_backward_torch(
                    q=q,
                    k=k,
                    h=h,
                    gamma=gamma,
                    final_state=final_state,
                    grad_out=grad_out,
                )
        else:
            grad_q, grad_k, grad_h, grad_gamma = _decay_backward_torch(
                q=q,
                k=k,
                h=h,
                gamma=gamma,
                final_state=final_state,
                grad_out=grad_out,
            )

        return (
            grad_q if need_q else None,
            grad_k if need_k else None,
            grad_h if need_h else None,
            grad_gamma if need_gamma else None,
        )


def decay_kernel(q: torch.Tensor, k: torch.Tensor, h: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if not is_decay_kernel_available(q.device, q.size(-1)):
        raise RuntimeError("Decay Triton kernel is unavailable for this device/rank.")
    return _DecayKernelFn.apply(q.contiguous(), k.contiguous(), h.contiguous(), gamma.contiguous())
