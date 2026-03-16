# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Causal OmniAvatar wrapper for the student network in Self-Forcing distillation.

This module implements the causal variant of OmniAvatar's DiT, which adds:
  1. Causal self-attention via FlexAttention block masks
  2. KV cache management for chunk-by-chunk autoregressive generation
  3. Causal RoPE with frame offset
  4. Per-chunk audio slicing

The causal model is structurally DIFFERENT from the bidirectional model — it has
its own transformer block (``CausalDiTBlock``) with causal self-attention
(``CausalSelfAttention``) and KV cache support.  However, the WEIGHTS are
compatible: both variants share patch_embedding, text_embedding, time_embedding,
time_projection, head, and per-block q/k/v/o + cross-attn + ffn parameters.

Architecture reference:
    ``Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py``
    — ``CausalWanModel``, ``CausalWanSelfAttention``, ``CausalWanAttentionBlock``

Weight loading:
    1. Base Wan 2.1 T2V safetensor weights  →  CausalWanModel internals
    2. OmniAvatar LoRA + audio + patch_embedding checkpoint on top
    3. (optional) merge LoRA into base weights for inference
"""

import os
import re
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.OmniAvatar.audio_pack import AudioPack
import fastgen.utils.logging_utils as logger

# Re-use weight-loading utilities from the bidirectional wrapper
from fastgen.networks.OmniAvatar.network import (
    MODEL_CONFIGS,
    _COMMON_CFG,
    LORA_TARGET_MODULES,
    _load_state_dict,
    _smart_load_weights,
    _convert_diffusers_state_dict,
    _merge_lora_into_model,
    _map_omniavatar_lora_keys,
)


# ---------------------------------------------------------------------------
# FlexAttention imports (optional — falls back to SDP if unavailable)
# ---------------------------------------------------------------------------
_disable_flex_env = os.environ.get("FASTGEN_DISABLE_FLEX_ATTENTION", "0") == "1"
try:
    if _disable_flex_env:
        raise ImportError("FlexAttention disabled via env var")
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention as _flex_attention,
        BlockMask,
    )

    FLEX_ATTENTION_AVAILABLE = True

    # Wan 1.3B requires max-autotune for FlexAttention (PyTorch issue #133254)
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.optimize_ddp = False
    except Exception:
        pass

    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
    _disable_compile = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
        or os.environ.get("FASTGEN_FLEX_COMPILE", "1") == "0"
    )
    if not _disable_compile:
        flex_attention = torch.compile(
            _flex_attention, dynamic=False, mode=_compile_mode
        )
    else:
        flex_attention = _flex_attention
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

    class BlockMask:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Flash attention (for KV cache path — not FlexAttention)
# ---------------------------------------------------------------------------
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Flash attention for KV-cache path.  Input format: [B, S, H, D]."""
    half_dtypes = (torch.float16, torch.bfloat16)
    b, lq, lk = q.shape[0], q.shape[1], k.shape[1]
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

    q_flat = half(q.flatten(0, 1))
    k_flat = half(k.flatten(0, 1))
    v_flat = half(v.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)

    if FLASH_ATTN_3_AVAILABLE:
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        out = flash_attn_interface.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk,
        )
        y = out[0] if isinstance(out, (list, tuple)) else out
        return y.unflatten(0, (b, lq)).type(out_dtype)
    elif FLASH_ATTN_2_AVAILABLE:
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return flash_attn.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk,
        ).unflatten(0, (b, lq)).type(out_dtype)
    else:
        # Fallback: scaled dot-product attention
        q_t = q.transpose(1, 2).to(torch.bfloat16)
        k_t = k.transpose(1, 2).to(torch.bfloat16)
        v_t = v.transpose(1, 2).to(torch.bfloat16)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2).contiguous().type(out_dtype)


# ---------------------------------------------------------------------------
# Causal RoPE with frame offset
# ---------------------------------------------------------------------------

def _precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 1D RoPE frequency table as complex exponentials."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim)
    )
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 3D RoPE frequency tables for (F, H, W)."""
    f_freqs = _precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs = _precompute_freqs_cis(dim // 3, end, theta)
    w_freqs = _precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs, h_freqs, w_freqs


def causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply 3D RoPE with a temporal frame offset for causal generation.

    Args:
        x: [B, S, num_heads, head_dim]
        grid_sizes: [B, 3]  — (F, H, W) per sample
        freqs: tuple of 3 complex frequency tables (f, h, w)
        start_frame: temporal offset (number of frames already generated)

    Returns:
        Tensor same shape as x with RoPE applied.
    """
    n, c = x.size(2), x.size(3) // 2
    freq_f, freq_h, freq_w = freqs
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    freq_parts = (freq_f, freq_h, freq_w)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freq_parts[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freq_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # Append any padding tokens unchanged
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)


def rope_apply_full(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply standard (non-causal) 3D RoPE — start_frame=0."""
    return causal_rope_apply(x, grid_sizes, freqs, start_frame=0)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Norm layers (matching the reference CausalWanModel)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching WanRMSNorm from the reference."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm matching WanLayerNorm from the reference."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


# ---------------------------------------------------------------------------
# Causal Self-Attention with KV cache
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal self-attention with KV cache support and FlexAttention.

    In full-sequence mode (kv_cache=None): uses FlexAttention with block mask
    for causal attention over the entire sequence.

    In AR mode (kv_cache provided): uses flash attention with gradient-safe
    KV caching (detached writes, cat [detached_past | live_current]).

    Supports:
      - ``use_dynamic_rope``: cache raw K, apply window-local RoPE at attention
        time (better positional generalisation for long contexts).
      - ``local_attn_size``: rolling local attention window (in frames) with
        eviction.  ``-1`` means attend to everything.
      - ``sink_size``: number of initial frames always kept in the window
        (attention-sink tokens, never evicted).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_dynamic_rope = use_dynamic_rope

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        block_mask=None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        store_kv: bool = True,
        cache_local_end_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            seq_lens: [B]
            grid_sizes: [B, 3] — (F, H, W)
            freqs: 3D RoPE freq tables
            block_mask: FlexAttention block mask (for full-sequence causal mode)
            kv_cache: dict with 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: token offset for causal RoPE in AR mode
            store_kv: if True, write to cache and update metadata; if False,
                      use existing cache + live K/V for attention only.
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            # ----- Full-sequence mode (training / bidirectional eval) -----
            roped_q = rope_apply_full(q, grid_sizes, freqs).type_as(v)
            roped_k = rope_apply_full(k, grid_sizes, freqs).type_as(v)

            if block_mask is not None and FLEX_ATTENTION_AVAILABLE:
                # FlexAttention path
                pad_len = math.ceil(s / 128) * 128 - s
                if pad_len > 0:
                    pad_shape = (b, pad_len, n, d)
                    roped_q = torch.cat(
                        [roped_q, torch.zeros(pad_shape, device=q.device, dtype=v.dtype)],
                        dim=1,
                    )
                    roped_k = torch.cat(
                        [roped_k, torch.zeros(pad_shape, device=k.device, dtype=v.dtype)],
                        dim=1,
                    )
                    v_padded = torch.cat(
                        [v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)],
                        dim=1,
                    )
                else:
                    v_padded = v

                out = flex_attention(
                    query=roped_q.transpose(1, 2),
                    key=roped_k.transpose(1, 2),
                    value=v_padded.transpose(1, 2),
                    block_mask=block_mask,
                )
                if pad_len > 0:
                    out = out[:, :, :-pad_len]
                x = out.transpose(1, 2)  # [B, S, H, D]
            else:
                # Standard flash attention (no causal mask — fully bidirectional)
                x = _flash_attention(roped_q, roped_k, v)
        else:
            # ----- AR mode with KV cache (FastGen-style gradient-safe) -----
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            num_new_tokens = q.shape[1]
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen

            # 1. Prepare K for caching (mode-dependent)
            if not self.use_dynamic_rope:
                roped_q = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                roped_k = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                k_to_cache = roped_k
            else:
                k_to_cache = k  # cache raw keys; RoPE applied later

            # 2. Read cache metadata.
            # IMPORTANT: Use the explicit override (frozen before the block loop)
            # to ensure gradient-checkpointing recomputation sees the same state.
            # This follows FastGen's CausalWan pattern: proper_cache_len is computed
            # OUTSIDE the block loop and passed in, never read from mutable cache.
            kv_cache_size = kv_cache["k"].shape[1]
            if cache_local_end_override is not None:
                local_end = cache_local_end_override
                global_end = cache_local_end_override  # In no-eviction mode, they're equal
            else:
                global_end = kv_cache["global_end_index"].item()
                local_end = kv_cache["local_end_index"].item()

            # 3. Handle eviction (only when store_kv=True and cache overflows)
            if (
                store_kv
                and self.local_attn_size > 0
                and current_end > global_end
                and num_new_tokens + local_end > kv_cache_size
            ):
                # Rolling eviction needed
                num_evicted = num_new_tokens + local_end - kv_cache_size
                num_rolled = local_end - num_evicted - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
                new_local_end = local_end + (current_end - global_end) - num_evicted
                new_local_start = new_local_end - num_new_tokens
            else:
                # No eviction: simple append
                new_local_end = local_end + max(0, current_end - global_end)
                new_local_start = new_local_end - num_new_tokens

            # 4. Write to cache (detached, only if store_kv)
            if store_kv:
                kv_cache["k"][:, new_local_start:new_local_end] = k_to_cache.detach()
                kv_cache["v"][:, new_local_start:new_local_end] = v.detach()

            # 5. Build attention window
            max_attn_tokens = (
                self.local_attn_size * frame_seqlen
                if self.local_attn_size > 0
                else new_local_end
            )
            k_win_start = max(0, new_local_end - max_attn_tokens)

            if sink_tokens > 0 and k_win_start > 0:
                # Sink + rolling window (non-contiguous)
                available_rolling = max_attn_tokens - sink_tokens
                rolling_start = max(sink_tokens, new_local_end - available_rolling)

                with torch.no_grad():
                    k_past = torch.cat(
                        [kv_cache["k"][:, :sink_tokens],
                         kv_cache["k"][:, rolling_start:new_local_start]],
                        dim=1,
                    )
                    v_past = torch.cat(
                        [kv_cache["v"][:, :sink_tokens],
                         kv_cache["v"][:, rolling_start:new_local_start]],
                        dim=1,
                    )
                k_win = torch.cat([k_past, k_to_cache], dim=1)
                v_win = torch.cat([v_past, v], dim=1)

                # Query position within the attention window
                query_offset_in_win = sink_tokens + (new_local_start - rolling_start)
            else:
                # Simple contiguous case
                if new_local_start == 0:
                    k_win = k_to_cache
                    v_win = v
                else:
                    with torch.no_grad():
                        k_past = kv_cache["k"][:, k_win_start:new_local_start]
                        v_past = kv_cache["v"][:, k_win_start:new_local_start]
                    k_win = torch.cat([k_past, k_to_cache], dim=1)
                    v_win = torch.cat([v_past, v], dim=1)

                query_offset_in_win = new_local_start - k_win_start

            # 6. Apply RoPE (mode-dependent)
            if not self.use_dynamic_rope:
                # Original mode: Q already roped, k_win contains roped keys
                roped_query = roped_q
                roped_key = k_win
            else:
                # Dynamic mode: apply window-local RoPE
                F_window = k_win.shape[1] // frame_seqlen
                k_grid = grid_sizes.clone()
                k_grid[:, 0] = F_window
                roped_key = causal_rope_apply(
                    k_win, k_grid, freqs, start_frame=0
                ).type_as(v)

                q_frame_start = query_offset_in_win // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=q_frame_start
                ).type_as(v)

            # 7. Attention
            x = _flash_attention(roped_query, roped_key, v_win)

            # 8. Update metadata (only if store_kv)
            if store_kv:
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(new_local_end)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention (T2V style — same as bidirectional but with optional cache)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """T2V cross-attention with optional KV caching for text conditioning."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        crossattn_cache: Optional[Dict] = None,
    ) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = _flash_attention(q, k, v)
        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Causal DiT Block
# ---------------------------------------------------------------------------

class CausalDiTBlock(nn.Module):
    """Transformer block for the causal model.

    Structurally matches ``CausalWanAttentionBlock`` from the reference:
    norm1 -> CausalSelfAttention -> norm3 -> CrossAttention -> norm2 -> FFN
    with per-frame AdaLN modulation.

    IMPORTANT: norm ordering matches the original Wan convention:
        norm1 = self-attention, norm3 = cross-attention (learnable), norm2 = FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Self-attention (causal)
        self.self_attn = CausalSelfAttention(
            dim, num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            use_dynamic_rope=use_dynamic_rope,
            eps=eps,
        )

        # Norms: norm1=self-attn, norm3=cross-attn (learnable), norm2=FFN
        self.norm1 = LayerNorm(dim, eps)
        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = LayerNorm(dim, eps)

        # Cross-attention
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # AdaLN modulation (6 modulation vectors)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        block_mask=None,
        kv_cache: Optional[Dict] = None,
        crossattn_cache: Optional[Dict] = None,
        current_start: int = 0,
        store_kv: bool = True,
        cache_local_end_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            e: [B, F, 6, C]  — per-frame timestep modulation
            seq_lens: [B]
            grid_sizes: [B, 3]
            freqs: RoPE freq tables
            context: [B, L_ctx, C]
            context_lens: [B] or None
            block_mask: FlexAttention mask
            kv_cache: self-attention KV cache dict
            crossattn_cache: cross-attention KV cache dict
            current_start: token offset for causal RoPE
            store_kv: whether to update KV cache (passed to self_attn)
            cache_local_end_override: if set, use this as local_end instead of
                reading from cache (for gradient checkpointing determinism)
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames

        # AdaLN modulation — 6 vectors per frame
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # --- Self-attention ---
        norm_x = self.norm1(x).unflatten(1, (num_frames, frame_seqlen))
        norm_x = (norm_x * (1 + e_mod[1]) + e_mod[0]).flatten(1, 2)

        y = self.self_attn(
            norm_x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            store_kv=store_kv,
            cache_local_end_override=cache_local_end_override,
        )

        x = x + (
            y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[2]
        ).flatten(1, 2)

        # --- Cross-attention ---
        x = x + self.cross_attn(
            self.norm3(x), context, context_lens, crossattn_cache
        )

        # --- FFN ---
        norm_x = self.norm2(x).unflatten(1, (num_frames, frame_seqlen))
        ff_out = self.ffn((norm_x * (1 + e_mod[4]) + e_mod[3]).flatten(1, 2))
        x = x + (
            ff_out.unflatten(1, (num_frames, frame_seqlen)) * e_mod[5]
        ).flatten(1, 2)

        return x


# ---------------------------------------------------------------------------
# Causal Head
# ---------------------------------------------------------------------------

class CausalHead(nn.Module):
    """Output head with per-frame modulation (matches reference CausalHead)."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_channels = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            e: [B, F, 1, C]
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e_mod[1]) + e_mod[0]
        )
        return x


# ---------------------------------------------------------------------------
# CausalOmniAvatarWan: the student network
# ---------------------------------------------------------------------------

class CausalOmniAvatarWan(CausalFastGenNetwork):
    """Causal OmniAvatar DiT for use as the student in Self-Forcing distillation.

    This class implements the full causal model including:
      - CausalSelfAttention with KV cache + FlexAttention
      - Causal 3D RoPE with frame offset
      - AudioPack processing with per-chunk slicing
      - Per-frame AdaLN modulation
      - V2V conditioning (same as bidirectional wrapper)

    Two forward modes:
      - ``is_ar=True``:  chunk-based autoregressive generation with KV cache
      - ``is_ar=False``: full-sequence forward (like bidirectional, for verification)
    """

    def __init__(
        self,
        model_size: str = "1.3B",
        in_dim: int = 49,
        mode: str = "v2v",
        use_audio: bool = True,
        audio_hidden_size: int = 32,
        chunk_size: int = 3,
        total_num_frames: int = 21,
        base_model_paths: Optional[str] = None,
        omniavatar_ckpt_path: Optional[str] = None,
        merge_lora: bool = True,
        lora_rank: int = 128,
        lora_alpha: int = 64,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        mask_all_frames: bool = True,
        dtype: str = "bf16",
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_size: ``"14B"`` or ``"1.3B"``.
            in_dim: Input channels to patch embedding (49 for V2V, 65 for V2V+refseq).
            mode: ``"i2v"`` or ``"v2v"``.
            use_audio: Whether to include audio conditioning.
            audio_hidden_size: Hidden dim of AudioPack (default 32).
            chunk_size: Number of latent frames per autoregressive chunk.
            total_num_frames: Total latent frames in the full video (e.g. 21).
            base_model_paths: Comma-separated safetensor paths for base Wan 2.1 T2V.
            omniavatar_ckpt_path: Path to OmniAvatar checkpoint (.pt/.pth).
            merge_lora: If True, merge LoRA into base model in-place.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            net_pred_type: Network prediction type.
            schedule_type: Noise schedule type.
            mask_all_frames: Whether to apply spatial mask to all frames.
            dtype: Default dtype string.
            **kwargs: Additional kwargs passed to CausalFastGenNetwork.
        """
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **kwargs,
        )

        if model_size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Choose from {list(MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.in_dim = in_dim
        self.mode = mode
        self.use_audio = use_audio
        self.audio_hidden_size = audio_hidden_size
        self.merge_lora = merge_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.mask_all_frames = mask_all_frames
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_dynamic_rope = use_dynamic_rope

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._default_dtype = dtype_map.get(dtype, torch.bfloat16)

        cfg = MODEL_CONFIGS[model_size]
        dim = cfg["dim"]
        ffn_dim = cfg["ffn_dim"]
        num_heads = cfg["num_heads"]
        num_layers = cfg["num_layers"]
        out_dim = _COMMON_CFG["out_dim"]
        text_dim = _COMMON_CFG["text_dim"]
        freq_dim = _COMMON_CFG["freq_dim"]
        eps = _COMMON_CFG["eps"]
        patch_size = _COMMON_CFG["patch_size"]

        self._dim = dim
        self._freq_dim = freq_dim
        self._text_len = 512
        self._patch_size = patch_size
        self._out_dim = out_dim
        self._num_layers = num_layers
        self._num_heads = num_heads

        # --- Embeddings ---
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            CausalDiTBlock(
                dim, num_heads, ffn_dim, eps,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                use_dynamic_rope=use_dynamic_rope,
            )
            for _ in range(num_layers)
        ])

        # --- Output head ---
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # --- RoPE frequencies ---
        head_dim = dim // num_heads
        self.freqs = _precompute_freqs_cis_3d(head_dim)

        # --- Audio ---
        if self.use_audio:
            audio_input_dim = 10752  # OmniAvatar constant
            self.audio_proj = AudioPack(
                audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True
            )
            self.audio_cond_projs = nn.ModuleList()
            for _ in range(num_layers // 2 - 1):
                self.audio_cond_projs.append(nn.Linear(audio_hidden_size, dim))

        # --- FlexAttention block mask (lazily constructed) ---
        self.block_mask = None

        # --- KV caches (lazily allocated) ---
        self._kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None
        self._crossattn_caches: Optional[List[Dict]] = None

        # --- Load weights ---
        if not self._is_in_meta_context():
            self._load_weights(base_model_paths, omniavatar_ckpt_path)
            self.to(self._default_dtype)

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def _unpatchify(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Reconstruct video tensors from patchified features.

        Args:
            x: [B, F*H*W, out_dim * prod(patch_size)]  (batched, may be nested in F dim)
            grid_sizes: [B, 3]

        Returns:
            List of tensors with shape [C_out, F*p_t, H*p_h, W*p_w]
        """
        c = self._out_dim
        p = self._patch_size
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *p, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, p)])
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # Audio processing (IDENTICAL to bidirectional model)
    # ------------------------------------------------------------------

    def _process_audio_embeddings(
        self,
        audio_emb: Optional[torch.Tensor],
        x_shape: torch.Size,
    ) -> Optional[torch.Tensor]:
        """Process raw audio embeddings through AudioPack + per-layer projections.

        Args:
            audio_emb: [B, num_video_frames, 10752]
            x_shape: shape of x for batch-size extraction

        Returns:
            [B, num_layers//2-1, num_audio_frames, 1, 1, dim] or None
        """
        if audio_emb is None or not self.use_audio:
            return None

        # Step 1: Permute and add spatial dims  [B, 10752, T, 1, 1]
        audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]

        # Step 2: Pad temporal dimension (prepend 3 copies of first frame)
        audio_emb = torch.cat(
            [audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], dim=2
        )

        # Step 3: Apply AudioPack  [B, T', 1, 1, hidden_size]
        audio_emb = self.audio_proj(audio_emb)

        # Step 4: Apply per-layer linear projections and stack
        # Each proj: [B, T', 1, 1, hidden_size] -> [B, T', 1, 1, dim]
        # Concat along batch -> [B * num_projs, T', 1, 1, dim]
        audio_emb = torch.cat(
            [proj(audio_emb) for proj in self.audio_cond_projs], dim=0
        )

        # Step 5: Reshape for per-layer injection
        # [B * num_projs, T', 1, 1, dim] -> [B, num_projs, T', 1, 1, dim]
        audio_emb = audio_emb.reshape(
            x_shape[0],
            audio_emb.shape[0] // x_shape[0],
            -1,
            *audio_emb.shape[2:],
        )

        return audio_emb

    def _inject_audio_at_layer(
        self,
        x: torch.Tensor,
        audio_emb: Optional[torch.Tensor],
        block_index: int,
        num_blocks: int,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Inject audio conditioning at a specific transformer layer.

        Audio is injected at blocks with index in (1, num_blocks//2],
        i.e., block_index > 1 and block_index <= num_blocks // 2.
        """
        if not self.use_audio or audio_emb is None:
            return x
        if block_index <= 1 or block_index > num_blocks // 2:
            return x

        au_idx = block_index - 2
        f, h, w = (
            grid_sizes[0][0].item(),
            grid_sizes[0][1].item(),
            grid_sizes[0][2].item(),
        )
        lat_h, lat_w = h * 2, w * 2

        # Repeat audio spatially: [B, au_frames, lat_h//2, lat_w//2, dim]
        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1)

        # Patchify: [B, dim, au_frames, lat_h//2, lat_w//2] -> [B, (au_frames*h*w), dim]
        audio_cond = audio_emb_tmp.permute(0, 4, 1, 2, 3)
        B, dim_a, au_f, au_h, au_w = audio_cond.shape
        audio_cond = audio_cond.reshape(B, dim_a, -1).transpose(1, 2)

        x = audio_cond + x
        return x

    # ------------------------------------------------------------------
    # V2V conditioning (same as bidirectional)
    # ------------------------------------------------------------------

    def _build_y(
        self,
        condition: Dict[str, torch.Tensor],
        T: int,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """Build the V2V conditioning tensor ``y``, optionally sliced for a chunk.

        Args:
            condition: Dict with ref_latent, mask, masked_video, ref_sequence.
            T: Number of latent frames for this call (full sequence or chunk).
            start_frame: Starting latent frame index (0 for full sequence).

        Returns:
            y: [B, C_y, T, H, W] where C_y = in_dim - 16
        """
        ref_latent = condition["ref_latent"]  # [B, 16, 1, H, W]
        mask = condition["mask"]  # [H_lat, W_lat]
        masked_video = condition.get("masked_video")
        ref_sequence = condition.get("ref_sequence")

        B = ref_latent.shape[0]
        H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
        device = ref_latent.device
        dtype = ref_latent.dtype

        ref_repeated = ref_latent.repeat(1, 1, T, 1, 1)

        inverted_mask = 1.0 - mask.to(dtype=dtype)
        mask_ch = torch.zeros(B, 1, T, H_lat, W_lat, device=device, dtype=dtype)
        if self.mask_all_frames:
            mask_ch[:, :, :] = inverted_mask[None, None, None]
        else:
            for i in range(T):
                abs_frame = start_frame + i
                if abs_frame == 0:
                    mask_ch[:, :, i] = 0  # Frame 0: reference, no mask
                else:
                    mask_ch[:, :, i] = inverted_mask[None, None, None]

        parts = [ref_repeated, mask_ch]
        if masked_video is not None:
            # Slice to current chunk's temporal range
            parts.append(masked_video[:, :, start_frame:start_frame + T])
        if ref_sequence is not None:
            parts.append(ref_sequence[:, :, start_frame:start_frame + T])

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # FlexAttention block mask
    # ------------------------------------------------------------------

    def _build_block_mask(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
    ) -> Optional[BlockMask]:
        """Build a blockwise causal attention mask for full-sequence mode.

        Each frame's tokens can attend to all tokens in current and previous frames.
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        total_length = num_frames * frame_seqlen
        pad_len = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + pad_len, device=device, dtype=torch.long)

        # Fill: each frame block's tokens can attend up to end of their frame
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = (frame_idx + 1) * frame_seqlen
            ends[start:end] = end

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + pad_len,
            KV_LEN=total_length + pad_len,
            _compile=False,
            device=device,
        )
        return block_mask

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _init_caches(
        self,
        batch_size: int,
        total_tokens: int,
        frame_seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate KV caches for all transformer blocks.

        When ``local_attn_size > 0``, the cache is sized to hold exactly
        ``local_attn_size * frame_seqlen`` tokens (the rolling window).
        Otherwise it is sized for the full sequence (``total_tokens``).
        """
        n = self._num_heads
        d = self._dim // n

        if self.local_attn_size > 0:
            cache_tokens = self.local_attn_size * frame_seqlen
        else:
            cache_tokens = total_tokens

        self._kv_caches = []
        self._crossattn_caches = []
        for _ in self.blocks:
            self._kv_caches.append(
                {
                    "k": torch.zeros(batch_size, cache_tokens, n, d, device=device, dtype=dtype),
                    "v": torch.zeros(batch_size, cache_tokens, n, d, device=device, dtype=dtype),
                    "global_end_index": torch.tensor(0, device=device, dtype=torch.long),
                    "local_end_index": torch.tensor(0, device=device, dtype=torch.long),
                }
            )
            self._crossattn_caches.append({"is_init": False})

    def clear_caches(self) -> None:
        """Clear all KV caches in all transformer blocks."""
        if self._kv_caches is not None:
            for cache in self._kv_caches:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["global_end_index"].fill_(0)
                cache["local_end_index"].fill_(0)
        if self._crossattn_caches is not None:
            for cache in self._crossattn_caches:
                cache["is_init"] = False
                cache.pop("k", None)
                cache.pop("v", None)
        self._kv_caches = None
        self._crossattn_caches = None
        self.block_mask = None

    # ------------------------------------------------------------------
    # Internal forward (full-sequence mode)
    # ------------------------------------------------------------------

    def _forward_full_sequence(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Full-sequence forward — like bidirectional but with causal block mask.

        This mode processes ALL frames at once. Used for verification against
        the bidirectional model and for teacher-forcing training.

        Args:
            x: [B, 16, T, H, W]  noisy latent
            timestep: [B]  timestep values (already rescaled)
            context: [B, 512, 4096]  text embeddings
            y: [B, C_y, T, H, W]  V2V conditioning
            audio_emb: [B, num_video_frames, 10752]  or None
            use_gradient_checkpointing: enable gradient checkpointing

        Returns:
            [B, 16, T, H, W]  model output
        """
        device = self.patch_embedding.weight.device
        if self.freqs[0].device != device:
            self.freqs = tuple(f.to(device) for f in self.freqs)

        # Concatenate conditioning
        if y is not None:
            x = torch.cat([x, y], dim=1)  # [B, in_dim, T, H, W]

        lat_h, lat_w = x.shape[-2], x.shape[-1]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, dim, f, h, w]
        grid_sizes = torch.tensor(
            [list(x.shape[2:])], dtype=torch.long, device=device
        ).expand(x.shape[0], -1)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, f*h*w, dim]
        seq_lens = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long, device=device)

        # Time embedding
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self._freq_dim, timestep).type_as(x)
        )
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self._dim))
        # Expand to per-frame: [B, 6, dim] -> [B, F, 6, dim]
        t_mod = t_mod.unsqueeze(1).expand(-1, f, -1, -1)

        # Text embedding
        context = self.text_embedding(context)

        # Audio processing
        processed_audio = self._process_audio_embeddings(audio_emb, x.shape)

        # Build block mask (causal, per-frame)
        if self.block_mask is None and FLEX_ATTENTION_AVAILABLE:
            frame_seqlen = h * w
            self.block_mask = self._build_block_mask(device, f, frame_seqlen)

        # Create custom forward for gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # Transformer blocks
        for block_index, block in enumerate(self.blocks):
            # Audio injection
            x = self._inject_audio_at_layer(
                x, processed_audio, block_index, len(self.blocks), grid_sizes
            )

            kwargs = dict(
                e=t_mod,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=None,
                block_mask=self.block_mask,
            )

            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Head: t_emb is [B, dim], need [B, F, 1, dim]
        head_e = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, f, -1, -1)
        x = self.head(x, head_e)

        # Unpatchify
        out = self._unpatchify(x, grid_sizes)
        return torch.stack(out)

    # ------------------------------------------------------------------
    # Internal forward (AR / chunk-based mode)
    # ------------------------------------------------------------------

    def _forward_ar(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        current_start: int = 0,
        store_kv: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Chunk-based autoregressive forward with KV cache.

        Processes a chunk of frames (e.g. 3 frames) while attending to all
        previously cached frames via the KV cache.

        Args:
            x: [B, 16, chunk_frames, H, W]  noisy latent chunk
            timestep: [B]
            context: [B, 512, 4096]  text embeddings (full — NOT sliced)
            y: [B, C_y, chunk_frames, H, W]  V2V conditioning for this chunk
            audio_emb: [B, num_video_frames, 10752]  FULL audio (not sliced)
            current_start: token offset (= frame_offset * h * w)
            store_kv: whether to write to KV cache and update metadata
            use_gradient_checkpointing: enable gradient checkpointing

        Returns:
            [B, 16, chunk_frames, H, W]  model output for this chunk
        """
        device = self.patch_embedding.weight.device
        if self.freqs[0].device != device:
            self.freqs = tuple(f.to(device) for f in self.freqs)

        # Concatenate conditioning
        if y is not None:
            x = torch.cat([x, y], dim=1)

        lat_h, lat_w = x.shape[-2], x.shape[-1]

        # Patch embedding
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(
            [list(x.shape[2:])], dtype=torch.long, device=device
        ).expand(x.shape[0], -1)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long, device=device)

        # Allocate caches on first call
        frame_seqlen = h * w
        total_tokens = self.total_num_frames * frame_seqlen
        if self._kv_caches is None:
            self._init_caches(x.shape[0], total_tokens, frame_seqlen, device, x.dtype)

        # Time embedding
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self._freq_dim, timestep).type_as(x)
        )
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self._dim))
        t_mod = t_mod.unsqueeze(1).expand(-1, f, -1, -1)

        # Text embedding
        context = self.text_embedding(context)

        # Audio processing (full sequence first, then slice)
        processed_audio = self._process_audio_embeddings(audio_emb, x.shape)
        if processed_audio is not None:
            current_frame_start = current_start // frame_seqlen
            current_video_frames = grid_sizes[0][0].item()
            current_frame_end = current_frame_start + current_video_frames
            processed_audio = processed_audio[
                :, :, current_frame_start:current_frame_end, :, :, :
            ]

        # Transformer blocks
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            # Audio injection
            x = self._inject_audio_at_layer(
                x, processed_audio, block_index, len(self.blocks), grid_sizes
            )

            # Compute cache read position from current_start (immutable),
            # following FastGen's pattern (proper_cache_len = cur_start_frame * frame_seqlen).
            # This is deterministic and gradient-checkpointing safe.
            kv_cache_i = self._kv_caches[block_index]
            if self.local_attn_size <= 0:
                cache_local_end = current_start  # no eviction: local == global
            else:
                cache_local_end = kv_cache_i["local_end_index"].item()

            kwargs = dict(
                e=t_mod,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=None,
                block_mask=None,  # No FlexAttention in AR mode
                kv_cache=kv_cache_i,
                crossattn_cache=self._crossattn_caches[block_index],
                current_start=current_start,
                store_kv=store_kv,
                cache_local_end_override=cache_local_end,
            )

            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Head
        head_e = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, f, -1, -1)
        x = self.head(x, head_e)

        # Unpatchify
        out = self._unpatchify(x, grid_sizes)
        return torch.stack(out)

    # ------------------------------------------------------------------
    # Forward (FastGen CausalFastGenNetwork interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = True,
        use_gradient_checkpointing: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass — dispatches to full-sequence or AR mode.

        Args:
            x_t: Noisy latent [B, 16, T, H, W].
            t: Timestep in [0, 1) range, shape [B].
            condition: Dict with text_embeds, audio_emb, ref_latent, mask, etc.
            cur_start_frame: Frame offset for AR generation.
            store_kv: Whether to update KV cache (always True in AR mode).
            is_ar: If True, use AR mode with KV cache; if False, full-sequence.
            use_gradient_checkpointing: Enable gradient checkpointing.
            **fwd_kwargs: Additional kwargs.

        Returns:
            Model output tensor.
        """
        if feature_indices is None:
            feature_indices = set()

        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        elif fwd_pred_type not in NET_PRED_TYPES:
            raise ValueError(
                f"Unsupported fwd_pred_type '{fwd_pred_type}'. Supported: {NET_PRED_TYPES}"
            )

        # Unpack condition
        assert isinstance(condition, dict), f"condition must be a dict, got {type(condition)}"
        text_embeds = condition["text_embeds"]
        audio_emb = condition.get("audio_emb")

        # Build V2V y-tensor (sliced for chunk in AR mode)
        T = x_t.shape[2]
        y = self._build_y(condition, T, start_frame=cur_start_frame)

        # Rescale timestep
        timestep = self.noise_scheduler.rescale_t(t)

        # Compute frame/token offset
        # frame_seqlen will be computed inside forward methods from patch embedding output
        # cur_start_frame is in latent frame units
        # We need to convert to token offset for the internal model
        # Token offset = cur_start_frame * h * w where h, w are post-patchification spatial dims
        # h = H_lat / p_h, w = W_lat / p_w
        H_lat, W_lat = x_t.shape[-2], x_t.shape[-1]
        p_h, p_w = self._patch_size[1], self._patch_size[2]
        h_patches, w_patches = H_lat // p_h, W_lat // p_w
        current_start = cur_start_frame * h_patches * w_patches

        if is_ar:
            model_output = self._forward_ar(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                y=y,
                audio_emb=audio_emb,
                current_start=current_start,
                store_kv=store_kv,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            model_output = self._forward_full_sequence(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                y=y,
                audio_emb=audio_emb,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        # Convert prediction type
        out = self.noise_scheduler.convert_model_output(
            x_t, model_output, t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # Feature extraction — return expected tuple structure for DMD2 compatibility
        if return_features_early:
            return []

        if feature_indices is not None and len(feature_indices) > 0:
            if return_logvar:
                logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
                return [out, []], logvar
            return [out, []]

        if return_logvar:
            logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
            return out, logvar

        return out

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(
        self,
        base_model_paths: Optional[str],
        omniavatar_ckpt_path: Optional[str],
    ) -> None:
        """Multi-stage weight loading (same logic as bidirectional wrapper).

        Stage 1: Load base Wan 2.1 T2V safetensor weights, converting
                 diffusers keys to our internal key naming convention.
        Stage 2: Load OmniAvatar checkpoint (LoRA + audio + patch_embedding),
                 handling patch_embedding expansion and LoRA merging.

        The causal model's internal naming matches the bidirectional WanModel's
        naming because both use the same parameter names:
            patch_embedding, text_embedding, time_embedding, time_projection
            blocks.{i}.self_attn.{q,k,v,o}, blocks.{i}.self_attn.norm_{q,k}
            blocks.{i}.cross_attn.{q,k,v,o}, blocks.{i}.cross_attn.norm_{q,k}
            blocks.{i}.norm1, blocks.{i}.norm2, blocks.{i}.norm3
            blocks.{i}.ffn.{0,2}, blocks.{i}.modulation
            head.norm, head.head, head.modulation
            audio_proj.*, audio_cond_projs.*
        """
        # --- Stage 1: Base Wan 2.1 weights ---
        if base_model_paths is not None:
            paths = [p.strip() for p in base_model_paths.split(",") if p.strip()]
            logger.info(
                f"[CausalOmniAvatarWan] Loading base Wan 2.1 from {len(paths)} file(s)"
            )

            base_sd: Dict[str, torch.Tensor] = {}
            for p in paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Base model weight file not found: {p}")
                base_sd.update(_load_state_dict(p, dtype=self._default_dtype))

            # Detect and convert diffusers format
            sample_key = next(iter(base_sd.keys()), "")
            is_diffusers = any(
                marker in sample_key
                for marker in ("condition_embedder", "attn1", "attn2", "ffn.net")
            )

            if is_diffusers:
                logger.info("[CausalOmniAvatarWan] Detected diffusers format, converting keys")
                base_sd = _convert_diffusers_state_dict(base_sd)
            else:
                cleaned = {}
                for k, v in base_sd.items():
                    clean_k = k
                    for prefix in ("model.", "module.", "transformer."):
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                    cleaned[clean_k] = v
                base_sd = cleaned

            missing, unexpected = _smart_load_weights(self, base_sd)
            loaded = len(base_sd) - len(unexpected)
            logger.info(
                f"[CausalOmniAvatarWan] Base weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
            if missing:
                audio_missing = [k for k in missing if "audio" in k]
                other_missing = [k for k in missing if "audio" not in k]
                if audio_missing:
                    logger.info(
                        f"[CausalOmniAvatarWan] Expected missing (audio): {len(audio_missing)} keys"
                    )
                if other_missing:
                    logger.warning(
                        f"[CausalOmniAvatarWan] Unexpected missing keys: {other_missing[:10]}..."
                    )
        else:
            logger.info("[CausalOmniAvatarWan] No base_model_paths, using random init")

        # --- Stage 2: OmniAvatar checkpoint ---
        if omniavatar_ckpt_path is not None:
            if not os.path.isfile(omniavatar_ckpt_path):
                raise FileNotFoundError(
                    f"OmniAvatar checkpoint not found: {omniavatar_ckpt_path}"
                )

            logger.info(
                f"[CausalOmniAvatarWan] Loading OmniAvatar checkpoint: {omniavatar_ckpt_path}"
            )
            ckpt_sd = _load_state_dict(omniavatar_ckpt_path, dtype=self._default_dtype)

            # Separate LoRA from non-LoRA
            lora_sd: Dict[str, torch.Tensor] = {}
            non_lora_sd: Dict[str, torch.Tensor] = {}
            for k, v in ckpt_sd.items():
                if "lora_A" in k or "lora_B" in k:
                    lora_sd[k] = v
                else:
                    non_lora_sd[k] = v

            logger.info(
                f"[CausalOmniAvatarWan] Checkpoint: {len(lora_sd)} LoRA params, "
                f"{len(non_lora_sd)} non-LoRA params"
            )

            # Handle patch_embedding expansion
            pe_key = "patch_embedding.weight"
            if pe_key in non_lora_sd:
                model_pe = self.patch_embedding.weight
                if non_lora_sd[pe_key].shape != model_pe.shape:
                    old_in_ch = non_lora_sd[pe_key].shape[1]
                    new_in_ch = model_pe.shape[1]
                    logger.info(
                        f"[CausalOmniAvatarWan] Expanding patch_embedding: "
                        f"{list(non_lora_sd[pe_key].shape)} -> {list(model_pe.shape)}"
                    )
                    new_pe = torch.zeros_like(model_pe.data)
                    slices = tuple(slice(0, s) for s in non_lora_sd[pe_key].shape)
                    new_pe[slices] = non_lora_sd[pe_key]
                    non_lora_sd[pe_key] = new_pe

            pe_bias_key = "patch_embedding.bias"
            if pe_bias_key in non_lora_sd and self.patch_embedding.bias is not None:
                model_pe_bias = self.patch_embedding.bias
                if non_lora_sd[pe_bias_key].shape != model_pe_bias.shape:
                    new_bias = model_pe_bias.data.clone()
                    new_bias[: non_lora_sd[pe_bias_key].shape[0]] = non_lora_sd[pe_bias_key]
                    non_lora_sd[pe_bias_key] = new_bias

            # Load non-LoRA weights
            if non_lora_sd:
                missing, unexpected = self.load_state_dict(non_lora_sd, strict=False)
                loaded = len(non_lora_sd) - len(unexpected)
                logger.info(
                    f"[CausalOmniAvatarWan] Non-LoRA weights: {loaded} loaded, "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )

            # Handle LoRA weights
            if lora_sd:
                if self.merge_lora:
                    logger.info("[CausalOmniAvatarWan] Merging LoRA weights into base model")
                    _merge_lora_into_model(
                        self,
                        lora_sd,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        target_modules=LORA_TARGET_MODULES,
                    )
                else:
                    logger.info(
                        "[CausalOmniAvatarWan] Loading LoRA as PEFT adapters (not merged)"
                    )
                    mapped_lora_sd = _map_omniavatar_lora_keys(lora_sd, use_peft=True)
                    try:
                        from peft import LoraConfig, inject_adapter_in_model

                        lora_config = LoraConfig(
                            r=self.lora_rank,
                            lora_alpha=self.lora_alpha,
                            init_lora_weights=True,
                            target_modules=LORA_TARGET_MODULES,
                        )
                        # Note: inject_adapter_in_model modifies self in-place
                        inject_adapter_in_model(lora_config, self)
                        missing, unexpected = self.load_state_dict(
                            mapped_lora_sd, strict=False
                        )
                        logger.info(
                            f"[CausalOmniAvatarWan] PEFT LoRA: "
                            f"{len(mapped_lora_sd) - len(unexpected)} loaded"
                        )
                    except ImportError:
                        raise ImportError(
                            "merge_lora=False requires the `peft` package."
                        )
        else:
            logger.info(
                "[CausalOmniAvatarWan] No omniavatar_ckpt_path, using base/random init only"
            )
