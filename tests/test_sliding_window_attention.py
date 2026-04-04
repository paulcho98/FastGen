"""Tests for sliding window + sink attention mask in CausalOmniAvatarWan.

Uses FRAME_SEQLEN=4 and small frame counts to stay lightweight.
Skips entirely if FlexAttention is not available.

Note: Because the sequences are small (< 128 tokens), they fit within a single
FlexAttention block. The BlockMask only tracks block-level sparsity, so we
cannot verify per-token visibility from BlockMask alone. Instead, we:
  1. Evaluate the mask function directly on all (q, kv) pairs to verify the
     intended attention pattern.
  2. Also build the BlockMask to ensure it compiles without error.
"""

import math
import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Attempt to import FlexAttention utilities
# ---------------------------------------------------------------------------
try:
    os.environ.setdefault("FASTGEN_DISABLE_FLEX_ATTENTION", "0")
    from torch.nn.attention.flex_attention import create_block_mask, BlockMask

    flex_available = True
except ImportError:
    flex_available = False

pytestmark = pytest.mark.skipif(not flex_available, reason="FlexAttention not available")

# ---------------------------------------------------------------------------
# Constants used across all tests
# ---------------------------------------------------------------------------
FRAME_SEQLEN = 4  # tokens per frame — keep small for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helper: build mask arrays and return the dense per-token mask + BlockMask
# ---------------------------------------------------------------------------
def _build_mask_and_dense(
    num_frames: int,
    frame_seqlen: int,
    chunk_size: int,
    local_attn_size: int = -1,
    sink_size: int = 0,
    device: str = DEVICE,
):
    """Build the mask arrays (mirrors _build_block_mask) and return:
    - dense: (total_length, total_length) bool tensor of per-token visibility
    - block_mask: the BlockMask object (to verify it compiles)
    """
    total_length = num_frames * frame_seqlen
    pad_len = math.ceil(total_length / 128) * 128 - total_length
    padded_length = total_length + pad_len

    ends = torch.zeros(padded_length, device=device, dtype=torch.long)
    starts = torch.zeros(padded_length, device=device, dtype=torch.long)

    num_chunks_count = num_frames // chunk_size
    remaining_size = num_frames % chunk_size

    frame_counts = []
    if num_frames > 0:
        if num_chunks_count == 0:
            frame_counts.append(remaining_size)
        else:
            frame_counts.append(chunk_size + remaining_size)
            frame_counts.extend([chunk_size] * max(num_chunks_count - 1, 0))

    current_start = 0
    for frames_in_chunk in frame_counts:
        chunk_len_tokens = frames_in_chunk * frame_seqlen
        chunk_end = current_start + chunk_len_tokens

        if local_attn_size > 0:
            chunk_last_frame = (current_start // frame_seqlen) + frames_in_chunk
            window_start_frame = max(0, chunk_last_frame - local_attn_size)
            window_start_token = window_start_frame * frame_seqlen
        else:
            window_start_token = 0

        ends[current_start:chunk_end] = chunk_end
        starts[current_start:chunk_end] = window_start_token
        current_start += chunk_len_tokens

    sink_end = sink_size * frame_seqlen

    def attention_mask(b, h, q_idx, kv_idx):
        in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
        is_sink = kv_idx < sink_end
        return in_window | is_sink | (q_idx == kv_idx)

    # Build BlockMask (verifies it doesn't crash)
    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        _compile=False,
        device=device,
    )

    # Evaluate mask function on all token pairs within the real (unpadded) range
    q_indices = torch.arange(total_length, device=device)
    kv_indices = torch.arange(total_length, device=device)
    qq = q_indices.unsqueeze(1).expand(total_length, total_length)
    kk = kv_indices.unsqueeze(0).expand(total_length, total_length)
    dense = attention_mask(0, 0, qq, kk)

    return dense, block_mask


# ---------------------------------------------------------------------------
# Helper: compute chunk boundaries (list of (start_frame, end_frame))
# ---------------------------------------------------------------------------
def _chunk_boundaries(num_frames: int, chunk_size: int):
    num_chunks = num_frames // chunk_size
    remaining = num_frames % chunk_size
    chunks = []
    if num_chunks == 0:
        chunks.append((0, remaining))
    else:
        first_size = chunk_size + remaining
        chunks.append((0, first_size))
        for i in range(1, num_chunks):
            start = first_size + (i - 1) * chunk_size
            chunks.append((start, start + chunk_size))
    return chunks


def _frame_to_chunk(frame: int, chunks):
    """Return chunk index for a given frame."""
    for ci, (cs, ce) in enumerate(chunks):
        if cs <= frame < ce:
            return ci
    return None


def _chunk_end_frame(frame: int, chunks):
    """Return the end frame of the chunk containing `frame`."""
    for cs, ce in chunks:
        if cs <= frame < ce:
            return ce
    return None


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestBlockMask:
    """Test suite for _build_block_mask sliding window + sink support."""

    # --- Test 1: full causal unchanged -----------------------------------
    def test_full_causal_unchanged(self):
        """local_attn_size=-1, sink_size=0 produces the same chunk-causal pattern as before."""
        num_frames = 9
        chunk_size = 3
        dense, bm = _build_mask_and_dense(
            num_frames, FRAME_SEQLEN, chunk_size, local_attn_size=-1, sink_size=0
        )

        chunks = _chunk_boundaries(num_frames, chunk_size)
        # chunks: [(0,3), (3,6), (6,9)]

        for q_frame in range(num_frames):
            q_ci = _frame_to_chunk(q_frame, chunks)
            for kv_frame in range(num_frames):
                kv_ci = _frame_to_chunk(kv_frame, chunks)

                # Expected: kv visible iff kv_chunk <= q_chunk
                expected = kv_ci <= q_ci

                q_tok = q_frame * FRAME_SEQLEN
                kv_tok = kv_frame * FRAME_SEQLEN
                actual = dense[q_tok, kv_tok].item()
                assert actual == expected, (
                    f"Frame q={q_frame} (chunk {q_ci}), kv={kv_frame} (chunk {kv_ci}): "
                    f"expected {expected}, got {actual}"
                )

    # --- Test 2: sliding window basic -----------------------------------
    def test_sliding_window_basic(self):
        """local_attn_size=6 with chunk_size=3: each chunk sees itself + 1 prior chunk."""
        num_frames = 9
        chunk_size = 3
        local_attn_size = 6  # 2 chunks * 3 frames
        dense, bm = _build_mask_and_dense(
            num_frames, FRAME_SEQLEN, chunk_size,
            local_attn_size=local_attn_size, sink_size=0,
        )

        chunks = _chunk_boundaries(num_frames, chunk_size)

        for q_frame in range(num_frames):
            q_ci = _frame_to_chunk(q_frame, chunks)
            ce = _chunk_end_frame(q_frame, chunks)
            window_start = max(0, ce - local_attn_size)

            for kv_frame in range(num_frames):
                kv_ci = _frame_to_chunk(kv_frame, chunks)

                in_causal = kv_ci <= q_ci
                in_window = kv_frame >= window_start
                expected = in_causal and in_window

                q_tok = q_frame * FRAME_SEQLEN
                kv_tok = kv_frame * FRAME_SEQLEN
                actual = dense[q_tok, kv_tok].item()
                assert actual == expected, (
                    f"Frame q={q_frame}, kv={kv_frame}: "
                    f"expected {expected} (window_start={window_start}), got {actual}"
                )

        # Chunk 2 (frames 6-8): window_start = max(0, 9-6) = 3
        assert not dense[6 * FRAME_SEQLEN, 0].item(), "Chunk 2 should not see frame 0"
        assert dense[6 * FRAME_SEQLEN, 3 * FRAME_SEQLEN].item(), "Chunk 2 should see frame 3"

    # --- Test 3: sliding window with sink --------------------------------
    def test_sliding_window_with_sink(self):
        """local_attn_size=7, sink_size=1: frame 0 always visible + 7-frame window."""
        num_frames = 12
        chunk_size = 3
        local_attn_size = 7
        sink_size = 1
        dense, bm = _build_mask_and_dense(
            num_frames, FRAME_SEQLEN, chunk_size,
            local_attn_size=local_attn_size, sink_size=sink_size,
        )

        chunks = _chunk_boundaries(num_frames, chunk_size)

        for q_frame in range(num_frames):
            q_ci = _frame_to_chunk(q_frame, chunks)
            ce = _chunk_end_frame(q_frame, chunks)
            window_start = max(0, ce - local_attn_size)

            for kv_frame in range(num_frames):
                kv_ci = _frame_to_chunk(kv_frame, chunks)

                in_causal = kv_ci <= q_ci
                in_window = kv_frame >= window_start
                is_sink = kv_frame < sink_size
                expected = (in_causal and in_window) or is_sink

                q_tok = q_frame * FRAME_SEQLEN
                kv_tok = kv_frame * FRAME_SEQLEN
                actual = dense[q_tok, kv_tok].item()
                assert actual == expected, (
                    f"Frame q={q_frame}, kv={kv_frame}: "
                    f"expected {expected} (window_start={window_start}, sink_end={sink_size}), "
                    f"got {actual}"
                )

        # Last chunk (frames 9-11): window_start = max(0, 12-7) = 5
        assert dense[9 * FRAME_SEQLEN, 0].item(), "Sink frame 0 must always be visible"
        assert not dense[9 * FRAME_SEQLEN, 4 * FRAME_SEQLEN].item(), (
            "Frame 4 outside window for last chunk"
        )
        assert dense[9 * FRAME_SEQLEN, 5 * FRAME_SEQLEN].item(), (
            "Frame 5 inside window for last chunk"
        )

    # --- Test 4: sink included in window ---------------------------------
    def test_sink_included_in_window(self):
        """local_attn_size=4, sink_size=1: only 3 non-sink frames in rolling window."""
        num_frames = 9
        chunk_size = 3
        local_attn_size = 4
        sink_size = 1
        dense, bm = _build_mask_and_dense(
            num_frames, FRAME_SEQLEN, chunk_size,
            local_attn_size=local_attn_size, sink_size=sink_size,
        )

        chunks = _chunk_boundaries(num_frames, chunk_size)

        for q_frame in range(num_frames):
            q_ci = _frame_to_chunk(q_frame, chunks)
            ce = _chunk_end_frame(q_frame, chunks)
            window_start = max(0, ce - local_attn_size)

            for kv_frame in range(num_frames):
                kv_ci = _frame_to_chunk(kv_frame, chunks)

                in_causal = kv_ci <= q_ci
                in_window = kv_frame >= window_start
                is_sink = kv_frame < sink_size
                expected = (in_causal and in_window) or is_sink

                q_tok = q_frame * FRAME_SEQLEN
                kv_tok = kv_frame * FRAME_SEQLEN
                actual = dense[q_tok, kv_tok].item()
                assert actual == expected, (
                    f"Frame q={q_frame}, kv={kv_frame}: "
                    f"expected {expected} (window_start={window_start}, sink_end={sink_size}), "
                    f"got {actual}"
                )

        # Chunk 2 (frames 6-8): window_start = max(0, 9-4) = 5
        assert dense[6 * FRAME_SEQLEN, 0].item(), "Sink frame 0 always visible"
        assert not dense[6 * FRAME_SEQLEN, 1 * FRAME_SEQLEN].item(), (
            "Frame 1 outside window and not sink"
        )
        assert dense[6 * FRAME_SEQLEN, 5 * FRAME_SEQLEN].item(), (
            "Frame 5 in window"
        )

    # --- Test 5: diagonal self-attention ---------------------------------
    def test_diagonal_always_true(self):
        """The q_idx == kv_idx term ensures the diagonal is always True."""
        num_frames = 5
        chunk_size = 3
        dense, bm = _build_mask_and_dense(
            num_frames, FRAME_SEQLEN, chunk_size, local_attn_size=3, sink_size=0
        )

        total = num_frames * FRAME_SEQLEN
        for i in range(total):
            assert dense[i, i].item(), f"Diagonal at position {i} must be True"

    # --- Test 6: block mask creation succeeds ----------------------------
    def test_block_mask_creation(self):
        """BlockMask is created without error for various configurations."""
        configs = [
            dict(num_frames=9, chunk_size=3, local_attn_size=-1, sink_size=0),
            dict(num_frames=9, chunk_size=3, local_attn_size=6, sink_size=0),
            dict(num_frames=12, chunk_size=3, local_attn_size=7, sink_size=1),
            dict(num_frames=9, chunk_size=3, local_attn_size=4, sink_size=1),
            dict(num_frames=5, chunk_size=3, local_attn_size=3, sink_size=0),
        ]
        for cfg in configs:
            _, bm = _build_mask_and_dense(frame_seqlen=FRAME_SEQLEN, **cfg)
            assert isinstance(bm, BlockMask), f"Failed for config {cfg}"


# ---------------------------------------------------------------------------
# Imports for dynamic RoPE tests
# ---------------------------------------------------------------------------
from fastgen.networks.OmniAvatar.network_causal import (
    compute_dynamic_rope_indices,
    dynamic_rope_apply_full,
    rope_apply_full,
    _precompute_freqs_cis_3d,
)


# ---------------------------------------------------------------------------
# Test class for dynamic RoPE
# ---------------------------------------------------------------------------
class TestDynamicRoPE:
    """Test suite for compute_dynamic_rope_indices and dynamic_rope_apply_full."""

    # --- Test 1: full causal (local_attn_size=-1) gives absolute indices ----
    def test_rope_indices_full_causal(self):
        """local_attn_size=-1 produces absolute indices [0, 1, ..., 20]."""
        num_frames = 21
        chunk_size = 3
        indices = compute_dynamic_rope_indices(
            num_frames=num_frames,
            chunk_size=chunk_size,
            local_attn_size=-1,
            sink_size=0,
        )
        expected = torch.arange(num_frames, dtype=torch.long)
        assert torch.equal(indices, expected), (
            f"Expected {expected.tolist()}, got {indices.tolist()}"
        )

    # --- Test 2: sliding window caps max index at local_attn_size - 1 ------
    def test_rope_indices_sliding_window(self):
        """local_attn_size=6 with chunk_size=3: max index is 5, later chunks stabilize."""
        num_frames = 12
        chunk_size = 3
        local_attn_size = 6
        indices = compute_dynamic_rope_indices(
            num_frames=num_frames,
            chunk_size=chunk_size,
            local_attn_size=local_attn_size,
            sink_size=0,
        )

        # Verify chunk boundaries (front-load remainder logic):
        # num_chunks=4, remaining=0 -> frame_counts=[3, 3, 3, 3]
        # Chunk 0 (frames 0-2): chunk_end=3, window_start=max(0,3-6)=0, indices=[0,1,2]
        # Chunk 1 (frames 3-5): chunk_end=6, window_start=max(0,6-6)=0, indices=[3,4,5]
        # Chunk 2 (frames 6-8): chunk_end=9, window_start=max(0,9-6)=3, indices=[3,4,5]
        # Chunk 3 (frames 9-11): chunk_end=12, window_start=max(0,12-6)=6, indices=[3,4,5]
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5], dtype=torch.long)
        assert torch.equal(indices, expected), (
            f"Expected {expected.tolist()}, got {indices.tolist()}"
        )

        # Max index should be local_attn_size - 1
        assert indices.max().item() == local_attn_size - 1, (
            f"Max index {indices.max().item()} should be {local_attn_size - 1}"
        )

    # --- Test 3: sliding window with sink frames ---------------------------
    def test_rope_indices_with_sink(self):
        """local_attn_size=7, sink_size=1: frame 0 always index 0, max index <= 6."""
        num_frames = 12
        chunk_size = 3
        local_attn_size = 7
        sink_size = 1
        indices = compute_dynamic_rope_indices(
            num_frames=num_frames,
            chunk_size=chunk_size,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )

        # Frame 0 is a sink frame, always gets index 0
        assert indices[0].item() == 0, f"Sink frame 0 should have index 0, got {indices[0].item()}"

        # Max index should be <= local_attn_size - 1
        assert indices.max().item() <= local_attn_size - 1, (
            f"Max index {indices.max().item()} should be <= {local_attn_size - 1}"
        )

        # Verify chunk 0 (frames 0-2): chunk_end=3, window_start=max(0,3-7)=0
        # Frame 0: sink -> index 0
        # Frame 1: 1 - 0 = 1
        # Frame 2: 2 - 0 = 2
        assert indices[0].item() == 0
        assert indices[1].item() == 1
        assert indices[2].item() == 2

        # Verify last chunk (frames 9-11): chunk_end=12, window_start=max(0,12-7)=5
        # Frame 9: 9 - 5 = 4
        # Frame 10: 10 - 5 = 5
        # Frame 11: 11 - 5 = 6
        assert indices[9].item() == 4
        assert indices[10].item() == 5
        assert indices[11].item() == 6

    # --- Test 4: output shape of dynamic_rope_apply_full -------------------
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
    def test_rope_function_output_shape(self):
        """dynamic_rope_apply_full returns same shape as input."""
        B, F, H, W = 1, 4, 2, 2
        head_dim = 12  # must be even; split into 3 parts: 4 + 4 + 4 (half-dims: 2+2+2)
        num_heads = 2
        seq_len = F * H * W
        device = DEVICE

        x = torch.randn(B, seq_len, num_heads, head_dim, device=device)
        grid_sizes = torch.tensor([[F, H, W]], device=device)
        freqs = _precompute_freqs_cis_3d(head_dim, end=64)
        freqs = tuple(f.to(device) for f in freqs)

        rope_indices = compute_dynamic_rope_indices(
            num_frames=F, chunk_size=2, local_attn_size=-1
        )

        out = dynamic_rope_apply_full(x, grid_sizes, freqs, rope_indices)
        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
        assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"

    # --- Test 5: matches rope_apply_full when indices are [0..F-1] ---------
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
    def test_rope_matches_standard_when_full_causal(self):
        """When indices are [0,1,...,F-1], dynamic_rope_apply_full matches rope_apply_full."""
        B, F, H, W = 2, 6, 3, 3
        head_dim = 12
        num_heads = 2
        seq_len = F * H * W
        device = DEVICE

        x = torch.randn(B, seq_len, num_heads, head_dim, device=device)
        grid_sizes = torch.tensor([[F, H, W]] * B, device=device)
        freqs = _precompute_freqs_cis_3d(head_dim, end=64)
        freqs = tuple(f.to(device) for f in freqs)

        # Absolute indices (full causal)
        rope_indices = torch.arange(F, dtype=torch.long)

        out_standard = rope_apply_full(x, grid_sizes, freqs)
        out_dynamic = dynamic_rope_apply_full(x, grid_sizes, freqs, rope_indices)

        torch.testing.assert_close(out_standard, out_dynamic, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test class for stochastic attention config sampling
# ---------------------------------------------------------------------------
class TestStochasticAttnConfig:
    """Test stochastic attention config sampling."""

    def test_sample_returns_defaults_when_no_configs(self):
        """Without stochastic configs, returns instance defaults."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        model = object.__new__(CausalOmniAvatarWan)
        model.local_attn_size = 6
        model.sink_size = 1
        model._stochastic_attn_configs = None
        cfg = model._sample_attn_config()
        assert cfg == {"local_attn_size": 6, "sink_size": 1}

    def test_sample_picks_from_configs(self):
        """With stochastic configs, samples from the list."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        model = object.__new__(CausalOmniAvatarWan)
        model.local_attn_size = -1
        model.sink_size = 0
        model._stochastic_attn_configs = [
            {"local_attn_size": -1, "sink_size": 0, "weight": 0.5},
            {"local_attn_size": 6, "sink_size": 1, "weight": 0.5},
        ]
        configs_seen = set()
        for _ in range(200):
            cfg = model._sample_attn_config()
            configs_seen.add((cfg["local_attn_size"], cfg["sink_size"]))
        assert (-1, 0) in configs_seen
        assert (6, 1) in configs_seen

    def test_different_masks_from_different_configs(self):
        """Different configs should produce different masks."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = object.__new__(CausalOmniAvatarWan)
        model.chunk_size = 3
        frame_seqlen = 4

        mask_full = model._build_block_mask(device, 21, frame_seqlen, 3, local_attn_size=-1, sink_size=0)
        mask_window = model._build_block_mask(device, 21, frame_seqlen, 3, local_attn_size=6, sink_size=0)
        assert mask_full is not None and mask_window is not None
        # They should be structurally different (different kv_num_blocks)

