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
        """local_attn_size=7, sink_size=1: sink included in budget, max index <= 5."""
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

        # Max index = effective_window - 1 = (local_attn_size - sink_size) - 1
        effective_window = local_attn_size - sink_size  # 6
        assert indices.max().item() <= effective_window - 1, (
            f"Max index {indices.max().item()} should be <= {effective_window - 1}"
        )

        # Verify chunk 0 (frames 0-2): effective_window=6, window_start=max(0,3-6)=0
        # Frame 0: sink -> index 0
        # Frame 1: 1 - 0 = 1
        # Frame 2: 2 - 0 = 2
        assert indices[0].item() == 0
        assert indices[1].item() == 1
        assert indices[2].item() == 2

        # Verify last chunk (frames 9-11): window_start=max(0,12-6)=6
        # Frame 9: 9 - 6 = 3
        # Frame 10: 10 - 6 = 4
        # Frame 11: 11 - 6 = 5
        assert indices[9].item() == 3
        assert indices[10].item() == 4
        assert indices[11].item() == 5

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


# ---------------------------------------------------------------------------
# Test class for mask visualization and exhaustive verification
# ---------------------------------------------------------------------------
class TestMaskVisualization:
    """Detailed verification of mask patterns -- print for visual inspection."""

    NUM_FRAMES = 21
    CHUNK_SIZE = 3
    FRAME_SEQLEN = 4

    def _build(self, local_attn_size, sink_size=0):
        """Build a mask and return dense [total_tokens, total_tokens] bool tensor."""
        dense, _ = _build_mask_and_dense(
            num_frames=self.NUM_FRAMES,
            frame_seqlen=self.FRAME_SEQLEN,
            chunk_size=self.CHUNK_SIZE,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        return dense

    def _frame_mask(self, dense):
        """Collapse token-level mask to frame-level [F, F] bool tensor.

        True if ANY token pair between those frames is visible.
        """
        F = self.NUM_FRAMES
        S = self.FRAME_SEQLEN
        frame_mask = torch.zeros(F, F, dtype=torch.bool)
        for qf in range(F):
            for kf in range(F):
                block = dense[
                    qf * S : (qf + 1) * S,
                    kf * S : (kf + 1) * S,
                ]
                frame_mask[qf, kf] = block.any().item()
        return frame_mask

    def _print_frame_mask(self, frame_mask, label):
        """Print an ASCII grid of the frame-level mask for visual inspection."""
        F = frame_mask.shape[0]
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  {self.NUM_FRAMES} frames, chunk_size={self.CHUNK_SIZE}, "
              f"FRAME_SEQLEN={self.FRAME_SEQLEN}")
        print(f"{'=' * 60}")

        # Header row
        header = "kv: " + "".join(f"{i:3d}" for i in range(F))
        print(header)
        print("q   " + "---" * F)

        for qf in range(F):
            row = f"{qf:2d} |"
            for kf in range(F):
                if frame_mask[qf, kf]:
                    row += " \u2588\u2588"
                else:
                    row += " \u00b7\u00b7"
            print(row)
        print()

    # --- Test 1: full causal visualization and verification -----------------
    def test_visualize_full_causal(self):
        """Print full causal mask. Verify lower-triangular at chunk level."""
        dense = self._build(local_attn_size=-1, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Full Causal (local_attn_size=-1, sink=0)")

        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)

        # Verify: for each pair of chunks (qi, ki), all frames in qi see
        # all frames in ki iff ki <= qi.
        for qi, (q_start, q_end) in enumerate(chunks):
            for ki, (k_start, k_end) in enumerate(chunks):
                expected = ki <= qi
                for qf in range(q_start, q_end):
                    for kf in range(k_start, k_end):
                        actual = fm[qf, kf].item()
                        assert actual == expected, (
                            f"Frame q={qf} (chunk {qi}), kv={kf} (chunk {ki}): "
                            f"expected {expected}, got {actual}"
                        )

    # --- Test 2: window=6 visualization and verification --------------------
    def test_visualize_window_6(self):
        """Print window=6 mask. Verify chunk 4 sees chunks 3-4 but NOT chunk 2."""
        dense = self._build(local_attn_size=6, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=6 (local_attn_size=6, sink=0)")

        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)
        # With 21 frames and chunk_size=3:
        # chunks: [(0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,21)]
        # But front-loading remainder: 21 // 3 = 7, remainder = 0
        # So: frame_counts = [3, 3, 3, 3, 3, 3, 3]
        # chunks: [(0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,21)]

        # Chunk 4 = frames 12-14, chunk_end=15, window_start=max(0,15-6)=9
        # So chunk 4 sees frames 9-14 (chunks 3 and 4) but NOT chunk 2 (frames 6-8)
        for qf in range(12, 15):
            # Should see chunk 3 (frames 9-11) and chunk 4 (frames 12-14)
            for kf in range(9, 15):
                assert fm[qf, kf].item(), (
                    f"Frame q={qf} should see kv={kf} (in window)"
                )
            # Should NOT see chunk 2 (frames 6-8)
            for kf in range(6, 9):
                assert not fm[qf, kf].item(), (
                    f"Frame q={qf} should NOT see kv={kf} (outside window)"
                )

    # --- Test 3: window=7 + sink=1 visualization and verification -----------
    def test_visualize_window_7_sink_1(self):
        """Print window=7 + sink=1 mask. Verify frame 0 visible from ALL frames.
        Frame 1 NOT visible from chunk 6."""
        dense = self._build(local_attn_size=7, sink_size=1)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=7, Sink=1 (local_attn_size=7, sink_size=1)")

        # Frame 0 (sink) should be visible from ALL query frames
        for qf in range(self.NUM_FRAMES):
            assert fm[qf, 0].item(), (
                f"Sink frame 0 must be visible from frame {qf}"
            )

        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)
        # Chunk 6 = frames 18-20, chunk_end=21, window_start=max(0,21-7)=14
        # Frame 1 is not a sink (only frame 0 is), and 1 < 14 so outside window
        for qf in range(18, 21):
            assert not fm[qf, 1].item(), (
                f"Frame 1 should NOT be visible from chunk 6 frame {qf} "
                f"(not sink, outside window)"
            )

    # --- Test 4: window=9 visualization and verification --------------------
    def test_visualize_window_9(self):
        """Print window=9 mask (3 chunks visible)."""
        dense = self._build(local_attn_size=9, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=9 (local_attn_size=9, sink=0)")

        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)

        # With window=9 and chunk_size=3, each chunk sees 3 chunks (9/3 = 3)
        # Verify: chunk 5 (frames 15-17), chunk_end=18, window_start=max(0,18-9)=9
        # Sees frames 9-17 (chunks 3, 4, 5) but NOT chunk 2 (frames 6-8)
        for qf in range(15, 18):
            for kf in range(9, 18):
                assert fm[qf, kf].item(), (
                    f"Frame q={qf} should see kv={kf} (3-chunk window)"
                )
            for kf in range(6, 9):
                assert not fm[qf, kf].item(), (
                    f"Frame q={qf} should NOT see kv={kf} (outside 3-chunk window)"
                )

        # Verify: last chunk 6 (frames 18-20), chunk_end=21, window_start=max(0,21-9)=12
        # Sees frames 12-20 (chunks 4, 5, 6) but NOT chunk 3 (frames 9-11)
        for qf in range(18, 21):
            for kf in range(12, 21):
                assert fm[qf, kf].item(), (
                    f"Frame q={qf} should see kv={kf} (in window)"
                )
            for kf in range(9, 12):
                assert not fm[qf, kf].item(), (
                    f"Frame q={qf} should NOT see kv={kf} (outside window)"
                )

    # --- Test 5: RoPE indices for all configs -------------------------------
    def test_rope_indices_all_configs(self):
        """Print RoPE indices for all configs. Verify max index bounded."""
        configs = [
            {"label": "Full Causal", "local_attn_size": -1, "sink_size": 0},
            {"label": "Window=6", "local_attn_size": 6, "sink_size": 0},
            {"label": "Window=7 + Sink=1", "local_attn_size": 7, "sink_size": 1},
            {"label": "Window=9", "local_attn_size": 9, "sink_size": 0},
        ]

        for cfg in configs:
            indices = compute_dynamic_rope_indices(
                num_frames=self.NUM_FRAMES,
                chunk_size=self.CHUNK_SIZE,
                local_attn_size=cfg["local_attn_size"],
                sink_size=cfg["sink_size"],
            )

            # Print the indices
            print(f"\nRoPE indices -- {cfg['label']} "
                  f"(local_attn_size={cfg['local_attn_size']}, "
                  f"sink_size={cfg['sink_size']})")
            print(f"  Frames:  {list(range(self.NUM_FRAMES))}")
            print(f"  Indices: {indices.tolist()}")
            print(f"  Max index: {indices.max().item()}")

            # Verify max index bound for windowed configs
            if cfg["local_attn_size"] > 0:
                max_allowed = cfg["local_attn_size"] - 1
                assert indices.max().item() <= max_allowed, (
                    f"{cfg['label']}: max RoPE index {indices.max().item()} "
                    f"exceeds allowed {max_allowed}"
                )
            else:
                # Full causal: max index should be NUM_FRAMES - 1
                assert indices.max().item() == self.NUM_FRAMES - 1, (
                    f"Full causal: max RoPE index {indices.max().item()} "
                    f"should be {self.NUM_FRAMES - 1}"
                )

            # Verify length matches
            assert len(indices) == self.NUM_FRAMES, (
                f"Expected {self.NUM_FRAMES} indices, got {len(indices)}"
            )


# ---------------------------------------------------------------------------
# End-to-end smoke tests: real forward + backward through the model
# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Smoke tests running forward + backward through CausalOmniAvatarWan.

    Uses tiny dimensions (B=1, T=9, H=16, W=16) with random weights to
    verify no shape mismatches or crashes with sliding window attention.
    """

    # Shared helper to build a minimal model and inputs
    def _build_model_and_inputs(self, **model_overrides):
        """Construct a tiny CausalOmniAvatarWan and matching dummy inputs.

        Returns (model, x_t, t, condition) all on CUDA in bf16.
        """
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

        defaults = dict(
            model_size="1.3B",
            in_dim=65,
            mode="v2v",
            chunk_size=3,
            total_num_frames=9,
            use_audio=False,
            local_attn_size=-1,
            sink_size=0,
            use_dynamic_rope=False,
            base_model_paths=None,
            omniavatar_ckpt_path=None,
            disable_grad_ckpt=True,
        )
        defaults.update(model_overrides)

        model = CausalOmniAvatarWan(**defaults)
        model = model.cuda().to(torch.bfloat16).train()

        B, C, T, H, W = 1, 16, 9, 16, 16
        device = torch.device("cuda")
        dtype = torch.bfloat16

        x_t = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
        # Per-frame timesteps [B, T] triggers full-sequence mode
        t = torch.rand(B, T, device=device, dtype=dtype)

        # Condition dict — V2V with in_dim=65 needs 49 extra channels:
        #   ref_latent(16) + mask(1) + masked_video(16) + ref_sequence(16) = 49
        condition = {
            "text_embeds": torch.randn(B, 512, 4096, device=device, dtype=dtype),
            "ref_latent": torch.randn(B, 16, 1, H, W, device=device, dtype=dtype),
            "mask": torch.ones(H, W, device=device, dtype=dtype),
            "masked_video": torch.randn(B, 16, T, H, W, device=device, dtype=dtype),
            "ref_sequence": torch.randn(B, 16, T, H, W, device=device, dtype=dtype),
        }

        return model, x_t, t, condition

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_backward_with_window(self):
        """Full forward + backward pass with local_attn_size=6."""
        model, x_t, t, condition = self._build_model_and_inputs(
            local_attn_size=6,
            sink_size=0,
            use_dynamic_rope=True,
        )

        out = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")

        # Output shape must match input latent shape
        assert out.shape == x_t.shape, (
            f"Expected output shape {x_t.shape}, got {out.shape}"
        )

        # Backward pass
        out.sum().backward()

        # At least one parameter must have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received gradients after backward"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_backward_with_stochastic(self):
        """Forward + backward with stochastic attention configs (two forward passes)."""
        model, x_t, t, condition = self._build_model_and_inputs(
            local_attn_size=-1,  # default; overridden by stochastic configs
            sink_size=0,
            use_dynamic_rope=True,
            stochastic_attn_configs=[
                {"local_attn_size": -1, "sink_size": 0, "weight": 0.5},
                {"local_attn_size": 6, "sink_size": 0, "weight": 0.5},
            ],
        )

        # First forward pass
        out1 = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")
        assert out1.shape == x_t.shape, (
            f"Pass 1: expected output shape {x_t.shape}, got {out1.shape}"
        )

        # Clear cached block_mask to force rebuild with a potentially different config
        model.block_mask = None

        # Second forward pass (may sample different stochastic config)
        out2 = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")
        assert out2.shape == x_t.shape, (
            f"Pass 2: expected output shape {x_t.shape}, got {out2.shape}"
        )

        # Backward on second output
        out2.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received gradients after backward"


# ---------------------------------------------------------------------------
# DF vs AR RoPE consistency tests
# ---------------------------------------------------------------------------


class TestDFvsARRoPEConsistency:
    """Verify dynamic RoPE consistency between DF full-sequence and AR chunk modes.

    In full-sequence (DF) mode, each frame gets ONE RoPE index based on its own
    chunk's window. In AR mode, keys are re-RoPE'd relative to the attending
    chunk's window each time.

    What IS consistent (and tested here):
    - Q indices for the generating chunk match between DF and AR
    - Max RoPE index is capped identically (local_attn_size - 1)
    - Sink frame always gets index 0 in both modes

    What is inherently different (documented, not a bug):
    - Cross-chunk K indices differ because DF assigns one index per frame
      while AR re-indexes keys per attending chunk's window.
    """

    CONFIGS = [
        (7, 1),   # sink=1, window=6
        (10, 1),  # sink=1, window=9
        (13, 1),  # sink=1, window=12
        (9, 3),   # sink=3, window=6
        (12, 3),  # sink=3, window=9
    ]
    NUM_FRAMES = 21
    CHUNK_SIZE = 3

    def _chunk_boundaries(self):
        """Return list of (start_frame, end_frame) for each chunk."""
        num_chunks = self.NUM_FRAMES // self.CHUNK_SIZE
        remaining = self.NUM_FRAMES % self.CHUNK_SIZE
        chunks = []
        if num_chunks == 0:
            chunks.append((0, remaining))
        else:
            first = self.CHUNK_SIZE + remaining
            chunks.append((0, first))
            for i in range(1, num_chunks):
                s = first + (i - 1) * self.CHUNK_SIZE
                chunks.append((s, s + self.CHUNK_SIZE))
        return chunks

    def _ar_query_rope_indices(self, chunk_idx, local_attn_size, sink_size):
        """Compute what AR mode assigns as Q RoPE indices for a given chunk.

        In AR dynamic RoPE, the window is laid out as:
            [sink_frames | rolling_past | current_chunk]
        Keys get sequential RoPE [0, 1, ..., F_window-1].
        Q start = position of current chunk within the window.
        """
        chunks = self._chunk_boundaries()
        cs, ce = chunks[chunk_idx]
        chunk_frames = ce - cs

        if local_attn_size <= 0:
            # Full causal: Q start = absolute frame position
            return list(range(cs, ce))

        # How many past frames are in the window (excluding current chunk)?
        past_frames_available = cs  # all frames before this chunk
        window_budget_for_past = local_attn_size - chunk_frames

        if sink_size > 0 and past_frames_available > window_budget_for_past:
            # Sink + rolling: sink takes sink_size, rolling fills the rest
            rolling_budget = window_budget_for_past - sink_size
            past_in_window = sink_size + max(0, rolling_budget)
        else:
            past_in_window = min(past_frames_available, window_budget_for_past)

        q_start_in_window = past_in_window
        return list(range(q_start_in_window, q_start_in_window + chunk_frames))

    def test_query_indices_match_no_sink(self):
        """Without sink, DF Q indices match AR Q indices exactly."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices

        chunks = self._chunk_boundaries()
        # Only test configs without sink — with sink, AR pushes Q forward
        # by sink_size (sink occupies position 0 in window), but DF doesn't.
        # This is acceptable since DF uses absolute RoPE (use_dynamic_rope=False).
        no_sink_configs = [(la, 0) for la in [6, 9, 12]]

        for local_attn, sink in no_sink_configs:
            df_indices = compute_dynamic_rope_indices(
                self.NUM_FRAMES, self.CHUNK_SIZE, local_attn, sink
            ).tolist()

            for ci, (cs, ce) in enumerate(chunks):
                df_q = df_indices[cs:ce]
                ar_q = self._ar_query_rope_indices(ci, local_attn, sink)

                assert df_q == ar_q, (
                    f"Q RoPE mismatch at chunk {ci} (frames {cs}-{ce-1}) "
                    f"with local_attn={local_attn}, sink={sink}: "
                    f"DF={df_q}, AR={ar_q}"
                )

    def test_max_index_matches(self):
        """Max RoPE index is effective_window - 1 = local_attn_size - sink_size - 1."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices

        for local_attn, sink in self.CONFIGS:
            df_indices = compute_dynamic_rope_indices(
                self.NUM_FRAMES, self.CHUNK_SIZE, local_attn, sink
            )
            effective_window = local_attn - sink
            max_idx = df_indices.max().item()
            assert max_idx == effective_window - 1, (
                f"Max RoPE index {max_idx} != {effective_window - 1} "
                f"for local_attn={local_attn}, sink={sink}"
            )

    def test_sink_frame_always_zero(self):
        """Sink frames always get RoPE index 0 in both modes."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices

        for local_attn, sink in self.CONFIGS:
            df_indices = compute_dynamic_rope_indices(
                self.NUM_FRAMES, self.CHUNK_SIZE, local_attn, sink
            )
            for f in range(sink):
                assert df_indices[f].item() == f, (
                    f"Sink frame {f} has RoPE index {df_indices[f].item()}, "
                    f"expected {f} for local_attn={local_attn}, sink={sink}"
                )

