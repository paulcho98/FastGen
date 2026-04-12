# Sliding Window Attention for Causal OmniAvatar

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable sliding window attention, dynamic RoPE, attention sinks, and stochastic attention config sampling to the causal student network's full-sequence training mode (DF/KD), matching the existing AR-mode support.

**Architecture:** The causal network (`CausalOmniAvatarWan`) already supports `local_attn_size`, `sink_size`, and `use_dynamic_rope` in AR mode (chunk-by-chunk with KV cache). We extend these to the full-sequence path (`_forward_full_sequence`) used by DF training: (1) modify `_build_block_mask` to produce sliding-window + sink FlexAttention masks, (2) add a per-frame dynamic RoPE function for full-sequence mode, (3) add stochastic attention config sampling that rebuilds the mask each forward pass. All changes are in the causal network only — the bidirectional teacher/fake_score are unaffected.

**Tech Stack:** PyTorch FlexAttention (`create_block_mask`, `flex_attention`), 3D RoPE, gradient checkpointing (`use_reentrant=False`)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `fastgen/networks/OmniAvatar/network_causal.py` | Modify | Core changes: `_build_block_mask`, RoPE, `_forward_full_sequence` |
| `fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py` | Modify | Wire new params into DF experiment config |
| `fastgen/configs/experiments/OmniAvatar/config_sf.py` | Modify | Wire new params into SF experiment config |
| `tests/test_sliding_window_attention.py` | Create | Verification tests for mask, RoPE, and end-to-end |

---

## Task 1: Sliding Window FlexAttention Mask

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:1221-1274` (`_build_block_mask`)
- Create: `tests/test_sliding_window_attention.py`

### Explanation

Currently `_build_block_mask` only builds the `ends[]` array. We add a `starts[]` array for the window lower bound and a `sink_end` constant for sink tokens. The mask becomes:

```
in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
is_sink   = kv_idx < sink_end
allow     = in_window | is_sink | (q_idx == kv_idx)
```

With `local_attn_size=-1` (default), `starts[]` is all zeros → full causal (unchanged behavior). With `local_attn_size=6, sink_size=1`: each chunk sees itself + previous chunks within the 6-frame budget, minus 1 frame reserved for the sink.

- [ ] **Step 1: Write mask verification test**

```python
# tests/test_sliding_window_attention.py
"""Tests for sliding window attention mask, dynamic RoPE, and stochastic configs."""

import pytest
import math
import torch

# Skip entire module if FlexAttention not available
flex_available = False
try:
    from torch.nn.attention.flex_attention import create_block_mask
    flex_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not flex_available, reason="FlexAttention not available")


def _dense_mask_from_block_mask(block_mask, Q_LEN, KV_LEN):
    """Convert a BlockMask to a dense [Q, KV] boolean tensor for verification."""
    dense = torch.zeros(Q_LEN, KV_LEN, dtype=torch.bool, device=block_mask.kv_num_blocks.device)
    BLOCK = block_mask.BLOCK_SIZE[0]
    for q_block in range(math.ceil(Q_LEN / BLOCK)):
        q_start = q_block * BLOCK
        q_end = min(q_start + BLOCK, Q_LEN)
        num_kv = block_mask.kv_num_blocks[0, 0, q_block].item()
        for idx in range(num_kv):
            kv_block = block_mask.kv_indices[0, 0, q_block, idx].item()
            kv_start = kv_block * BLOCK
            kv_end = min(kv_start + BLOCK, KV_LEN)
            dense[q_start:q_end, kv_start:kv_end] = True
    return dense


def _chunk_boundaries(num_frames, chunk_size):
    """Return list of (start_frame, end_frame) for each chunk."""
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


class TestBlockMask:
    """Test _build_block_mask with various local_attn_size and sink_size settings."""

    # Use small dimensions for fast tests
    NUM_FRAMES = 21
    CHUNK_SIZE = 3
    FRAME_SEQLEN = 4  # tiny spatial dim for speed

    def _build(self, local_attn_size=-1, sink_size=0):
        """Import and call the real _build_block_mask."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        # Call the static-ish method directly (it only uses self.chunk_size for default)
        # We'll call the unbound function with explicit args
        model = object.__new__(CausalOmniAvatarWan)
        model.chunk_size = self.CHUNK_SIZE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = CausalOmniAvatarWan._build_block_mask(
            model, device, self.NUM_FRAMES, self.FRAME_SEQLEN,
            chunk_size=self.CHUNK_SIZE,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        total_len = self.NUM_FRAMES * self.FRAME_SEQLEN
        padded_len = math.ceil(total_len / 128) * 128
        dense = _dense_mask_from_block_mask(mask, padded_len, padded_len)
        # Trim to actual sequence length
        return dense[:total_len, :total_len]

    def _frame_of(self, token_idx):
        return token_idx // self.FRAME_SEQLEN

    def _chunk_of(self, frame):
        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)
        for i, (s, e) in enumerate(chunks):
            if s <= frame < e:
                return i
        return -1

    def test_full_causal_unchanged(self):
        """local_attn_size=-1, sink_size=0 should match original causal behavior."""
        dense = self._build(local_attn_size=-1, sink_size=0)
        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)
        total = self.NUM_FRAMES * self.FRAME_SEQLEN

        for q in range(0, total, self.FRAME_SEQLEN):  # sample one token per frame
            q_frame = self._frame_of(q)
            q_chunk = self._chunk_of(q_frame)
            q_chunk_end_frame = chunks[q_chunk][1]

            for k in range(0, total, self.FRAME_SEQLEN):
                k_frame = self._frame_of(k)
                expected = k_frame < q_chunk_end_frame or q == k
                actual = dense[q, k].item()
                assert actual == expected, (
                    f"Full causal mismatch at q_frame={q_frame} k_frame={k_frame}: "
                    f"expected={expected}, got={actual}"
                )

    def test_sliding_window_basic(self):
        """local_attn_size=6 (2 chunks): each chunk sees itself + 1 prior chunk."""
        dense = self._build(local_attn_size=6, sink_size=0)
        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)

        # Chunk 0 (frames 0-2): sees only itself
        q_token = 0 * self.FRAME_SEQLEN  # frame 0
        assert dense[q_token, 0].item() is True  # frame 0
        assert dense[q_token, 2 * self.FRAME_SEQLEN].item() is True  # frame 2 (same chunk)

        # Chunk 3 (frames 9-11): sees chunks 2-3 (frames 6-11), NOT chunk 1 (frames 3-5)
        q_token = 9 * self.FRAME_SEQLEN  # frame 9
        assert dense[q_token, 6 * self.FRAME_SEQLEN].item() is True   # frame 6 (chunk 2, in window)
        assert dense[q_token, 11 * self.FRAME_SEQLEN].item() is True  # frame 11 (same chunk)
        assert dense[q_token, 5 * self.FRAME_SEQLEN].item() is False  # frame 5 (chunk 1, outside window)

    def test_sliding_window_with_sink(self):
        """local_attn_size=7, sink_size=1: window of 7 frames + first frame always visible."""
        dense = self._build(local_attn_size=7, sink_size=1)
        sink_end = 1 * self.FRAME_SEQLEN  # 1 frame of sink

        # Chunk 6 (frames 18-20): window covers frames 14-20 (7 frames)
        # But also always sees frame 0 (sink)
        q_token = 18 * self.FRAME_SEQLEN  # frame 18
        assert dense[q_token, 0].item() is True   # frame 0 (sink)
        assert dense[q_token, 14 * self.FRAME_SEQLEN].item() is True  # frame 14 (in window)
        assert dense[q_token, 13 * self.FRAME_SEQLEN].item() is False  # frame 13 (outside window, not sink)
        assert dense[q_token, 1 * self.FRAME_SEQLEN].item() is False   # frame 1 (not sink, not in window)

    def test_sink_included_in_window(self):
        """Verify sink frames consume part of local_attn_size budget."""
        # local_attn_size=4, sink_size=1: total budget = 4 frames
        # sink uses 1, rolling window gets 3 (= 1 chunk)
        dense = self._build(local_attn_size=4, sink_size=1)

        # Chunk 4 (frames 12-14): window = frames 12,13,14 (generating) + sink (frame 0)
        # Frames 9,10,11 should NOT be visible (window budget exhausted)
        q_token = 12 * self.FRAME_SEQLEN
        assert dense[q_token, 0].item() is True   # sink
        assert dense[q_token, 12 * self.FRAME_SEQLEN].item() is True  # own chunk
        assert dense[q_token, 11 * self.FRAME_SEQLEN].item() is False  # NOT in window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestBlockMask -v 2>&1 | head -40`
Expected: FAIL — `_build_block_mask` doesn't accept `local_attn_size` or `sink_size` params.

- [ ] **Step 3: Implement sliding window mask in `_build_block_mask`**

Replace the existing `_build_block_mask` method at lines 1221-1274 of `network_causal.py`:

```python
    def _build_block_mask(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int = None,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ) -> Optional[BlockMask]:
        """Build a chunk-wise causal attention mask for full-sequence mode.

        Tokens within the same chunk attend bidirectionally. Tokens can attend
        to all previous chunks within the sliding window. Optionally, the first
        ``sink_size`` frames are always visible (attention sinks).

        Args:
            device: Target device.
            num_frames: Number of latent frames.
            frame_seqlen: Tokens per frame (h * w after patchification).
            chunk_size: Frames per chunk (defaults to self.chunk_size).
            local_attn_size: Total attention window in frames (including sink
                and current chunk). -1 means unlimited (full causal).
            sink_size: Number of initial frames always kept visible.
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        if chunk_size is None:
            chunk_size = self.chunk_size

        total_length = num_frames * frame_seqlen
        pad_len = math.ceil(total_length / 128) * 128 - total_length
        padded_length = total_length + pad_len

        # Build chunk boundaries — front-load remainder into first chunk
        num_chunks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size

        frame_counts = []
        if num_frames > 0:
            if num_chunks == 0:
                frame_counts.append(remaining_size)
            else:
                frame_counts.append(chunk_size + remaining_size)
                frame_counts.extend([chunk_size] * max(num_chunks - 1, 0))

        # ends[token] = exclusive end of that token's chunk (in token units)
        ends = torch.zeros(padded_length, device=device, dtype=torch.long)
        # starts[token] = inclusive start of that token's attention window (in token units)
        starts = torch.zeros(padded_length, device=device, dtype=torch.long)

        current_start = 0
        for frames_in_chunk in frame_counts:
            chunk_len_tokens = frames_in_chunk * frame_seqlen
            chunk_end = current_start + chunk_len_tokens

            # Window start: how far back can this chunk see?
            if local_attn_size > 0:
                # Window covers local_attn_size frames ending at chunk_end
                window_start_frame = max(0, (current_start // frame_seqlen + frames_in_chunk) - local_attn_size)
                window_start_token = window_start_frame * frame_seqlen
            else:
                window_start_token = 0  # full causal — see everything

            ends[current_start : chunk_end] = chunk_end
            starts[current_start : chunk_end] = window_start_token
            current_start = chunk_end

        sink_end = sink_size * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
            is_sink = kv_idx < sink_end
            return in_window | is_sink | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            _compile=False,
            device=device,
        )
        return block_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestBlockMask -v 2>&1 | tail -20`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_sliding_window_attention.py
git commit -m "feat: add sliding window + sink support to full-sequence FlexAttention mask"
```

---

## Task 2: Dynamic RoPE for Full-Sequence Mode

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:181-234` (add new RoPE function)
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:1337-1458` (`_forward_full_sequence`)
- Modify: `tests/test_sliding_window_attention.py` (add RoPE tests)

### Explanation

Currently `_forward_full_sequence` calls `rope_apply_full` which assigns absolute temporal RoPE indices `[0, 1, 2, ..., 20]`. With dynamic RoPE, each frame gets an index relative to its chunk's window start, capping the max index at `local_attn_size - 1`. This matches AR-mode dynamic RoPE behavior.

We add a new function `dynamic_rope_apply_full` that takes a per-frame index tensor instead of using a linear range. We also need to pass the window config into `_forward_full_sequence`.

- [ ] **Step 1: Write RoPE index verification test**

Append to `tests/test_sliding_window_attention.py`:

```python
class TestDynamicRoPE:
    """Test per-frame RoPE index computation for full-sequence dynamic RoPE."""

    def test_rope_indices_full_causal(self):
        """With local_attn_size=-1: indices are absolute [0, 1, ..., 20]."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices
        indices = compute_dynamic_rope_indices(
            num_frames=21, chunk_size=3, local_attn_size=-1, sink_size=0,
        )
        assert indices.tolist() == list(range(21))

    def test_rope_indices_sliding_window(self):
        """With local_attn_size=6: max index is 5 (window size - 1)."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices
        indices = compute_dynamic_rope_indices(
            num_frames=21, chunk_size=3, local_attn_size=6, sink_size=0,
        )
        # Chunk 0 (frames 0-2): window_start=0, indices = [0, 1, 2]
        assert indices[0:3].tolist() == [0, 1, 2]
        # Chunk 1 (frames 3-5): window_start=0, indices = [3, 4, 5]
        assert indices[3:6].tolist() == [3, 4, 5]
        # Chunk 2 (frames 6-8): window_start=3, indices = [3, 4, 5]
        assert indices[6:9].tolist() == [3, 4, 5]
        # Chunk 6 (frames 18-20): window_start=15, indices = [3, 4, 5]
        assert indices[18:21].tolist() == [3, 4, 5]
        # No index exceeds local_attn_size - 1
        assert indices.max().item() <= 5

    def test_rope_indices_with_sink(self):
        """With sink_size=1: sink frame gets index 0, window frames offset accordingly."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices
        indices = compute_dynamic_rope_indices(
            num_frames=21, chunk_size=3, local_attn_size=7, sink_size=1,
        )
        # Frame 0 is sink — gets index 0
        assert indices[0].item() == 0
        # Chunk 0 (frames 0-2): all within window, indices = [0, 1, 2]
        assert indices[0:3].tolist() == [0, 1, 2]
        # Later chunks: sink at 0, rolling context starts at 1
        # Chunk 4 (frames 12-14): window_start_frame=8, frames 8-14 in window
        # sink(frame 0)=0, frame 8=1, ..., frame 14=7? No — sink is included in budget
        # Window = 7 frames: frames 8,9,10,11,12,13,14. Sink(frame 0) at index 0.
        # But frame 0 IS in the window for chunk 0, not for chunk 4.
        # For chunk 4: window covers frames 8-14 (7 frames). Indices = [0,1,2,3,4,5,6]
        # Frames 12,13,14 map to indices 4,5,6
        assert indices[12:15].tolist() == [4, 5, 6]
        # Max index capped
        assert indices.max().item() <= 6

    def test_rope_function_output_shape(self):
        """dynamic_rope_apply_full returns same shape as input."""
        from fastgen.networks.OmniAvatar.network_causal import (
            dynamic_rope_apply_full,
            _precompute_freqs_cis_3d,
        )
        B, S, N, D = 2, 21 * 4, 8, 16  # 21 frames, 4 tokens/frame, 8 heads, 16 dim
        x = torch.randn(B, S, N, D)
        grid_sizes = torch.tensor([[21, 2, 2]] * B, dtype=torch.long)
        freqs = _precompute_freqs_cis_3d(D)
        rope_indices = torch.arange(21)

        result = dynamic_rope_apply_full(x, grid_sizes, freqs, rope_indices)
        assert result.shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestDynamicRoPE -v 2>&1 | head -30`
Expected: FAIL — `compute_dynamic_rope_indices` and `dynamic_rope_apply_full` don't exist.

- [ ] **Step 3: Implement `compute_dynamic_rope_indices`**

Add after `rope_apply_full` (after line 234) in `network_causal.py`:

```python
def compute_dynamic_rope_indices(
    num_frames: int,
    chunk_size: int,
    local_attn_size: int = -1,
    sink_size: int = 0,
) -> torch.Tensor:
    """Compute per-frame RoPE temporal indices for dynamic RoPE in full-sequence mode.

    With sliding window, each frame gets an index relative to its chunk's
    window start. This caps the maximum temporal RoPE index at
    ``local_attn_size - 1``, matching AR-mode dynamic RoPE behavior.

    Args:
        num_frames: Total number of latent frames.
        chunk_size: Frames per chunk.
        local_attn_size: Attention window in frames (-1 = unlimited → absolute indices).
        sink_size: Number of initial sink frames (always index 0..sink_size-1).

    Returns:
        [num_frames] tensor of per-frame RoPE temporal indices.
    """
    if local_attn_size <= 0:
        # Full causal — absolute indices
        return torch.arange(num_frames)

    # Build chunk boundaries (same logic as _build_block_mask)
    num_chunks = num_frames // chunk_size
    remaining = num_frames % chunk_size

    frame_counts = []
    if num_chunks == 0:
        frame_counts.append(remaining)
    else:
        frame_counts.append(chunk_size + remaining)
        frame_counts.extend([chunk_size] * max(num_chunks - 1, 0))

    indices = torch.zeros(num_frames, dtype=torch.long)
    current_frame = 0
    for frames_in_chunk in frame_counts:
        chunk_end_frame = current_frame + frames_in_chunk
        window_start_frame = max(0, chunk_end_frame - local_attn_size)

        for f in range(current_frame, chunk_end_frame):
            indices[f] = f - window_start_frame

        current_frame = chunk_end_frame

    return indices
```

- [ ] **Step 4: Implement `dynamic_rope_apply_full`**

Add after `compute_dynamic_rope_indices` in `network_causal.py`:

```python
def dynamic_rope_apply_full(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rope_frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply 3D RoPE with per-frame temporal indices for dynamic positioning.

    Like ``causal_rope_apply`` but uses arbitrary per-frame temporal indices
    instead of a contiguous range. Spatial (H, W) indices remain unchanged.

    Args:
        x: [B, S, num_heads, head_dim]
        grid_sizes: [B, 3] — (F, H, W) per sample
        freqs: tuple of 3 complex frequency tables (f, h, w)
        rope_frame_indices: [num_frames] per-frame temporal RoPE indices

    Returns:
        Tensor same shape as x with RoPE applied.
    """
    n, c = x.size(2), x.size(3) // 2
    freq_f, freq_h, freq_w = freqs

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )

        # Gather temporal frequencies using per-frame indices instead of contiguous range
        frame_indices = rope_frame_indices[:f].to(freq_f.device)
        f_freqs = freq_f[frame_indices]  # [F, freq_dim]

        freqs_i = torch.cat(
            [
                f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)
```

- [ ] **Step 5: Run RoPE tests to verify they pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestDynamicRoPE -v 2>&1 | tail -20`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_sliding_window_attention.py
git commit -m "feat: add compute_dynamic_rope_indices and dynamic_rope_apply_full"
```

---

## Task 3: Wire Window/RoPE into `_forward_full_sequence`

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:1337-1458` (`_forward_full_sequence`)
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:1413-1416` (mask building)
- Modify: `tests/test_sliding_window_attention.py` (add integration test)

### Explanation

`_forward_full_sequence` currently builds the block mask lazily and caches it (`self.block_mask`). We need to:
1. Pass `local_attn_size` and `sink_size` to `_build_block_mask`
2. When `use_dynamic_rope=True`, compute per-frame indices and use `dynamic_rope_apply_full` instead of `rope_apply_full`
3. When stochastic configs are active (Task 4), rebuild the mask each call. For now, just pass the instance-level params.

- [ ] **Step 1: Write integration test**

Append to `tests/test_sliding_window_attention.py`:

```python
class TestForwardFullSequence:
    """Test that _forward_full_sequence uses window/RoPE params correctly."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mask_uses_local_attn_size(self):
        """Verify the mask built inside _forward_full_sequence respects local_attn_size."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        import os

        # Instantiate a minimal model (no weights needed — just check mask)
        os.environ["FASTGEN_DISABLE_FLEX_ATTENTION"] = "0"
        model = CausalOmniAvatarWan.__new__(CausalOmniAvatarWan)
        model.chunk_size = 3
        model.local_attn_size = 6
        model.sink_size = 0
        model.use_dynamic_rope = False
        model.block_mask = None
        model._stochastic_attn_configs = None

        device = torch.device("cuda")
        frame_seqlen = 32 * 32  # = 1024 for 64x64 latent with patch (1,2,2)

        # Build mask via the same path _forward_full_sequence uses
        mask = model._build_block_mask(
            device, 21, frame_seqlen, model.chunk_size,
            local_attn_size=model.local_attn_size,
            sink_size=model.sink_size,
        )
        assert mask is not None
        # Just verify it was built without error — detailed mask tests are in Task 1
```

- [ ] **Step 2: Modify `_forward_full_sequence` to use window params and dynamic RoPE**

In `network_causal.py`, modify `_forward_full_sequence` (lines 1363-1457). The key changes are in the mask building block and the RoPE application. Replace lines 1413-1416 (mask building) and lines 1367-1369 (RoPE):

Replace the mask building section (currently lines 1413-1416):
```python
        # Build block mask with window/sink params
        # Rebuild when stochastic configs change the params (Task 4);
        # otherwise cache as before.
        if self.block_mask is None and FLEX_ATTENTION_AVAILABLE:
            frame_seqlen = h * w
            attn_local = getattr(self, '_current_local_attn_size', self.local_attn_size)
            attn_sink = getattr(self, '_current_sink_size', self.sink_size)
            self.block_mask = self._build_block_mask(
                device, f, frame_seqlen, self.chunk_size,
                local_attn_size=attn_local,
                sink_size=attn_sink,
            )
```

Replace the RoPE application (currently line 1368-1369 inside `CausalSelfAttention.forward`, full-sequence branch). Actually, RoPE is applied inside `CausalSelfAttention.forward()` at line 368-369, not in `_forward_full_sequence`. We need to change `_forward_full_sequence` to pass RoPE info to the blocks.

Actually, RoPE is applied inside `CausalSelfAttention` which reads `freqs` from kwargs. The cleanest approach: when `use_dynamic_rope=True` in full-sequence mode, pre-modify `self.freqs` to use the dynamic indices, or pass dynamic grid_sizes. But the simplest approach is to modify `rope_apply_full` to accept optional indices:

Replace the full-sequence RoPE path in `CausalSelfAttention.forward()` (line 368-369):
```python
            # ----- Full-sequence mode (training / bidirectional eval) -----
            rope_frame_indices = kwargs.get("rope_frame_indices", None) if kwargs else None
            if rope_frame_indices is not None:
                roped_q = dynamic_rope_apply_full(q, grid_sizes, freqs, rope_frame_indices).type_as(v)
                roped_k = dynamic_rope_apply_full(k, grid_sizes, freqs, rope_frame_indices).type_as(v)
            else:
                roped_q = rope_apply_full(q, grid_sizes, freqs).type_as(v)
                roped_k = rope_apply_full(k, grid_sizes, freqs).type_as(v)
```

Wait — `CausalSelfAttention.forward()` doesn't accept `**kwargs`. We need to add a `rope_frame_indices` parameter. Let me revise:

Modify `CausalSelfAttention.forward()` signature (line 336) to add `rope_frame_indices=None`:
```python
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
        rope_frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
```

Then in the full-sequence branch (line 367-369), change to:
```python
            if rope_frame_indices is not None:
                roped_q = dynamic_rope_apply_full(q, grid_sizes, freqs, rope_frame_indices).type_as(v)
                roped_k = dynamic_rope_apply_full(k, grid_sizes, freqs, rope_frame_indices).type_as(v)
            else:
                roped_q = rope_apply_full(q, grid_sizes, freqs).type_as(v)
                roped_k = rope_apply_full(k, grid_sizes, freqs).type_as(v)
```

In `CausalDiTBlock.forward()` — pass through `rope_frame_indices`. Find where `self.self_attn(...)` is called (around line 689) and add the parameter. Also update the `CausalDiTBlock.forward()` signature to accept it.

In `_forward_full_sequence` — compute and pass `rope_frame_indices` in kwargs (line 1431):
```python
        # Compute dynamic RoPE indices if enabled
        rope_frame_indices = None
        if self.use_dynamic_rope:
            attn_local = getattr(self, '_current_local_attn_size', self.local_attn_size)
            attn_sink = getattr(self, '_current_sink_size', self.sink_size)
            rope_frame_indices = compute_dynamic_rope_indices(
                f, self.chunk_size, attn_local, attn_sink,
            ).to(device)

        # ...in the block loop kwargs:
            kwargs = dict(
                e=t_mod,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=None,
                block_mask=self.block_mask,
                rope_frame_indices=rope_frame_indices,
            )
```

- [ ] **Step 3: Update `CausalDiTBlock` to pass `rope_frame_indices` through**

Find `CausalDiTBlock.forward()` (around line 655). Add `rope_frame_indices=None` to its signature and pass it to `self.self_attn(...)`:

```python
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Tuple[torch.Tensor, ...],
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        block_mask=None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        crossattn_cache: Optional[Dict] = None,
        current_start: int = 0,
        store_kv: bool = True,
        cache_local_end_override: Optional[int] = None,
        rope_frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
```

And in the self_attn call:
```python
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
            rope_frame_indices=rope_frame_indices,
        )
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py -v 2>&1 | tail -30`
Expected: All tests PASS (mask tests + RoPE tests + integration test).

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_sliding_window_attention.py
git commit -m "feat: wire sliding window mask + dynamic RoPE into full-sequence forward path"
```

---

## Task 4: Stochastic Attention Config Sampling

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:770-930` (constructor), `1337-1458` (`_forward_full_sequence`)
- Modify: `tests/test_sliding_window_attention.py`

### Explanation

Add a `stochastic_attn_configs` parameter to the constructor. When set (list of dicts), each forward pass in training mode samples one config and rebuilds the block mask + recomputes dynamic RoPE indices. During eval, use the instance-level defaults.

- [ ] **Step 1: Write stochastic sampling test**

Append to `tests/test_sliding_window_attention.py`:

```python
class TestStochasticAttnConfig:
    """Test stochastic attention config sampling."""

    def test_config_sampling(self):
        """Verify that different configs produce different masks."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = object.__new__(CausalOmniAvatarWan)
        model.chunk_size = 3
        frame_seqlen = 4

        mask_full = model._build_block_mask(
            device, 21, frame_seqlen, 3, local_attn_size=-1, sink_size=0,
        )
        mask_window = model._build_block_mask(
            device, 21, frame_seqlen, 3, local_attn_size=6, sink_size=0,
        )

        # The masks should differ — window mask blocks more tokens
        total_len = 21 * frame_seqlen
        padded_len = math.ceil(total_len / 128) * 128
        dense_full = _dense_mask_from_block_mask(mask_full, padded_len, padded_len)
        dense_window = _dense_mask_from_block_mask(mask_window, padded_len, padded_len)
        # Window mask should have strictly fewer True entries
        assert dense_window[:total_len, :total_len].sum() < dense_full[:total_len, :total_len].sum()

    def test_sample_attn_config(self):
        """Verify _sample_attn_config returns valid configs."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        model = object.__new__(CausalOmniAvatarWan)
        model.local_attn_size = -1
        model.sink_size = 0
        model._stochastic_attn_configs = [
            {"local_attn_size": -1, "sink_size": 0, "weight": 0.5},
            {"local_attn_size": 6, "sink_size": 1, "weight": 0.5},
        ]
        # Sample many times — should get both configs
        configs_seen = set()
        for _ in range(100):
            cfg = model._sample_attn_config()
            configs_seen.add((cfg["local_attn_size"], cfg["sink_size"]))
        assert len(configs_seen) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestStochasticAttnConfig -v 2>&1 | head -20`
Expected: FAIL — `_sample_attn_config` doesn't exist.

- [ ] **Step 3: Add `stochastic_attn_configs` to constructor and `_sample_attn_config` method**

In the `CausalOmniAvatarWan.__init__` (around line 795), add parameter and storage:

Add to constructor signature:
```python
        stochastic_attn_configs: Optional[list] = None,
```

Add after `self._use_gradient_checkpointing` assignment (line 926):
```python
        self._stochastic_attn_configs = stochastic_attn_configs
```

Add new method after `clear_caches` (after line 1331):
```python
    def _sample_attn_config(self) -> dict:
        """Sample an attention config from stochastic_attn_configs.

        Returns dict with 'local_attn_size' and 'sink_size' keys.
        Falls back to instance defaults if no stochastic configs.
        """
        if not self._stochastic_attn_configs:
            return {
                "local_attn_size": self.local_attn_size,
                "sink_size": self.sink_size,
            }

        import random
        weights = [c.get("weight", 1.0) for c in self._stochastic_attn_configs]
        chosen = random.choices(self._stochastic_attn_configs, weights=weights, k=1)[0]
        return {
            "local_attn_size": chosen.get("local_attn_size", self.local_attn_size),
            "sink_size": chosen.get("sink_size", self.sink_size),
        }
```

- [ ] **Step 4: Modify `_forward_full_sequence` to rebuild mask when stochastic**

Replace the mask-building block in `_forward_full_sequence` (the section modified in Task 3):

```python
        # Determine attention config (stochastic during training, defaults during eval)
        if self.training and self._stochastic_attn_configs:
            attn_cfg = self._sample_attn_config()
            attn_local = attn_cfg["local_attn_size"]
            attn_sink = attn_cfg["sink_size"]
            # Rebuild mask every forward pass (stochastic)
            if FLEX_ATTENTION_AVAILABLE:
                frame_seqlen = h * w
                self.block_mask = self._build_block_mask(
                    device, f, frame_seqlen, self.chunk_size,
                    local_attn_size=attn_local,
                    sink_size=attn_sink,
                )
        else:
            attn_local = self.local_attn_size
            attn_sink = self.sink_size
            # Cache mask (non-stochastic path)
            if self.block_mask is None and FLEX_ATTENTION_AVAILABLE:
                frame_seqlen = h * w
                self.block_mask = self._build_block_mask(
                    device, f, frame_seqlen, self.chunk_size,
                    local_attn_size=attn_local,
                    sink_size=attn_sink,
                )

        # Compute dynamic RoPE indices (depends on current attn config)
        rope_frame_indices = None
        if self.use_dynamic_rope:
            rope_frame_indices = compute_dynamic_rope_indices(
                f, self.chunk_size, attn_local, attn_sink,
            ).to(device)
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py -v 2>&1 | tail -30`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_sliding_window_attention.py
git commit -m "feat: add stochastic attention config sampling for training"
```

---

## Task 5: Experiment Configs

**Files:**
- Modify: `fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py:40-52`
- Modify: `fastgen/configs/experiments/OmniAvatar/config_sf.py:66-78`

### Explanation

Wire the new params into experiment configs. For the initial version, set conservative defaults (no window = unchanged behavior) and add commented-out examples for sliding window experiments.

- [ ] **Step 1: Update DF config**

In `config_df_shift_5.py`, modify the student config (lines 40-52):

```python
CausalOmniAvatar_V2V_1_3B_Config: dict = L(CausalOmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT,
    net_pred_type="flow",
    schedule_type="rf",
    # Sliding window attention (default: full causal, no window)
    local_attn_size=-1,
    sink_size=0,
    use_dynamic_rope=False,
    # Stochastic attention configs (uncomment to enable):
    # stochastic_attn_configs=[
    #     {"local_attn_size": -1, "sink_size": 0, "weight": 0.25},   # full causal
    #     {"local_attn_size": 6,  "sink_size": 0, "weight": 0.25},   # tight window
    #     {"local_attn_size": 9,  "sink_size": 0, "weight": 0.25},   # medium window
    #     {"local_attn_size": 7,  "sink_size": 1, "weight": 0.25},   # window + sink
    # ],
)
```

- [ ] **Step 2: Update SF config**

In `config_sf.py`, modify the student config (lines 66-78):

```python
CausalOmniAvatar_V2V_1_3B_Student: dict = L(CausalOmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT,
    net_pred_type="flow",
    schedule_type="rf",
    # Sliding window attention — should match DF training config
    local_attn_size=-1,
    sink_size=0,
    use_dynamic_rope=False,
)
```

- [ ] **Step 3: Commit**

```bash
git add fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py fastgen/configs/experiments/OmniAvatar/config_sf.py
git commit -m "feat: wire sliding window attention params into DF and SF experiment configs"
```

---

## Task 6: Comprehensive Mask and RoPE Visualization/Verification

**Files:**
- Modify: `tests/test_sliding_window_attention.py`

### Explanation

Beyond unit tests, we need visual/numerical verification that the masks and RoPE indices look correct for various configurations. These tests print detailed diagnostics.

- [ ] **Step 1: Add comprehensive verification tests**

Append to `tests/test_sliding_window_attention.py`:

```python
class TestMaskVisualization:
    """Detailed verification of mask patterns — print for visual inspection."""

    NUM_FRAMES = 21
    CHUNK_SIZE = 3
    FRAME_SEQLEN = 4

    def _build(self, local_attn_size=-1, sink_size=0):
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
        model = object.__new__(CausalOmniAvatarWan)
        model.chunk_size = self.CHUNK_SIZE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = CausalOmniAvatarWan._build_block_mask(
            model, device, self.NUM_FRAMES, self.FRAME_SEQLEN,
            chunk_size=self.CHUNK_SIZE,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        total_len = self.NUM_FRAMES * self.FRAME_SEQLEN
        padded_len = math.ceil(total_len / 128) * 128
        dense = _dense_mask_from_block_mask(mask, padded_len, padded_len)
        return dense[:total_len, :total_len]

    def _frame_mask(self, dense):
        """Collapse token-level mask to frame-level: True if ANY token pair is visible."""
        F = self.NUM_FRAMES
        S = self.FRAME_SEQLEN
        frame_mask = torch.zeros(F, F, dtype=torch.bool)
        for qf in range(F):
            for kf in range(F):
                block = dense[qf*S:(qf+1)*S, kf*S:(kf+1)*S]
                frame_mask[qf, kf] = block.any()
        return frame_mask

    def _print_frame_mask(self, frame_mask, label):
        F = frame_mask.shape[0]
        print(f"\n{'='*60}")
        print(f"Frame-level mask: {label}")
        print(f"{'='*60}")
        header = "Q\\K  " + "".join(f"{k:3d}" for k in range(F))
        print(header)
        for q in range(F):
            row = f"{q:3d}  " + "".join(
                " ██" if frame_mask[q, k] else " ··" for k in range(F)
            )
            print(row)

    def test_visualize_full_causal(self):
        """Print full causal mask for visual verification."""
        dense = self._build(local_attn_size=-1, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Full Causal (local_attn_size=-1, sink=0)")

        # Verify: lower-triangular at chunk level
        chunks = _chunk_boundaries(self.NUM_FRAMES, self.CHUNK_SIZE)
        for qi, (qs, qe) in enumerate(chunks):
            for ki, (ks, ke) in enumerate(chunks):
                for qf in range(qs, qe):
                    for kf in range(ks, ke):
                        if ki <= qi:
                            assert fm[qf, kf], f"frame {qf} should see frame {kf}"
                        else:
                            assert not fm[qf, kf], f"frame {qf} should NOT see frame {kf}"

    def test_visualize_window_6(self):
        """Print window=6 mask."""
        dense = self._build(local_attn_size=6, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=6 (2 chunks visible)")

        # Chunk 4 (frames 12-14) should see chunks 3-4 (frames 9-14) but NOT chunk 2 (frames 6-8)
        assert fm[12, 9].item() is True
        assert fm[12, 8].item() is False

    def test_visualize_window_7_sink_1(self):
        """Print window=7 + sink=1 mask."""
        dense = self._build(local_attn_size=7, sink_size=1)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=7, Sink=1")

        # Frame 0 (sink) visible from everywhere
        for qf in range(self.NUM_FRAMES):
            assert fm[qf, 0].item() is True, f"frame {qf} should see sink frame 0"
        # Frame 1 NOT visible from chunk 6 (frames 18-20)
        assert fm[18, 1].item() is False

    def test_visualize_window_9_sink_0(self):
        """Print window=9 mask — 3 chunks visible."""
        dense = self._build(local_attn_size=9, sink_size=0)
        fm = self._frame_mask(dense)
        self._print_frame_mask(fm, "Window=9 (3 chunks visible)")

    def test_rope_indices_all_configs(self):
        """Print RoPE indices for multiple configs."""
        from fastgen.networks.OmniAvatar.network_causal import compute_dynamic_rope_indices

        configs = [
            (-1, 0, "Full causal"),
            (6, 0, "Window=6"),
            (7, 1, "Window=7, Sink=1"),
            (9, 0, "Window=9"),
        ]
        for local, sink, label in configs:
            indices = compute_dynamic_rope_indices(21, 3, local, sink)
            print(f"\nRoPE indices ({label}): {indices.tolist()}")
            if local > 0:
                assert indices.max().item() <= local - 1, f"Max index {indices.max()} exceeds {local-1}"
```

- [ ] **Step 2: Run verification tests with output**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestMaskVisualization -v -s 2>&1`
Expected: All tests PASS and printed mask grids show correct patterns:
- Full causal: lower-triangular at chunk level
- Window=6: banded diagonal (2 chunks wide)
- Window=7+Sink=1: banded + first column always filled
- Window=9: banded (3 chunks wide)

Visually inspect the printed masks and RoPE indices. Verify no unexpected patterns.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sliding_window_attention.py
git commit -m "test: comprehensive mask visualization and RoPE index verification"
```

---

## Task 7: End-to-End Smoke Test

**Files:**
- Modify: `tests/test_sliding_window_attention.py`

### Explanation

Run a minimal DF training step with sliding window enabled to verify no crashes or shape mismatches through the full forward + backward path.

- [ ] **Step 1: Write end-to-end smoke test**

Append to `tests/test_sliding_window_attention.py`:

```python
class TestEndToEnd:
    """End-to-end smoke tests: full forward + backward with sliding window."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_backward_with_window(self):
        """Full forward + backward pass with local_attn_size=6."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

        # Build a tiny model (1.3B architecture but we just need it to run)
        # This will be slow on first run due to weight loading — skip weights
        model = CausalOmniAvatarWan(
            model_size="1.3B",
            in_dim=65,
            mode="v2v",
            chunk_size=3,
            total_num_frames=9,  # Small for speed: 3 chunks
            use_audio=False,  # Skip audio for simplicity
            local_attn_size=6,
            sink_size=0,
            use_dynamic_rope=True,
            base_model_paths=None,  # Skip loading weights
            omniavatar_ckpt_path=None,
            disable_grad_ckpt=True,  # Skip checkpointing for small test
        )
        model = model.cuda().to(torch.bfloat16)
        model.train()

        B, C, T, H, W = 1, 16, 9, 16, 16  # Tiny spatial dims
        x_t = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
        t = torch.rand(B, T, device="cuda", dtype=torch.bfloat16)  # per-frame timesteps

        condition = {
            "text_embeds": torch.randn(B, 512, 4096, device="cuda", dtype=torch.bfloat16),
        }

        # Forward
        out = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")
        assert out.shape == (B, C, T, H, W), f"Output shape mismatch: {out.shape}"

        # Backward
        loss = out.sum()
        loss.backward()

        # Verify gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_backward_with_stochastic(self):
        """Full forward + backward pass with stochastic attention configs."""
        from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

        model = CausalOmniAvatarWan(
            model_size="1.3B",
            in_dim=65,
            mode="v2v",
            chunk_size=3,
            total_num_frames=9,
            use_audio=False,
            local_attn_size=-1,
            sink_size=0,
            use_dynamic_rope=True,
            base_model_paths=None,
            omniavatar_ckpt_path=None,
            disable_grad_ckpt=True,
            stochastic_attn_configs=[
                {"local_attn_size": -1, "sink_size": 0, "weight": 0.5},
                {"local_attn_size": 6, "sink_size": 0, "weight": 0.5},
            ],
        )
        model = model.cuda().to(torch.bfloat16)
        model.train()

        B, C, T, H, W = 1, 16, 9, 16, 16
        x_t = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
        t = torch.rand(B, T, device="cuda", dtype=torch.bfloat16)

        condition = {
            "text_embeds": torch.randn(B, 512, 4096, device="cuda", dtype=torch.bfloat16),
        }

        # Run forward twice — should use different masks
        out1 = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")
        model.block_mask = None  # Force rebuild
        out2 = model(x_t, t, condition=condition, is_ar=False, fwd_pred_type="x0")

        assert out1.shape == out2.shape == (B, C, T, H, W)
        # Outputs may differ due to different masks (stochastic)
        loss = out2.sum()
        loss.backward()
```

- [ ] **Step 2: Run end-to-end tests**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/test_sliding_window_attention.py::TestEndToEnd -v -s 2>&1 | tail -30`
Expected: Both tests PASS without OOM or shape errors.

Note: The first run may be slow due to model construction. If `base_model_paths=None` causes issues in the constructor, we may need to skip weight loading more carefully. Adjust as needed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sliding_window_attention.py
git commit -m "test: end-to-end smoke tests for sliding window + stochastic attention"
```

---

## Summary of Changes

| Component | What Changes |
|-----------|-------------|
| `_build_block_mask` | Accepts `local_attn_size` and `sink_size`, builds `starts[]` array + sink term |
| `compute_dynamic_rope_indices` | New function: per-frame RoPE indices based on window position |
| `dynamic_rope_apply_full` | New function: applies RoPE with arbitrary per-frame indices |
| `CausalSelfAttention.forward` | Accepts optional `rope_frame_indices`, uses dynamic RoPE when provided |
| `CausalDiTBlock.forward` | Passes `rope_frame_indices` through to self-attention |
| `_forward_full_sequence` | Samples stochastic config, rebuilds mask, computes dynamic RoPE, passes to blocks |
| `CausalOmniAvatarWan.__init__` | Accepts `stochastic_attn_configs` param |
| `_sample_attn_config` | New method: weighted random sampling from config list |
| Experiment configs | Wire `local_attn_size`, `sink_size`, `use_dynamic_rope` with defaults |

## Future Work (Not in This Plan)
- **Lookahead sink**: Attend to future frames as identity anchors, with RoPE-based future positioning
- **Per-sample stochastic configs**: Different configs per sample in a batch (requires per-sample block masks)
- **Sliding window in AR mode verification**: AR mode already supports these params — verify consistency with full-sequence training
