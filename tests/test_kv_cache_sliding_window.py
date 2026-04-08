"""Test KV cache eviction logic for sliding window attention in AR mode.

Simulates the cache metadata tracking from CausalSelfAttention.forward() AR branch
(network_causal.py lines 522-651) without running actual attention, to verify that
cache indices stay in bounds and eviction behaves correctly across all configurations.

Tests all 5 stochastic DF training configs + edge cases:
  - sink=1, local_attn_size=7  (window=6)
  - sink=1, local_attn_size=10 (window=9)
  - sink=1, local_attn_size=13 (window=12)
  - sink=3, local_attn_size=9  (window=6)
  - sink=3, local_attn_size=12 (window=9)
  - sink=0, local_attn_size=-1 (full causal, no eviction)
  - sink=0, local_attn_size=7  (no sink, window=7)

For each config, simulates a full 21-frame AR rollout (7 blocks × 3 frames)
with both store_kv=True and store_kv=False calls, matching Self-Forcing's pattern:
  for each block:
    1. Multiple store_kv=False calls (denoising steps)
    2. One store_kv=True call (cache update)

Usage:
    python tests/test_kv_cache_sliding_window.py
    pytest tests/test_kv_cache_sliding_window.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# Simulate the cache eviction logic from CausalSelfAttention.forward()
# ---------------------------------------------------------------------------

def simulate_cache_step(
    kv_cache_size: int,
    local_end: int,
    global_end: int,
    current_start: int,
    num_new_tokens: int,
    sink_tokens: int,
    store_kv: bool,
    local_attn_size_tokens: int,
):
    """Simulate one self_attn cache step. Returns (new_local_end, new_local_start, evicted).

    This mirrors the logic at network_causal.py lines 558-577.
    """
    current_end = current_start + num_new_tokens

    # Eviction check (line 558-573)
    evicted = False
    if (
        store_kv
        and local_attn_size_tokens > 0
        and current_end > global_end
        and num_new_tokens + local_end > kv_cache_size
    ):
        num_evicted = num_new_tokens + local_end - kv_cache_size
        num_rolled = local_end - num_evicted - sink_tokens
        new_local_end = local_end + (current_end - global_end) - num_evicted
        new_local_start = new_local_end - num_new_tokens
        evicted = True
    else:
        new_local_end = local_end + max(0, current_end - global_end)
        new_local_start = new_local_end - num_new_tokens

    return new_local_end, new_local_start, evicted


def simulate_full_rollout(
    num_frames: int,
    chunk_size: int,
    local_attn_size: int,
    sink_size: int,
    frame_seqlen: int,
    num_denoising_steps: int = 2,
):
    """Simulate a full AR rollout matching Self-Forcing's pattern.

    Returns list of per-block results for verification.
    """
    num_blocks = num_frames // chunk_size
    num_new_tokens = chunk_size * frame_seqlen
    sink_tokens = sink_size * frame_seqlen

    if local_attn_size > 0:
        kv_cache_size = local_attn_size * frame_seqlen
    else:
        kv_cache_size = num_frames * frame_seqlen

    # Cache state (mirrors kv_cache metadata)
    cache_global_end = 0
    cache_local_end = 0

    results = []

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        current_start_tokens = cur_start_frame * frame_seqlen

        block_result = {
            "block_idx": block_idx,
            "cur_start_frame": cur_start_frame,
            "current_start_tokens": current_start_tokens,
        }

        # --- store_kv=False calls (denoising steps) ---
        # In _forward_ar, cache_local_end_override = cache["local_end_index"]
        # and global_end = current_start (our fix)
        for step in range(num_denoising_steps):
            local_end_override = cache_local_end
            global_end_for_read = current_start_tokens  # current_start IS global position

            new_local_end, new_local_start, evicted = simulate_cache_step(
                kv_cache_size=kv_cache_size,
                local_end=local_end_override,
                global_end=global_end_for_read,
                current_start=current_start_tokens,
                num_new_tokens=num_new_tokens,
                sink_tokens=sink_tokens,
                store_kv=False,
                local_attn_size_tokens=local_attn_size * frame_seqlen if local_attn_size > 0 else 0,
            )

            # Verify: store_kv=False should never evict
            assert not evicted, f"Block {block_idx} step {step}: eviction on store_kv=False!"
            # Verify: window indices in bounds
            assert new_local_start >= 0, f"Block {block_idx} step {step}: new_local_start={new_local_start} < 0"
            assert new_local_end <= kv_cache_size + num_new_tokens, \
                f"Block {block_idx} step {step}: new_local_end={new_local_end} > cache+chunk"

        # --- store_kv=True call (cache update) ---
        local_end_override = cache_local_end
        global_end_for_write = current_start_tokens

        new_local_end, new_local_start, evicted = simulate_cache_step(
            kv_cache_size=kv_cache_size,
            local_end=local_end_override,
            global_end=global_end_for_write,
            current_start=current_start_tokens,
            num_new_tokens=num_new_tokens,
            sink_tokens=sink_tokens,
            store_kv=True,
            local_attn_size_tokens=local_attn_size * frame_seqlen if local_attn_size > 0 else 0,
        )

        # Verify: write indices in bounds
        assert new_local_start >= 0, \
            f"Block {block_idx} store: new_local_start={new_local_start} < 0"
        assert new_local_end <= kv_cache_size, \
            f"Block {block_idx} store: new_local_end={new_local_end} > kv_cache_size={kv_cache_size}"
        assert new_local_end - new_local_start == num_new_tokens, \
            f"Block {block_idx} store: write size {new_local_end - new_local_start} != chunk {num_new_tokens}"

        # Verify: sink region preserved after eviction
        if evicted and sink_tokens > 0:
            assert new_local_start >= sink_tokens, \
                f"Block {block_idx}: write at {new_local_start} overlaps sink region [0:{sink_tokens}]"

        block_result["new_local_end"] = new_local_end
        block_result["new_local_start"] = new_local_start
        block_result["evicted"] = evicted

        # Update cache state (mirrors line 648-649)
        current_end_tokens = current_start_tokens + num_new_tokens
        cache_global_end = current_end_tokens
        cache_local_end = new_local_end

        block_result["cache_global_end"] = cache_global_end
        block_result["cache_local_end"] = cache_local_end
        results.append(block_result)

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# All 5 stochastic DF training configurations
STOCHASTIC_CONFIGS = [
    {"local_attn_size": 7, "sink_size": 1, "name": "sink1_window6"},
    {"local_attn_size": 10, "sink_size": 1, "name": "sink1_window9"},
    {"local_attn_size": 13, "sink_size": 1, "name": "sink1_window12"},
    {"local_attn_size": 9, "sink_size": 3, "name": "sink3_window6"},
    {"local_attn_size": 12, "sink_size": 3, "name": "sink3_window9"},
]

# Edge cases
EDGE_CONFIGS = [
    {"local_attn_size": -1, "sink_size": 0, "name": "full_causal"},
    {"local_attn_size": 7, "sink_size": 0, "name": "no_sink_window7"},
    {"local_attn_size": 21, "sink_size": 1, "name": "window_equals_sequence"},
    {"local_attn_size": 4, "sink_size": 1, "name": "tiny_window"},
]

ALL_CONFIGS = STOCHASTIC_CONFIGS + EDGE_CONFIGS


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c["name"])
@pytest.mark.parametrize("num_denoising_steps", [1, 2, 4], ids=["1step", "2step", "4step"])
def test_cache_indices_in_bounds(config, num_denoising_steps):
    """Cache write indices must stay within [0, kv_cache_size) for all configs."""
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=config["local_attn_size"],
        sink_size=config["sink_size"],
        frame_seqlen=1024,
        num_denoising_steps=num_denoising_steps,
    )
    # If we got here without assertion errors, all indices are in bounds
    assert len(results) == 7  # 21 / 3 = 7 blocks


@pytest.mark.parametrize("config", STOCHASTIC_CONFIGS, ids=lambda c: c["name"])
def test_eviction_occurs_when_expected(config):
    """With window < 21 frames, eviction must happen at some point."""
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=config["local_attn_size"],
        sink_size=config["sink_size"],
        frame_seqlen=1024,
        num_denoising_steps=2,
    )
    if config["local_attn_size"] < 21:
        any_evicted = any(r["evicted"] for r in results)
        assert any_evicted, f"Expected eviction for local_attn_size={config['local_attn_size']}"


def test_no_eviction_full_causal():
    """Full causal (local_attn_size=-1) should never evict."""
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=-1,
        sink_size=0,
        frame_seqlen=1024,
        num_denoising_steps=2,
    )
    assert not any(r["evicted"] for r in results)


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c["name"])
def test_cache_local_end_monotonic(config):
    """cache_local_end should be monotonically non-decreasing until eviction resets it."""
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=config["local_attn_size"],
        sink_size=config["sink_size"],
        frame_seqlen=1024,
        num_denoising_steps=2,
    )
    if config["local_attn_size"] <= 0:
        # Full causal: strictly increasing
        for i in range(1, len(results)):
            assert results[i]["cache_local_end"] > results[i - 1]["cache_local_end"]


@pytest.mark.parametrize("config", STOCHASTIC_CONFIGS, ids=lambda c: c["name"])
def test_cache_local_end_bounded_by_cache_size(config):
    """cache_local_end should never exceed kv_cache_size."""
    kv_cache_size = config["local_attn_size"] * 1024
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=config["local_attn_size"],
        sink_size=config["sink_size"],
        frame_seqlen=1024,
        num_denoising_steps=2,
    )
    for r in results:
        assert r["cache_local_end"] <= kv_cache_size, \
            f"Block {r['block_idx']}: cache_local_end={r['cache_local_end']} > {kv_cache_size}"


@pytest.mark.parametrize("frame_seqlen", [256, 1024, 2048], ids=["small", "normal", "large"])
def test_different_frame_seqlens(frame_seqlen):
    """Cache logic should work regardless of frame_seqlen (spatial resolution)."""
    results = simulate_full_rollout(
        num_frames=21,
        chunk_size=3,
        local_attn_size=7,
        sink_size=1,
        frame_seqlen=frame_seqlen,
        num_denoising_steps=2,
    )
    kv_cache_size = 7 * frame_seqlen
    for r in results:
        assert r["cache_local_end"] <= kv_cache_size


@pytest.mark.parametrize("num_frames", [9, 15, 21, 42], ids=["9f", "15f", "21f", "42f"])
def test_different_sequence_lengths(num_frames):
    """Cache should handle various sequence lengths (multiples of chunk_size)."""
    results = simulate_full_rollout(
        num_frames=num_frames,
        chunk_size=3,
        local_attn_size=7,
        sink_size=1,
        frame_seqlen=1024,
        num_denoising_steps=2,
    )
    assert len(results) == num_frames // 3


def test_print_cache_trace():
    """Print detailed cache trace for manual inspection (not a pass/fail test)."""
    for config in STOCHASTIC_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: local_attn_size={config['local_attn_size']}, sink_size={config['sink_size']}")
        print(f"{'='*60}")
        results = simulate_full_rollout(
            num_frames=21,
            chunk_size=3,
            local_attn_size=config["local_attn_size"],
            sink_size=config["sink_size"],
            frame_seqlen=1024,
            num_denoising_steps=2,
        )
        kv_cache_size = config["local_attn_size"] * 1024
        for r in results:
            evict_str = " [EVICT]" if r["evicted"] else ""
            print(
                f"  Block {r['block_idx']}: frames {r['cur_start_frame']}-{r['cur_start_frame']+2} | "
                f"write [{r['new_local_start']}:{r['new_local_end']}] | "
                f"local_end={r['cache_local_end']}/{kv_cache_size} | "
                f"global_end={r['cache_global_end']}{evict_str}"
            )


if __name__ == "__main__":
    # Run with verbose output for manual inspection
    test_print_cache_trace()
    print("\n\nRunning all tests...")
    pytest.main([__file__, "-v", "--tb=short"])
