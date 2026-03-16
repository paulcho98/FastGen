#!/usr/bin/env python3
"""T2V Self-Forcing memory baseline using the known-working CausalWan pipeline.

Runs the same Self-Forcing rollout + VSD loss as OmniAvatar but with the T2V
1.3B→1.3B pipeline (no audio, 16ch input) to establish a memory baseline.

Comparison: OmniAvatar adds audio conditioning + 65ch input (vs 16ch).
If OmniAvatar uses significantly more memory, the delta indicates overhead.

Run: CUDA_VISIBLE_DEVICES=2 python scripts/test_t2v_memory_baseline.py
"""

import gc
import time

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# T2V uses 480p: [16, 21, 60, 104] = 6240 tokens/frame
# OmniAvatar uses 512x512: [16, 21, 64, 64] = 4096 tokens/frame
# Use OmniAvatar's resolution for fair comparison
INPUT_SHAPE_T2V = [16, 21, 64, 64]  # Same spatial res as OmniAvatar
INPUT_SHAPE_OMNI = [16, 21, 64, 64]


def gpu_mem():
    return torch.cuda.memory_allocated(DEVICE) / 1024**3


def gpu_peak():
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**3


def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def run_t2v_baseline():
    """Run T2V 1.3B SF pipeline — 3x 1.3B models, synthetic data."""
    print_header("T2V 1.3B Self-Forcing Memory Baseline")
    torch.cuda.reset_peak_memory_stats(DEVICE)

    from fastgen.networks.Wan.network import Wan
    from fastgen.networks.Wan.network_causal import CausalWan

    B = 1
    C, T, H, W = INPUT_SHAPE_T2V

    # --- Load 3 models ---
    print("  Loading 1.3B teacher (bidirectional)...")
    teacher = Wan(
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        net_pred_type="flow",
        schedule_type="rf",
    )
    teacher = teacher.to(device=DEVICE, dtype=DTYPE)
    teacher.eval().requires_grad_(False)
    mem_teacher = gpu_mem()
    print(f"  Teacher: {mem_teacher:.1f} GB")

    print("  Loading 1.3B fake_score (bidirectional)...")
    fake_score = Wan(
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        net_pred_type="flow",
        schedule_type="rf",
    )
    fake_score = fake_score.to(device=DEVICE, dtype=DTYPE)
    fake_score.eval().requires_grad_(False)
    mem_fs = gpu_mem()
    print(f"  Fake score: +{mem_fs - mem_teacher:.1f} GB (total: {mem_fs:.1f} GB)")

    print("  Loading 1.3B student (causal)...")
    student = CausalWan(
        model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        chunk_size=3,
        total_num_frames=T,
        net_pred_type="flow",
        schedule_type="rf",
    )
    student = student.to(device=DEVICE, dtype=DTYPE)
    student.train()
    mem_all = gpu_mem()
    print(f"  Student: +{mem_all - mem_fs:.1f} GB (total: {mem_all:.1f} GB)")

    # --- Synthetic data ---
    condition = torch.randn(B, 512, 4096, device=DEVICE, dtype=DTYPE)
    real_data = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)

    # --- Self-Forcing rollout ---
    print(f"\n  Running Self-Forcing rollout ({T // student.chunk_size} chunks)...")
    student.clear_caches()

    chunk_size = student.chunk_size
    num_blocks = T // chunk_size
    noise = torch.randn_like(real_data)
    noise = student.noise_scheduler.latents(noise=noise)

    t_list = torch.tensor([0.999, 0.937, 0.833, 0.624, 0.0], device=DEVICE, dtype=torch.float64)
    denoised_blocks = []

    for block_idx in range(num_blocks):
        cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        exit_step = torch.randint(0, 4, (1,)).item()
        for step in range(len(t_list) - 1):
            t_cur = t_list[step].expand(B)
            if step == exit_step:
                enable_grad = block_idx >= num_blocks - 2
                with torch.set_grad_enabled(enable_grad):
                    x0_pred = student(
                        noisy_input, t_cur, condition=condition,
                        fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                        store_kv=False, is_ar=True,
                    )
                break
            else:
                with torch.no_grad():
                    x0_pred = student(
                        noisy_input, t_cur, condition=condition,
                        fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                        store_kv=False, is_ar=True,
                    )
                t_next = t_list[step + 1].expand(B)
                eps_infer = torch.randn_like(x0_pred)
                noisy_input = student.noise_scheduler.forward_process(x0_pred, eps_infer, t_next)

        denoised_blocks.append(x0_pred)

        with torch.no_grad():
            t_zero = torch.zeros(B, device=DEVICE, dtype=DTYPE)
            _ = student(
                x0_pred, t_zero, condition=condition,
                fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                store_kv=True, is_ar=True,
            )

    gen_data = torch.cat(denoised_blocks, dim=2)
    mem_rollout = gpu_mem()
    peak_rollout = gpu_peak()
    print(f"  Rollout: {gen_data.shape}")
    print(f"  Memory after rollout: {mem_rollout:.1f} GB (peak: {peak_rollout:.1f} GB)")

    # --- VSD loss ---
    print("  Computing VSD loss...")
    t_vsd = student.noise_scheduler.sample_t(B, time_dist_type="shifted", min_t=0.001, max_t=0.999, device=DEVICE)
    eps_vsd = torch.randn_like(gen_data)
    perturbed = student.noise_scheduler.forward_process(gen_data, eps_vsd, t_vsd)

    with torch.no_grad():
        teacher_x0 = teacher(perturbed, t_vsd, condition=condition, fwd_pred_type="x0")
        fake_x0 = fake_score(perturbed, t_vsd, condition=condition, fwd_pred_type="x0")

    w = 1.0 / (torch.abs(gen_data - teacher_x0) + 1e-6)
    vsd_grad = (fake_x0 - teacher_x0) * w
    pseudo_target = gen_data - vsd_grad
    vsd_loss = 0.5 * F.mse_loss(gen_data, pseudo_target.detach(), reduction="mean")
    vsd_loss.backward()

    grad_count = sum(1 for p in student.parameters() if p.grad is not None)
    mem_after_backward = gpu_mem()
    peak_after_backward = gpu_peak()
    print(f"  VSD loss: {vsd_loss.item():.6f}")
    print(f"  Student params with gradients: {grad_count}")
    print(f"  Memory after backward: {mem_after_backward:.1f} GB (peak: {peak_after_backward:.1f} GB)")

    student.clear_caches()

    print(f"\n  === T2V MEMORY SUMMARY ===")
    print(f"  3 models loaded:       {mem_all:.1f} GB")
    print(f"  After rollout:         {mem_rollout:.1f} GB (peak: {peak_rollout:.1f} GB)")
    print(f"  After VSD backward:    {mem_after_backward:.1f} GB (peak: {peak_after_backward:.1f} GB)")

    del teacher, fake_score, student
    gc.collect()
    torch.cuda.empty_cache()


def run_omniavatar_baseline():
    """Run OmniAvatar 1.3B SF pipeline — 3x 1.3B models, synthetic data."""
    print_header("OmniAvatar 1.3B Self-Forcing Memory Baseline (synthetic data)")
    torch.cuda.reset_peak_memory_stats(DEVICE)

    from fastgen.networks.OmniAvatar.network import OmniAvatarWan
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    OMNIAVATAR_ROOT = "/home/work/.local/OmniAvatar"
    BASE_1_3B = f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

    B = 1
    C, T, H, W = INPUT_SHAPE_OMNI

    # --- Load 3 models ---
    print("  Loading 1.3B teacher (bidirectional, V2V 65ch)...")
    teacher = OmniAvatarWan(
        model_size="1.3B", in_dim=65, mode="v2v",
        use_audio=True, audio_hidden_size=32,
        base_model_paths=BASE_1_3B, omniavatar_ckpt_path=None,
        net_pred_type="flow", schedule_type="rf",
    )
    teacher = teacher.to(device=DEVICE, dtype=DTYPE)
    teacher.eval().requires_grad_(False)
    mem_teacher = gpu_mem()
    print(f"  Teacher: {mem_teacher:.1f} GB")

    print("  Loading 1.3B fake_score (bidirectional, V2V 65ch)...")
    fake_score = OmniAvatarWan(
        model_size="1.3B", in_dim=65, mode="v2v",
        use_audio=True, audio_hidden_size=32,
        base_model_paths=BASE_1_3B, omniavatar_ckpt_path=None,
        net_pred_type="flow", schedule_type="rf",
    )
    fake_score = fake_score.to(device=DEVICE, dtype=DTYPE)
    fake_score.eval().requires_grad_(False)
    mem_fs = gpu_mem()
    print(f"  Fake score: +{mem_fs - mem_teacher:.1f} GB (total: {mem_fs:.1f} GB)")

    print("  Loading 1.3B student (causal, V2V 65ch)...")
    student = CausalOmniAvatarWan(
        model_size="1.3B", in_dim=65, mode="v2v",
        use_audio=True, audio_hidden_size=32,
        chunk_size=3, total_num_frames=T,
        base_model_paths=BASE_1_3B, omniavatar_ckpt_path=None,
        net_pred_type="flow", schedule_type="rf",
    )
    student = student.to(device=DEVICE, dtype=DTYPE)
    student.train()
    mem_all = gpu_mem()
    print(f"  Student: +{mem_all - mem_fs:.1f} GB (total: {mem_all:.1f} GB)")

    # --- Synthetic data ---
    condition = {
        "text_embeds": torch.randn(B, 512, 4096, device=DEVICE, dtype=DTYPE),
        "audio_emb": torch.randn(B, 81, 10752, device=DEVICE, dtype=DTYPE),
        "ref_latent": torch.randn(B, 16, 1, H, W, device=DEVICE, dtype=DTYPE),
        "mask": torch.ones(H, W, device=DEVICE, dtype=torch.float32),
        "masked_video": torch.randn(B, 16, T, H, W, device=DEVICE, dtype=DTYPE),
        "ref_sequence": torch.randn(B, 16, T, H, W, device=DEVICE, dtype=DTYPE),
    }
    neg_condition = {
        "text_embeds": torch.zeros(B, 512, 4096, device=DEVICE, dtype=DTYPE),
        "audio_emb": torch.zeros(B, 81, 10752, device=DEVICE, dtype=DTYPE),
        "ref_latent": condition["ref_latent"],
        "mask": condition["mask"],
        "masked_video": condition["masked_video"],
        "ref_sequence": condition["ref_sequence"],
    }
    real_data = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)

    # --- Self-Forcing rollout ---
    print(f"\n  Running Self-Forcing rollout ({T // student.chunk_size} chunks)...")
    student.clear_caches()

    chunk_size = student.chunk_size
    num_blocks = T // chunk_size
    noise = torch.randn_like(real_data)
    noise = student.noise_scheduler.latents(noise=noise)

    t_list = torch.tensor([0.999, 0.937, 0.833, 0.624, 0.0], device=DEVICE, dtype=torch.float64)
    denoised_blocks = []

    for block_idx in range(num_blocks):
        cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        exit_step = torch.randint(0, 4, (1,)).item()
        for step in range(len(t_list) - 1):
            t_cur = t_list[step].expand(B)
            if step == exit_step:
                enable_grad = block_idx >= num_blocks - 2
                with torch.set_grad_enabled(enable_grad):
                    x0_pred = student(
                        noisy_input, t_cur, condition=condition,
                        fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                        store_kv=False, is_ar=True,
                    )
                break
            else:
                with torch.no_grad():
                    x0_pred = student(
                        noisy_input, t_cur, condition=condition,
                        fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                        store_kv=False, is_ar=True,
                    )
                t_next = t_list[step + 1].expand(B)
                eps_infer = torch.randn_like(x0_pred)
                noisy_input = student.noise_scheduler.forward_process(x0_pred, eps_infer, t_next)

        denoised_blocks.append(x0_pred)

        with torch.no_grad():
            t_zero = torch.zeros(B, device=DEVICE, dtype=DTYPE)
            _ = student(
                x0_pred, t_zero, condition=condition,
                fwd_pred_type="x0", cur_start_frame=cur_start_frame,
                store_kv=True, is_ar=True,
            )

    gen_data = torch.cat(denoised_blocks, dim=2)
    mem_rollout = gpu_mem()
    peak_rollout = gpu_peak()
    print(f"  Rollout: {gen_data.shape}")
    print(f"  Memory after rollout: {mem_rollout:.1f} GB (peak: {peak_rollout:.1f} GB)")

    # --- VSD loss ---
    print("  Computing VSD loss...")
    t_vsd = student.noise_scheduler.sample_t(B, time_dist_type="shifted", min_t=0.001, max_t=0.999, device=DEVICE)
    eps_vsd = torch.randn_like(gen_data)
    perturbed = student.noise_scheduler.forward_process(gen_data, eps_vsd, t_vsd)

    with torch.no_grad():
        teacher_x0 = teacher(perturbed, t_vsd, condition=condition, fwd_pred_type="x0")
        fake_x0 = fake_score(perturbed, t_vsd, condition=condition, fwd_pred_type="x0")

    w = 1.0 / (torch.abs(gen_data - teacher_x0) + 1e-6)
    vsd_grad = (fake_x0 - teacher_x0) * w
    pseudo_target = gen_data - vsd_grad
    vsd_loss = 0.5 * F.mse_loss(gen_data, pseudo_target.detach(), reduction="mean")
    vsd_loss.backward()

    grad_count = sum(1 for p in student.parameters() if p.grad is not None)
    mem_after_backward = gpu_mem()
    peak_after_backward = gpu_peak()
    print(f"  VSD loss: {vsd_loss.item():.6f}")
    print(f"  Student params with gradients: {grad_count}")
    print(f"  Memory after backward: {mem_after_backward:.1f} GB (peak: {peak_after_backward:.1f} GB)")

    student.clear_caches()

    print(f"\n  === OMNIAVATAR MEMORY SUMMARY ===")
    print(f"  3 models loaded:       {mem_all:.1f} GB")
    print(f"  After rollout:         {mem_rollout:.1f} GB (peak: {peak_rollout:.1f} GB)")
    print(f"  After VSD backward:    {mem_after_backward:.1f} GB (peak: {peak_after_backward:.1f} GB)")

    del teacher, fake_score, student
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Free: {torch.cuda.mem_get_info(DEVICE)[0]/1024**3:.1f} GB")

    # Run T2V baseline first, then OmniAvatar
    run_t2v_baseline()

    print("\n" + "=" * 70)
    print("  Waiting 5s for GPU memory to settle...")
    print("=" * 70)
    gc.collect()
    torch.cuda.empty_cache()
    import time; time.sleep(5)

    run_omniavatar_baseline()

    print_header("COMPARISON")
    print("  Check the summaries above for memory deltas.")
    print("  The difference = audio adapter + 65ch vs 16ch overhead.")
