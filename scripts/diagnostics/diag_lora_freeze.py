"""CPU-only diagnostic for the PEFT LoRA + selective-unfreeze freeze flow.

Constructs a 1.3B CausalOmniAvatarWan with no checkpoint loading, then
manually replays each step of the merge_lora=False path:
  1. PEFT inject_adapter_in_model
  2. _apply_unfreeze
  3. dtype casts (bf16, then fp32)

Reports trainable vs total parameter counts after each step so we can
identify exactly where the freeze gets undone (if anywhere).

Expected behavior in a working LoRA + selective-unfreeze setup:
  - After PEFT inject:   trainable ~= LoRA A/B count (small)
  - After _apply_unfreeze: trainable ~= LoRA + listed submodules (small)
  - After casts:         unchanged (casts preserve requires_grad)

If the smoke OOM was caused by PEFT not freezing the base, this
diagnostic will surface it before any GPU run.
"""

import sys
sys.path.insert(0, "/home/work/.local/hyunbin/FastGen-redmd")

import torch  # noqa: E402
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

from fastgen.networks.OmniAvatar.network_causal import (  # noqa: E402
    CausalOmniAvatarWan,
    LORA_TARGET_MODULES,
)


def report(model: torch.nn.Module, label: str) -> None:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    diff = n_total - n_trainable
    print(
        f"[{label:40s}] trainable {n_trainable/1e6:9.2f}M / "
        f"total {n_total/1e6:9.2f}M  frozen {diff/1e6:9.2f}M"
    )


def main() -> int:
    torch.manual_seed(0)
    print("=== Constructing 1.3B CausalOmniAvatarWan on CPU (no ckpt loading) ===")

    model = CausalOmniAvatarWan(
        model_size="1.3B",
        in_dim=49,
        use_audio=True,
        base_model_paths=None,
        omniavatar_ckpt_path=None,
        merge_lora=False,
        unfreeze_modules=[
            "_core.audio_proj",
            "_core.audio_cond_projs",
            "_core.patch_embedding",
        ],
        lora_rank=128,
        lora_alpha=64,
        dtype="fp32",
    )

    print(f"\nLORA_TARGET_MODULES: {LORA_TARGET_MODULES}")
    print()
    report(model, "after construction (no PEFT)")

    # Step 1: Manually inject PEFT (same as _load_weights's merge_lora=False path).
    print("\n=== Step 1: PEFT inject_adapter_in_model ===")
    lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights=True,
        target_modules=LORA_TARGET_MODULES,
    )
    inject_adapter_in_model(lora_config, model._core)
    report(model, "after PEFT inject")

    # Step 2: Apply selective unfreeze.
    print("\n=== Step 2: _apply_unfreeze ===")
    model._apply_unfreeze(model.unfreeze_modules)
    report(model, "after _apply_unfreeze")

    # Step 3: dtype casts (mimics _finish_init's to(default_dtype) and on_train_begin's precision cast).
    print("\n=== Step 3: dtype casts ===")
    model.to(torch.bfloat16)
    report(model, "after to(bfloat16)")
    model.to(torch.float32)
    report(model, "after to(float32)")

    # Step 4: simulate FastGenModel.build_model:260's "wipe" of requires_grad.
    print("\n=== Step 4: simulate FastGenModel.build_model:260 wipe ===")
    print("(this is what was breaking the freeze in the live training)")
    model.train().requires_grad_(True)
    report(model, "after model.train().requires_grad_(True) [WIPED]")

    # Step 5: apply the fix.
    print("\n=== Step 5: apply_lora_freeze fix ===")
    model.apply_lora_freeze()
    report(model, "after apply_lora_freeze [FIX]")

    # Step 6: simulate the wipe being called AGAIN (e.g., if some downstream
    # op re-enables) and confirm the fix is idempotent.
    print("\n=== Step 6: simulate second wipe + second fix call ===")
    model.train().requires_grad_(True)
    report(model, "after second wipe")
    model.apply_lora_freeze()
    report(model, "after second apply_lora_freeze")

    # Sample names to verify the freeze structure.
    print("\n=== Sample param names (post-inject, post-unfreeze) ===")
    all_params = list(model.named_parameters())
    print(f"Total named parameters: {len(all_params)}")

    lora_params = [(n, p.requires_grad) for n, p in all_params if "lora_" in n]
    base_in_blocks = [
        (n, p.requires_grad)
        for n, p in all_params
        if "lora_" not in n and "_core.blocks." in n
    ]
    audio_proj_params = [
        (n, p.requires_grad)
        for n, p in all_params
        if n.startswith("_core.audio_proj.")
    ]

    print(f"\nLoRA params: {len(lora_params)}  (expect: trainable=True)")
    for n, rg in lora_params[:5]:
        print(f"  {n}  requires_grad={rg}")

    print(f"\nBase params inside blocks: {len(base_in_blocks)}  (expect: trainable=False)")
    for n, rg in base_in_blocks[:5]:
        print(f"  {n}  requires_grad={rg}")

    print(f"\naudio_proj params: {len(audio_proj_params)}  (expect: trainable=True)")
    for n, rg in audio_proj_params[:5]:
        print(f"  {n}  requires_grad={rg}")

    # Final verdict.
    print("\n=== Verdict ===")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if n_trainable == n_total:
        print(f"BROKEN: trainable == total = {n_total/1e6:.2f}M")
        print("PEFT freeze did not take effect.  No params are frozen.")
        return 1
    elif n_trainable < n_total * 0.2:
        print(f"OK: trainable {n_trainable/1e6:.2f}M << total {n_total/1e6:.2f}M")
        print("Freeze is working: most params are correctly frozen.")
        return 0
    else:
        pct = 100 * n_trainable / n_total
        print(f"PARTIAL: trainable {n_trainable/1e6:.2f}M is {pct:.1f}% of total")
        print("Some freeze applied but more params trainable than expected.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
