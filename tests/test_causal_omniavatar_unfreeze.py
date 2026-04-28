"""Tests for CausalOmniAvatarWan's selective-unfreeze logic.

These tests cover the helper that re-enables requires_grad on specific
submodules after PEFT's inject_adapter_in_model has frozen the base.
We do NOT construct the full 14B model — the helper is testable against
a tiny dummy nn.Module fixture.
"""
import pytest
import torch
import torch.nn as nn


class _DummyCore(nn.Module):
    """Tiny stand-in for CausalOmniAvatarWan._core to test the unfreeze helper."""

    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(8, 16)
        self.audio_cond_projs = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])


class _DummyHost(nn.Module):
    """Stand-in for CausalOmniAvatarWan: holds _core and exposes the helper."""

    def __init__(self):
        super().__init__()
        self._core = _DummyCore()

    # The real implementation lives on CausalOmniAvatarWan; we copy the body
    # here only because importing the real class would require GPU + heavy deps.
    def _apply_unfreeze(self, unfreeze_modules):
        if not unfreeze_modules:
            return
        for path in unfreeze_modules:
            module = self.get_submodule(path)
            for p in module.parameters():
                p.requires_grad_(True)


def _set_all_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad_(value)


def test_construction_smoke():
    """Class constructs with unfreeze_modules kwarg; storage is correct."""
    # This will be replaced in Task 3 with a real call to CausalOmniAvatarWan
    # once we mock the heavy weight-loading. For Task 1, just verify the
    # dummy host pattern works.
    host = _DummyHost()
    assert hasattr(host, "_core")
    assert hasattr(host._core, "audio_proj")


def test_unfreeze_specific_submodule_re_enables_grad():
    """After freezing all params, _apply_unfreeze re-enables requires_grad
    on the parameters of the specified submodule and leaves others alone."""
    host = _DummyHost()

    # Simulate PEFT freezing the base
    _set_all_requires_grad(host, value=False)
    for p in host.parameters():
        assert p.requires_grad is False

    # Unfreeze just _core.audio_proj
    host._apply_unfreeze(["_core.audio_proj"])

    # audio_proj params should be trainable
    for p in host._core.audio_proj.parameters():
        assert p.requires_grad is True, "audio_proj.weight/bias should be trainable"

    # Everything else should remain frozen
    for p in host._core.audio_cond_projs.parameters():
        assert p.requires_grad is False
    for p in host._core.blocks.parameters():
        assert p.requires_grad is False


def test_unfreeze_modulelist_unfreezes_all_children():
    """Unfreezing a ModuleList path re-enables grad on every child Linear."""
    host = _DummyHost()
    _set_all_requires_grad(host, value=False)

    host._apply_unfreeze(["_core.audio_cond_projs"])

    for proj in host._core.audio_cond_projs:
        for p in proj.parameters():
            assert p.requires_grad is True
    # Other submodules untouched
    for p in host._core.audio_proj.parameters():
        assert p.requires_grad is False


def test_unfreeze_empty_list_is_noop():
    """Passing an empty (or None) unfreeze list does not change requires_grad."""
    host = _DummyHost()
    _set_all_requires_grad(host, value=False)

    host._apply_unfreeze([])

    for p in host.parameters():
        assert p.requires_grad is False


def test_unfreeze_unknown_path_raises_attribute_error():
    """Walking get_submodule for a non-existent path is a hard error.

    Choosing strict-failure over silent skip so a typo in the config is
    caught immediately rather than silently leaving the intended module
    frozen for the entire training run.
    """
    host = _DummyHost()
    with pytest.raises(AttributeError):
        host._apply_unfreeze(["_core.does_not_exist"])


def test_init_signature_has_unfreeze_modules():
    """Confirm the new kwarg is in the real CausalOmniAvatarWan signature."""
    import inspect
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    sig = inspect.signature(CausalOmniAvatarWan.__init__)
    assert "unfreeze_modules" in sig.parameters, (
        "CausalOmniAvatarWan.__init__ should accept an unfreeze_modules kwarg"
    )
    # Default should be None (i.e., no-op when not provided)
    assert sig.parameters["unfreeze_modules"].default is None


def test_apply_unfreeze_is_callable():
    """Confirm the helper is bound to the real class."""
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    assert callable(CausalOmniAvatarWan._apply_unfreeze)


def test_apply_lora_freeze_is_callable():
    """Confirm the freeze recovery hook is bound to the real class.

    apply_lora_freeze is the recovery method called by
    OmniAvatarDiffusionForcingModel.build_model after super wipes
    requires_grad. It needs to exist on the real class.
    """
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    assert callable(CausalOmniAvatarWan._apply_unfreeze)
    assert callable(CausalOmniAvatarWan.apply_lora_freeze)


def test_omniavatarwan_apply_lora_freeze_is_callable():
    """Bidirectional OmniAvatarWan must also have apply_lora_freeze.

    Used by OmniAvatarSelfForcingModel.build_model to recover the freeze
    on fake_score (defensive: fake_score isn't subject to the wipe today,
    but for symmetry the method should exist on both classes).
    """
    from fastgen.networks.OmniAvatar.network import OmniAvatarWan

    assert callable(OmniAvatarWan._apply_unfreeze)
    assert callable(OmniAvatarWan.apply_lora_freeze)


def test_omniavatarwan_init_signature_has_unfreeze_modules():
    """Confirm OmniAvatarWan.__init__ accepts unfreeze_modules with default None."""
    import inspect
    from fastgen.networks.OmniAvatar.network import OmniAvatarWan

    sig = inspect.signature(OmniAvatarWan.__init__)
    assert "unfreeze_modules" in sig.parameters
    assert sig.parameters["unfreeze_modules"].default is None


# --- apply_lora_freeze behavior tests on a CPU-only fixture --------------
#
# These tests exercise the exact wipe-then-recover cycle that
# FastGenModel.build_model:260 + OmniAvatarDiffusionForcingModel.build_model
# triggers. We bind the real CausalOmniAvatarWan.apply_lora_freeze method
# onto a tiny dummy fixture so we don't need to construct the full 14B
# model (which requires GPU + heavy weight loading).


class _DummyLoraHost(nn.Module):
    """Stand-in for a PEFT-injected CausalOmniAvatarWan.

    Carries the same attributes apply_lora_freeze inspects:
      - merge_lora flag
      - unfreeze_modules list
      - parameters whose names contain ``"lora_"`` (mimicking PEFT's
        injected adapters) plus parameters that don't (mimicking the
        frozen base + the explicitly-unfreezable submodules).
    """

    def __init__(self, merge_lora=False, unfreeze_modules=None):
        super().__init__()
        self.merge_lora = merge_lora
        self.unfreeze_modules = list(unfreeze_modules) if unfreeze_modules else []

        self._core = nn.Module()
        # Fake "blocks" with base + LoRA A/B params (mimic PEFT-injected layout).
        self._core.blocks = nn.ModuleList(
            [_make_lora_injected_linear(dim=8, rank=4) for _ in range(2)]
        )
        # Non-block submodules — these are the user-listed unfreeze targets.
        self._core.audio_proj = nn.Linear(4, 8)
        self._core.audio_cond_projs = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self._core.patch_embedding = nn.Linear(16, 8)


def _make_lora_injected_linear(dim, rank):
    """A nn.Module shaped like PEFT's LoraLinear for testing.

    The naming convention mirrors PEFT (parameters with "lora_" in their
    name should be treated as trainable adapter weights, all others as
    frozen base) — apply_lora_freeze relies on that prefix.
    """
    mod = nn.Module()
    # Frozen base (PEFT's `base_layer`).
    mod.base_layer = nn.Linear(dim, dim)
    # Trainable LoRA adapters — the names must contain "lora_" so
    # apply_lora_freeze identifies them.
    mod.lora_A = nn.Linear(dim, rank, bias=False)
    mod.lora_B = nn.Linear(rank, dim, bias=False)
    return mod


def _bind_apply_lora_freeze(host):
    """Attach the real CausalOmniAvatarWan methods to the dummy host."""
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    host._apply_unfreeze = CausalOmniAvatarWan._apply_unfreeze.__get__(host)
    host.apply_lora_freeze = CausalOmniAvatarWan.apply_lora_freeze.__get__(host)


def test_apply_lora_freeze_restores_freeze_after_wipe():
    """The exact bug: build_model wipes requires_grad, recovery restores it."""
    host = _DummyLoraHost(
        merge_lora=False,
        unfreeze_modules=["_core.audio_proj", "_core.patch_embedding"],
    )
    _bind_apply_lora_freeze(host)

    # Simulate FastGenModel.build_model:260 wiping the freeze.
    host.train().requires_grad_(True)
    n_total = sum(p.numel() for p in host.parameters())
    n_trainable_pre = sum(p.numel() for p in host.parameters() if p.requires_grad)
    assert n_trainable_pre == n_total, "wipe should leave all params trainable"

    # Apply the recovery hook.
    host.apply_lora_freeze()

    # LoRA params must stay trainable.
    for n, p in host.named_parameters():
        if "lora_" in n:
            assert p.requires_grad is True, f"LoRA param {n} should be trainable"

    # Block base_layer params must be frozen.
    for n, p in host.named_parameters():
        if "base_layer" in n:
            assert p.requires_grad is False, f"base_layer param {n} should be frozen"

    # Listed unfreeze submodules must be trainable.
    for p in host._core.audio_proj.parameters():
        assert p.requires_grad is True
    for p in host._core.patch_embedding.parameters():
        assert p.requires_grad is True

    # Non-listed submodule (audio_cond_projs) is NOT in unfreeze_modules,
    # and its params don't carry the "lora_" prefix → must end up frozen.
    for p in host._core.audio_cond_projs.parameters():
        assert p.requires_grad is False, "non-listed submodule should be frozen"


def test_apply_lora_freeze_idempotent():
    """Calling apply_lora_freeze multiple times yields identical state."""
    host = _DummyLoraHost(
        merge_lora=False,
        unfreeze_modules=["_core.audio_proj"],
    )
    _bind_apply_lora_freeze(host)

    host.train().requires_grad_(True)
    host.apply_lora_freeze()
    state_after_first = {n: p.requires_grad for n, p in host.named_parameters()}

    # Wipe and re-apply.
    host.train().requires_grad_(True)
    host.apply_lora_freeze()
    state_after_second = {n: p.requires_grad for n, p in host.named_parameters()}

    assert state_after_first == state_after_second


def test_apply_lora_freeze_noop_when_merge_lora_true():
    """If merge_lora=True, no LoRA layers exist and recovery should not freeze
    anything — the model is in plain full-FT mode."""
    host = _DummyLoraHost(
        merge_lora=True,  # full FT mode
        unfreeze_modules=[],
    )
    _bind_apply_lora_freeze(host)

    host.train().requires_grad_(True)  # all trainable (full FT default)
    host.apply_lora_freeze()  # should be a no-op

    n_total = sum(p.numel() for p in host.parameters())
    n_trainable = sum(p.numel() for p in host.parameters() if p.requires_grad)
    assert n_trainable == n_total, "full-FT mode should keep everything trainable"


def test_apply_lora_freeze_noop_when_no_lora_params():
    """If merge_lora=False but no params have 'lora_' in their name (PEFT
    injection didn't actually run), apply_lora_freeze should bail out
    rather than freezing the entire base — that would leave the model
    with NOTHING trainable, which is worse."""

    class _NoLoraHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.merge_lora = False
            self.unfreeze_modules = []
            self._core = nn.Module()
            self._core.audio_proj = nn.Linear(4, 8)
            self._core.blocks = nn.ModuleList([nn.Linear(8, 8)])

    host = _NoLoraHost()
    _bind_apply_lora_freeze(host)

    host.train().requires_grad_(True)
    host.apply_lora_freeze()  # should be a no-op (no lora_ params present)

    n_total = sum(p.numel() for p in host.parameters())
    n_trainable = sum(p.numel() for p in host.parameters() if p.requires_grad)
    assert n_trainable == n_total, (
        "no-lora-params case should leave everything trainable; "
        "freezing the base with no LoRA to compensate would brick the run"
    )
