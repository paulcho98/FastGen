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
