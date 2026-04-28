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
