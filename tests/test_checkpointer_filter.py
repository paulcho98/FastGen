"""Tests for FSDPCheckpointer's trainable-only state-dict filter.

We don't construct an FSDP-wrapped model here (that requires GPUs +
distributed init) — instead we exercise the filter logic directly on
a vanilla nn.Module to verify it correctly partitions params by
requires_grad while preserving non-parameter state.

The filter mirrors the logic at FSDPCheckpointer.save:

    params_dict = dict(v.named_parameters())
    filtered = {
        key: tensor
        for key, tensor in model_state_dict.items()
        if key not in params_dict or params_dict[key].requires_grad
    }

So this test reproduces that filter on a small fixture and asserts the
expected partition.
"""
import torch
import torch.nn as nn


def _filter_state_dict_to_trainable(model: nn.Module, state_dict: dict) -> dict:
    """Reproduces the filter in FSDPCheckpointer.save."""
    params_dict = dict(model.named_parameters())
    return {
        key: tensor
        for key, tensor in state_dict.items()
        if key not in params_dict or params_dict[key].requires_grad
    }


def test_filter_drops_frozen_params():
    """Frozen layer's params are filtered out; trainable layer's are kept."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # Freeze the second layer
    for p in model[1].parameters():
        p.requires_grad_(False)

    sd = model.state_dict()
    filtered = _filter_state_dict_to_trainable(model, sd)

    # Layer 0 params present
    assert "0.weight" in filtered
    assert "0.bias" in filtered
    # Layer 1 params absent
    assert "1.weight" not in filtered
    assert "1.bias" not in filtered


def test_filter_noop_when_all_trainable():
    """Full-FT case: filter is a no-op; all keys preserved."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    sd = model.state_dict()
    filtered = _filter_state_dict_to_trainable(model, sd)
    assert filtered.keys() == sd.keys()


def test_filter_preserves_non_param_state():
    """Buffers and other non-parameter state are preserved even when frozen.

    BatchNorm has running_mean/running_var as buffers (not nn.Parameter).
    Even if all params are frozen, the buffers must still be saved so
    inference works correctly.
    """
    model = nn.BatchNorm1d(4)
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    sd = model.state_dict()
    filtered = _filter_state_dict_to_trainable(model, sd)

    # Frozen params dropped
    assert "weight" not in filtered
    assert "bias" not in filtered
    # Non-parameter buffers preserved
    assert "running_mean" in filtered
    assert "running_var" in filtered
    assert "num_batches_tracked" in filtered


def test_filter_handles_mixed_freeze_within_layer():
    """A Linear with weight frozen but bias trainable: only bias survives."""
    model = nn.Linear(4, 4)
    model.weight.requires_grad_(False)
    # bias stays requires_grad=True

    sd = model.state_dict()
    filtered = _filter_state_dict_to_trainable(model, sd)

    assert "weight" not in filtered
    assert "bias" in filtered


def test_filter_with_peft_like_naming():
    """LoRA-style adapter: 'lora_' params trainable, 'base_layer' params frozen.

    Mirrors the actual structure PEFT creates after inject_adapter_in_model.
    """
    model = nn.Module()
    # Simulate PEFT's layout
    model.base_layer_weight = nn.Parameter(torch.randn(4, 4))
    model.lora_A = nn.Parameter(torch.randn(2, 4))
    model.lora_B = nn.Parameter(torch.randn(4, 2))
    # PEFT freezes the base, keeps LoRA trainable
    model.base_layer_weight.requires_grad_(False)
    # lora_A and lora_B stay requires_grad=True

    sd = model.state_dict()
    filtered = _filter_state_dict_to_trainable(model, sd)

    assert "base_layer_weight" not in filtered
    assert "lora_A" in filtered
    assert "lora_B" in filtered


def test_modelwrapper_default_options_strict_false():
    """ModelWrapper's default StateDictOptions sets strict=False so partial
    loads (where saved state is missing frozen-base keys) succeed."""
    from fastgen.utils.checkpointer import ModelWrapper

    model = nn.Linear(4, 4)
    wrapper = ModelWrapper(model)
    assert wrapper.options is not None
    assert wrapper.options.strict is False


def test_modelwrapper_respects_explicit_options():
    """If the caller passes an explicit StateDictOptions, it isn't overridden."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from fastgen.utils.checkpointer import ModelWrapper

    model = nn.Linear(4, 4)
    custom = StateDictOptions(strict=True, full_state_dict=True)
    wrapper = ModelWrapper(model, options=custom)
    assert wrapper.options is custom
    assert wrapper.options.strict is True
