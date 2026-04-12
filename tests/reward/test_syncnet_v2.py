import os
import pytest
import torch

from fastgen.methods.reward.syncnet_v2 import SyncNetV2

CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model"


def test_instantiates():
    m = SyncNetV2()
    assert sum(p.numel() for p in m.parameters()) > 10_000_000


def test_forward_shapes():
    m = SyncNetV2().eval()
    with torch.no_grad():
        lip = torch.randn(2, 3, 5, 224, 224)
        aud = torch.randn(2, 1, 13, 20)
        lip_emb = m.forward_lip(lip)
        aud_emb = m.forward_aud(aud)
    assert lip_emb.shape == (2, 1024)
    assert aud_emb.shape == (2, 1024)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="SyncNet-v2 checkpoint not present locally")
def test_loads_checkpoint():
    m = SyncNetV2()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    m.load_state_dict(state, strict=True)
