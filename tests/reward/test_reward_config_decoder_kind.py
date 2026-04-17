"""RewardConfig must carry a decoder_kind field that defaults to the Wan VAE."""


def test_reward_config_has_decoder_kind_field():
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(enabled=True, checkpoint_path="/fake/syncnet.model")
    assert hasattr(c, "decoder_kind")
    assert c.decoder_kind == "vae", (
        f"default must preserve existing Wan VAE behavior, got {c.decoder_kind!r}"
    )
    assert hasattr(c, "taew_checkpoint_path")
    assert c.taew_checkpoint_path == ""


def test_reward_config_accepts_taew_kind():
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(
        enabled=True,
        checkpoint_path="/fake/syncnet.model",
        decoder_kind="taew",
        taew_checkpoint_path="/fake/taew.pth",
    )
    assert c.decoder_kind == "taew"
    assert c.taew_checkpoint_path == "/fake/taew.pth"


def test_reward_config_rejects_unknown_kind():
    # The attrs class shouldn't hard-reject at construction time (we validate
    # at build_model time instead, to give a clearer error message), so this
    # test just confirms the field accepts arbitrary strings. The Re-DMD model
    # handles the unknown-kind case.
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(enabled=True, checkpoint_path="/fake", decoder_kind="nonsense")
    assert c.decoder_kind == "nonsense"
