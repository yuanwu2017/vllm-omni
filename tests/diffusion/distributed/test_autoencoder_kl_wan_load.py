# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
from diffusers.models import modeling_utils as diffusers_modeling_utils

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_from_pretrained_skips_and_restores_diffusers_empty_cache_on_xpu(monkeypatch):
    original_empty_device_cache = diffusers_modeling_utils.empty_device_cache
    model = object.__new__(DistributedAutoencoderKLWan)
    init_calls = []

    def fake_from_pretrained(cls, *args, **kwargs):
        del cls, args, kwargs
        assert diffusers_modeling_utils.empty_device_cache is not original_empty_device_cache
        diffusers_modeling_utils.empty_device_cache()
        return model

    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.current_omni_platform.is_xpu",
        lambda: True,
    )
    monkeypatch.setattr(model, "init_distributed", lambda: init_calls.append(True))

    with patch.object(OmniAutoencoderKLWan, "from_pretrained", classmethod(fake_from_pretrained)):
        loaded = DistributedAutoencoderKLWan.from_pretrained("model", subfolder="vae")

    assert loaded is model
    assert init_calls == [True]
    assert diffusers_modeling_utils.empty_device_cache is original_empty_device_cache


def test_from_pretrained_restores_diffusers_empty_cache_after_xpu_failure(monkeypatch):
    original_empty_device_cache = diffusers_modeling_utils.empty_device_cache

    def fake_from_pretrained(cls, *args, **kwargs):
        del cls, args, kwargs
        assert diffusers_modeling_utils.empty_device_cache is not original_empty_device_cache
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.current_omni_platform.is_xpu",
        lambda: True,
    )

    with patch.object(OmniAutoencoderKLWan, "from_pretrained", classmethod(fake_from_pretrained)):
        with pytest.raises(RuntimeError, match="load failed"):
            DistributedAutoencoderKLWan.from_pretrained("model", subfolder="vae")

    assert diffusers_modeling_utils.empty_device_cache is original_empty_device_cache


def test_from_pretrained_keeps_diffusers_empty_cache_on_non_xpu(monkeypatch):
    original_empty_device_cache = diffusers_modeling_utils.empty_device_cache
    model = object.__new__(DistributedAutoencoderKLWan)

    def fake_from_pretrained(cls, *args, **kwargs):
        del cls, args, kwargs
        assert diffusers_modeling_utils.empty_device_cache is original_empty_device_cache
        return model

    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.current_omni_platform.is_xpu",
        lambda: False,
    )
    monkeypatch.setattr(model, "init_distributed", lambda: None)

    with patch.object(OmniAutoencoderKLWan, "from_pretrained", classmethod(fake_from_pretrained)):
        loaded = DistributedAutoencoderKLWan.from_pretrained("model", subfolder="vae")

    assert loaded is model
    assert diffusers_modeling_utils.empty_device_cache is original_empty_device_cache
