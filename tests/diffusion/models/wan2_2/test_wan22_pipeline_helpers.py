# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.models.interface import StageBoundary, StagePayload
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    create_transformer_from_config,
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _LatentDist:
    def sample(self, generator):
        assert isinstance(generator, torch.Generator)
        return torch.tensor([1.0])

    def mode(self):
        return torch.tensor([2.0])


def test_retrieve_latents_supports_sample_mode_argmax_and_direct_latents() -> None:
    generator = torch.Generator(device="cpu")

    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), generator).item() == 1.0
    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), sample_mode="argmax").item() == 2.0
    torch.testing.assert_close(retrieve_latents(SimpleNamespace(latents=torch.tensor([3.0]))), torch.tensor([3.0]))


def test_retrieve_latents_rejects_unknown_encoder_output() -> None:
    with pytest.raises(AttributeError, match="Could not access latents"):
        retrieve_latents(SimpleNamespace())


def test_load_transformer_config_reads_local_subfolder_config(tmp_path) -> None:
    config_dir = tmp_path / "transformer_2"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({"patch_size": [1, 2, 2], "num_layers": 2}))

    assert load_transformer_config(str(tmp_path), "transformer_2") == {"patch_size": [1, 2, 2], "num_layers": 2}
    assert load_transformer_config(str(tmp_path), "missing") == {}


def test_create_transformer_from_config_maps_supported_keys(monkeypatch) -> None:
    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    transformer = create_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "num_attention_heads": 8,
            "attention_head_dim": 128,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 4096,
            "vace_layers": [0],
            "ignored": "value",
        }
    )

    assert isinstance(transformer, FakeTransformer)
    assert captured == {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
    }


def _make_state(request_id: str = "req-wan") -> DiffusionRequestState:
    return DiffusionRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(),
    )


def test_wan_encode_to_dit_payload_round_trip() -> None:
    pipeline = object.__new__(Wan22Pipeline)
    source = _make_state()
    source.prompt_embeds = torch.ones(1, 2, 3)
    source.negative_prompt_embeds = torch.zeros(1, 2, 3)
    source.do_true_cfg = True
    source.extra.update({"guidance_low": 3.0, "guidance_high": 4.0, "output_type": "latent"})

    payload = pipeline.pack_stage_state(source, StageBoundary.ENCODE_TO_DIT)
    target = pipeline.unpack_stage_state(payload, _make_state())

    assert payload.boundary == StageBoundary.ENCODE_TO_DIT
    assert torch.equal(target.prompt_embeds, source.prompt_embeds)
    assert torch.equal(target.negative_prompt_embeds, source.negative_prompt_embeds)
    assert target.do_true_cfg
    assert target.extra["guidance_low"] == 3.0
    assert target.extra["output_type"] == "latent"


def test_wan_encode_payload_requires_prompt_embeds() -> None:
    pipeline = object.__new__(Wan22Pipeline)

    with pytest.raises(ValueError, match="no prompt_embeds"):
        pipeline.pack_stage_state(_make_state(), StageBoundary.ENCODE_TO_DIT)


@pytest.mark.parametrize(
    ("payload_version", "request_id", "match"),
    [(2, "req-wan", "version"), (1, "different", "does not match")],
)
def test_wan_unpack_rejects_invalid_payload_identity(payload_version, request_id, match) -> None:
    pipeline = object.__new__(Wan22Pipeline)
    payload = StagePayload(
        request_id=request_id,
        boundary=StageBoundary.ENCODE_TO_DIT,
        scalar_fields={},
        tensor_fields={"prompt_embeds": torch.ones(1, 2, 3)},
        private_scalar_fields={},
        private_tensor_fields={},
        payload_version=payload_version,
    )

    with pytest.raises(ValueError, match=match):
        pipeline.unpack_stage_state(payload, _make_state())


def test_wan_dit_to_decode_payload_round_trip() -> None:
    pipeline = object.__new__(Wan22Pipeline)
    source = _make_state()
    source.latents = torch.ones(1, 4, 2, 2, 2)
    source.extra["latent_condition"] = torch.zeros_like(source.latents)
    source.extra["output_type"] = "np"

    payload = pipeline.pack_stage_state(source, StageBoundary.DIT_TO_DECODE)
    target = pipeline.unpack_stage_state(payload, _make_state())

    assert torch.equal(target.latents, source.latents)
    assert torch.equal(target.extra["latent_condition"], source.extra["latent_condition"])
    assert target.extra["output_type"] == "np"
