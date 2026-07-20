# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.config.stage_config import DiffusionStageRole
from vllm_omni.diffusion import io_support
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dummy_run_num_frames_uses_explicit_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    class JointAudioVideoModel:
        dummy_run_num_frames = 2

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: JointAudioVideoModel,
    )

    assert io_support.get_dummy_run_num_frames("joint_audio_video", supports_audio_input=False) == 2


def test_dummy_run_num_frames_keeps_audio_output_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioOutputModel:
        support_audio_output = True

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: AudioOutputModel,
    )

    assert io_support.get_dummy_run_num_frames("audio_output", supports_audio_input=False) == 2


def test_dummy_run_num_frames_defaults_to_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoOnlyModel:
        pass

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: VideoOnlyModel,
    )

    assert io_support.get_dummy_run_num_frames("video_only", supports_audio_input=False) == 1


def test_dummy_run_num_frames_uses_audio_input_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: None,
    )

    assert io_support.get_dummy_run_num_frames("unknown", supports_audio_input=True) == 2


@pytest.mark.parametrize(
    "stage_role",
    [DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE, DiffusionStageRole.DECODE],
)
def test_dummy_run_skips_stage_roles_that_require_upstream_payload(stage_role: DiffusionStageRole) -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = type("Config", (), {"stage_role": stage_role, "model_stage": None})()

    engine._dummy_run()


def test_dummy_run_does_not_skip_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = type(
        "Config",
        (),
        {
            "stage_role": DiffusionStageRole.FULL,
            "model_stage": None,
            "model_class_name": "video_only",
        },
    )()
    engine.pre_process_func = None
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.supports_multimodal_input",
        lambda _config: (False, False),
    )
    monkeypatch.setattr(engine, "add_req_and_wait_for_response", lambda _request: type("Output", (), {"error": None})())

    engine._dummy_run()
