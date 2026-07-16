# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.model_executor.stage_input_processors.diffusion_disagg import diffusion_stage_handoff


def test_diffusion_stage_handoff_preserves_typed_payload_and_multimodal_data() -> None:
    inline_payload = {
        "request_id": "req-0",
        "boundary": "encode_to_dit",
        "tensor_fields": {"prompt_embeds": "inline-tensor"},
    }
    transfer_handle = {"key": "req-0:encode_to_dit", "boundary": "encode_to_dit"}
    source_output = SimpleNamespace(
        _custom_output={
            "stage_payload": inline_payload,
            "_stage_payload_transfer": transfer_handle,
        }
    )
    image = object()
    prompt = {
        "prompt": "animate this image",
        "multi_modal_data": {"image": image},
        "num_frames": 17,
    }

    result = diffusion_stage_handoff([source_output], [prompt])

    assert result == [
        {
            "prompt": "animate this image",
            "multi_modal_data": {"image": image},
            "num_frames": 17,
            "stage_payload": inline_payload,
            "_stage_payload_transfer": transfer_handle,
        }
    ]
