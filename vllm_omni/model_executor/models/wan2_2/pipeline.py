# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.x disaggregated diffusion pipeline topologies (frozen).

Wan runs as a single-stage diffusion model by default (text-encode + DiT
denoise + VAE decode fused on one worker). These topologies describe the
*disaggregated* variants, defined in code (per RFC #4021) rather than in a
legacy ``platforms/.../stage_configs/*.yaml`` file:

  Encode/Generation (EG), 2 stages:
    Stage 0 (encode):  UMT5 text encoder only -> prompt embeddings.
    Stage 1 (denoise): DiT denoise + VAE decode -> video, consuming the
                       stage-0 embeddings (no text encoding in stage 1).

The stages are wired model-agnostically via ``DiffusionStageRole`` and the
generic cross-stage handoff processor
``vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff``,
so the same machinery generalizes to other DiT models and to a 3-way
Encode/Generation/Decode (EGD) split.
"""

from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_WAN_MODEL_ARCH = "WanPipeline"
_DIFFUSION_HANDOFF = (
    "vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff"
)

# Prompt-embedding payload transferred across the encode -> denoise edge.
_ENCODE_PAYLOAD_KEYS = ("prompt_embeds", "negative_prompt_embeds")


# --- Single-stage full pipeline ---------------------------------------------
WAN2_2_PIPELINE = PipelineConfig(
    model_type="wan2_2",
    model_arch=_WAN_MODEL_ARCH,
    diffusers_class_name="WanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.FULL,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch=_WAN_MODEL_ARCH,
            engine_output_type="video",
        ),
    ),
)


# --- Encode/Generation (EG): 2-stage disaggregation -------------------------
WAN2_2_EG_PIPELINE = PipelineConfig(
    model_type="wan2_2_eg",
    model_arch=_WAN_MODEL_ARCH,
    diffusers_class_name="WanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="text_encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_WAN_MODEL_ARCH,
            # Surface the emitted embeddings on the stage output's custom_output
            # so the orchestrator/connector forwards them downstream.
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE,
            stage_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            model_arch=_WAN_MODEL_ARCH,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)
