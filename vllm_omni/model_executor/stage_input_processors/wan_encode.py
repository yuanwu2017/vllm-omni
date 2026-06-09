# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Wan text-to-video: text_encode -> DiT transition.

The upstream ``text_encode`` stage runs only the UMT5 text encoder and emits
prompt embeddings in ``DiffusionOutput.custom_output`` (surfaced on the stage
request output as ``_custom_output``). This processor threads those embeddings
into the downstream DiT/decode stage's prompt dict so the decode stage skips
text encoding entirely (Encode/Prefill-Decode disaggregation).
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_EMBED_KEYS = ("prompt_embeds", "negative_prompt_embeds")
# Diffusion sampling/control fields worth forwarding verbatim to the DiT stage.
_PASSTHROUGH_KEYS = (
    "negative_prompt",
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "guidance_scale_2",
    "boundary_ratio",
    "fps",
    "seed",
    "modalities",
)


def _as_dict(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _extract_custom_output(source_output: Any) -> dict[str, Any]:
    """Pull the encode stage's emitted embeddings from a stage output object."""
    for attr in ("_custom_output", "custom_output"):
        value = getattr(source_output, attr, None)
        if isinstance(value, dict) and value:
            return value
    return {}


def encode2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Build DiT-stage prompt dicts from the text_encode stage outputs.

    Accepts the stage-pool transition interface
    ``encode2diffusion(source_outputs, prompt, requires_multimodal_data)``.
    """
    del requires_multimodal_data, streaming_context

    if not isinstance(prompt, list):
        prompts = [prompt] if prompt is not None else [{}]
    else:
        prompts = prompt

    diffusion_inputs: list[dict[str, Any]] = []
    for i, source_output in enumerate(source_outputs):
        original_prompt = _as_dict(prompts[i] if i < len(prompts) else {})
        custom_output = _extract_custom_output(source_output)

        next_prompt: dict[str, Any] = {}
        # Forward the original text so logging/metadata still has it. The DiT
        # pipeline drops the raw text when embeddings are present.
        if original_prompt.get("prompt") is not None:
            next_prompt["prompt"] = original_prompt["prompt"]
        for key in _PASSTHROUGH_KEYS:
            if original_prompt.get(key) is not None:
                next_prompt[key] = original_prompt[key]

        prompt_embeds = custom_output.get("prompt_embeds")
        if prompt_embeds is None:
            logger.warning(
                "[encode2diffusion] request %d: no prompt_embeds in upstream "
                "custom_output (keys=%s); falling back to raw-text encoding.",
                i,
                list(custom_output.keys()),
            )
        else:
            for key in _EMBED_KEYS:
                if custom_output.get(key) is not None:
                    next_prompt[key] = custom_output[key]

        diffusion_inputs.append(next_prompt)

    return diffusion_inputs
