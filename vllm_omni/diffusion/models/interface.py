# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@dataclass(frozen=True)
class ReferenceVideoDecodeSpec:
    max_frames: int | None = None
    keep: Literal["first", "last"] = "first"


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, and other operations that need to know
    the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)


# --- Generic role-based component loading for diffusion disaggregation ---------

# Which component groups (as declared by ``SupportsComponentDiscovery``) each
# diffusion stage role needs to load. ``encoder`` = text/image encoders,
# ``dit`` = denoising transformer(s), ``vae`` = VAE. A stage skips loading every
# group not listed for its role, which is what frees memory on disaggregated
# workers (e.g. an ENCODE stage drops the DiT + VAE). This mapping is
# model-agnostic: any pipeline that implements ``SupportsComponentDiscovery``
# becomes role-splittable without model-specific ``if`` branches.
_ROLE_COMPONENT_GROUPS: dict[str, frozenset[str]] = {
    "full": frozenset({"encoder", "dit", "vae"}),
    "encode": frozenset({"encoder"}),
    # DENOISE consumes upstream embeddings and produces the final media, so it
    # runs the transformer(s) and the VAE decode (fused generation stage).
    "denoise": frozenset({"dit", "vae"}),
    "decode": frozenset({"vae"}),
}


def stage_component_groups(role: str) -> frozenset[str]:
    """Return the component groups (``encoder``/``dit``/``vae``) a role loads."""
    return _ROLE_COMPONENT_GROUPS.get(role, _ROLE_COMPONENT_GROUPS["full"])


def role_loads_component(role: str, group: Literal["encoder", "dit", "vae"]) -> bool:
    """Whether a diffusion stage ``role`` should load the given component group.

    Use inside a pipeline ``__init__`` to decide, model-agnostically, whether to
    instantiate the encoder / DiT / VAE for the current stage role::

        if role_loads_component(role, "dit"):
            self.transformer = ...
        if role_loads_component(role, "vae"):
            self.vae = ...
    """
    return group in stage_component_groups(role)
