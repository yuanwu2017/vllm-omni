# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from vllm.platforms.cpu import CpuPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum


class CPUOmniPlatform(OmniPlatform, CpuPlatform):
    """CPU implementation of OmniPlatform for encode-only diffusion stages."""

    _omni_enum = OmniPlatformEnum.CPU

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm.worker.cpu_worker.CPUWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm.worker.cpu_worker.CPUWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/xpu/stage_configs"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is not None:
            backend = DiffusionAttentionBackendEnum[selected_backend.upper()]
            return backend.get_path()
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        return None

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        return None

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        return 0

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        return 0

    @classmethod
    def empty_cache(cls) -> None:
        return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return cls.get_device_total_memory(0)

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        total = cls.get_device_total_memory(0)
        return cls.get_free_memory(device), total
