"""
Base Module Interface for the Advanced Computational Speech Engineering Pipeline.

Provides abstract lifecycle management, device resolution (MPS/CUDA/CPU),
timing telemetry, and strict typing across all pipeline stages.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.core.state import PipelineState


def resolve_device(device_preference: str = "auto") -> Any:
    """
    Deterministically resolves the optimal compute device.
    Prioritizes Apple Silicon Metal (MPS) on macOS, CUDA on NVIDIA Linux/Colab,
    and falls back gracefully to CPU.
    """
    if not HAS_TORCH:
        return "cpu"

    if device_preference.lower() == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_preference.lower() == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif device_preference.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


class BaseModule(ABC):
    """
    Abstract Base Class for all pipeline modules (Preprocessing, BSS,
    Diarization, ASR, LLM Guardrailing).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = "auto"):
        self.config = config or {}
        self.device = resolve_device(device or self.config.get("device", "auto"))
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Loads models, allocates weight tensors, and verifies hardware availability.
        Subclasses must implement lazy or explicit initialization.
        """
        pass

    @abstractmethod
    def process(self, state: PipelineState) -> PipelineState:
        """
        Executes module inference, transforming and enriching the PipelineState.
        Must return the updated PipelineState object.
        """
        pass

    def cleanup(self) -> None:
        """
        Frees device cache, tears down lingering resources, and triggers garbage collection.
        """
        if not HAS_TORCH or torch is None:
            return

        device_type = getattr(self.device, "type", str(self.device))
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def __call__(self, state: PipelineState) -> PipelineState:
        """
        Executes the module with timing telemetry.
        """
        if not self._is_initialized:
            self.initialize()
            self._is_initialized = True

        start_time = time.perf_counter()
        updated_state = self.process(state)
        elapsed_sec = time.perf_counter() - start_time

        stage_name = self.__class__.__name__
        updated_state.execution_telemetry[stage_name] = {
            "elapsed_seconds": round(elapsed_sec, 4),
            "device": str(self.device),
        }

        self.cleanup()
        return updated_state
