"""
Conv-TasNet Source Separation Module (Comparative Baseline).

Implements Convolutional Time-domain Audio Separation Network (Conv-TasNet)
pre-trained on 2-speaker separation (Libri2Mix), as identified in the
academic proposal for comparative evaluation against Meta's Demucs.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal
import torch
import torchaudio

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio
from src.core.base_module import BaseModule
from src.core.state import PipelineState, SeparatedTrack, SeparationResult


class ConvTasNetSeparator(BaseModule):
    """
    Inference wrapper for Conv-TasNet 2-speaker separation architecture.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "auto",
    ):
        super().__init__(config=config, device=device)
        self.bundle = None
        self.model = None
        self.model_sr = 8000
        self.target_sr = self.config.get("target_sample_rate", 16000)

    def initialize(self) -> None:
        """Loads Conv-TasNet model weights from Torchaudio pipelines."""
        if self._is_initialized:
            return

        self.bundle = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX
        self.model = self.bundle.get_model()
        self.model.to(self.device)
        self.model.eval()
        self.model_sr = self.bundle.sample_rate
        self._is_initialized = True

    def separate_audio(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000,
        output_dir: Optional[Union[str, Path]] = None,
        file_prefix: str = "convtasnet_sep",
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Executes Conv-TasNet separation forward pass.

        Args:
            audio_tensor: Input audio shaped (channels, samples) or (samples,).
            sample_rate: Sample rate of input tensor.
            output_dir: Optional directory to save separated WAV files.
            file_prefix: Filename prefix for saved stems.

        Returns:
            Tuple of (list_of_16k_mono_numpy_arrays, metadata_dict).
        """
        if not self._is_initialized:
            self.initialize()

        assert self.model is not None

        # 1. Downmix to mono if stereo
        if audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        elif audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 2. Resample to Conv-TasNet model sample rate (8000 Hz)
        data = audio_tensor.cpu().numpy().squeeze(0)
        gcd = np.gcd(sample_rate, self.model_sr)
        up = self.model_sr // gcd
        down = sample_rate // gcd
        resampled_8k = scipy.signal.resample_poly(data, up, down).astype(np.float32)

        # Tensor shape for Conv-TasNet: (batch=1, channels=1, frames)
        input_tensor = torch.from_numpy(resampled_8k).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Model forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            # Output shape: (batch=1, num_sources=2, frames)
            separated_8k = self.model(input_tensor)
        infer_duration = time.perf_counter() - start_time

        sources_8k = separated_8k.squeeze(0).cpu().numpy() # (2, frames)

        # 4. Upsample separated stems back to target_sr (16 kHz)
        gcd_back = np.gcd(self.model_sr, self.target_sr)
        up_back = self.target_sr // gcd_back
        down_back = self.model_sr // gcd_back

        stems_16k = []
        for i in range(sources_8k.shape[0]):
            stem_16k = scipy.signal.resample_poly(sources_8k[i], up_back, down_back).astype(np.float32)
            stems_16k.append(stem_16k)

        # Energy ratios
        energies = [float(np.sum(s**2)) for s in stems_16k]
        total_energy = sum(energies) + 1e-12
        ratios = [e / total_energy for e in energies]

        meta: Dict[str, Any] = {
            "model_name": "conv_tasnet_base_libri2mix",
            "inference_time_seconds": round(infer_duration, 4),
            "num_stems": len(stems_16k),
            "stems": [],
        }

        # 5. Save stems if output_dir specified
        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            for i, (stem, ratio) in enumerate(zip(stems_16k, ratios)):
                label = f"speaker_{i}"
                stem_file = out_p / f"{file_prefix}_{label}.wav"
                save_audio(stem_file, stem, sample_rate=self.target_sr)
                meta["stems"].append({
                    "source_index": i,
                    "label": label,
                    "file_path": str(stem_file),
                    "energy_ratio": round(ratio, 4),
                    "metrics": compute_audio_metrics(stem, sample_rate=self.target_sr),
                })

        return stems_16k, meta

    def separate_file(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Separates an audio file from disk."""
        path = Path(audio_path)
        tensor, sr = load_audio(path, target_sr=self.target_sr, to_mono=True)
        prefix = path.stem
        return self.separate_audio(tensor, sample_rate=sr, output_dir=output_dir, file_prefix=prefix)

    def process(self, state: PipelineState) -> PipelineState:
        """Pipeline state processing."""
        input_file = state.normalized_audio_path or state.input_audio_path
        base_dir = Path(state.execution_telemetry.get("output_dir", Path("outputs/runs") / state.session_id))
        out_dir = base_dir / "01_separated_convtasnet"

        stems, meta = self.separate_file(input_file, output_dir=out_dir)

        tracks = []
        for s in meta["stems"]:
            tracks.append(SeparatedTrack(
                source_index=s["source_index"],
                track_path=s["file_path"],
                energy_ratio=s["energy_ratio"],
                peak_amplitude=s["metrics"]["peak_amplitude"],
            ))

        state.separation_result = SeparationResult(
            input_audio_path=input_file,
            num_sources_detected=len(tracks),
            separation_model="conv_tasnet_base_libri2mix",
            isolated_tracks=tracks,
            execution_time_seconds=meta["inference_time_seconds"],
        )
        return state
