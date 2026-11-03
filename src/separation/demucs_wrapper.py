"""
Demucs Blind Source Separation Module.

Wraps Meta's Hybrid Demucs (htdemucs) deep generative architecture for acoustic
speech separation and vocal isolation. Conforms to BaseModule lifecycle and
populates Contract 2 (SeparationResult) in PipelineState.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf
import torch

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio
from src.core.base_module import BaseModule
from src.core.state import PipelineState, SeparatedTrack, SeparationResult


class DemucsSeparator(BaseModule):
    """
    Production inference wrapper for Meta's Demucs separation model.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "auto",
        model_name: str = "htdemucs",
    ):
        super().__init__(config=config, device=device)
        self.model_name = self.config.get("model_name", model_name)
        self.model = None
        self.model_sr = 44100
        self.target_sr = self.config.get("target_sample_rate", 16000)

    def initialize(self) -> None:
        """Loads Demucs model weights and moves to target device."""
        if self._is_initialized:
            return

        from demucs.pretrained import get_model

        # Load pre-trained model (cached locally in ~/.cache/torch/hub/checkpoints)
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_sr = self.model.samplerate
        self._is_initialized = True

    def separate_audio(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000,
        output_dir: Optional[Union[str, Path]] = None,
        file_prefix: str = "separated",
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Executes Demucs separation forward pass on an audio tensor.

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

        # 1. Ensure 2D tensor (channels, samples)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 2. Resample to Demucs native model sample rate (44.1 kHz)
        if sample_rate != self.model_sr:
            # Polyphase resample via scipy for high anti-aliasing fidelity
            data = audio_tensor.cpu().numpy()
            gcd = np.gcd(sample_rate, self.model_sr)
            up = self.model_sr // gcd
            down = sample_rate // gcd
            resampled = [scipy.signal.resample_poly(data[c], up, down) for c in range(data.shape[0])]
            audio_tensor = torch.from_numpy(np.stack(resampled)).float()

        # Demucs expects stereo (2 channels). If mono, duplicate channel
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)

        # Shape: (batch=1, channels=2, samples)
        mixture_tensor = audio_tensor.unsqueeze(0).to(self.device)

        from demucs.apply import apply_model

        # 3. Model forward inference
        start_time = time.perf_counter()
        with torch.no_grad():
            # sources shape: (batch=1, num_sources, channels=2, samples)
            sources = apply_model(self.model, mixture_tensor, device=self.device, shifts=0, split=True)
        infer_duration = time.perf_counter() - start_time

        # Extract stems: sources is (num_sources, 2, samples)
        sources = sources.squeeze(0).cpu().numpy()
        source_names = self.model.sources # ['drums', 'bass', 'other', 'vocals']

        # 4. Downmix each stem to mono and resample back to target_sr (16 kHz)
        stems_16k = []
        stem_records = []
        gcd = np.gcd(self.model_sr, self.target_sr)
        up_back = self.target_sr // gcd
        down_back = self.model_sr // gcd

        for idx, name in enumerate(source_names):
            stem_stereo = sources[idx]
            stem_mono = np.mean(stem_stereo, axis=0) # Average stereo to mono
            stem_16k = scipy.signal.resample_poly(stem_mono, up_back, down_back).astype(np.float32)
            stems_16k.append(stem_16k)

        # 5. Acoustic stem grouping:
        # In speech mixtures, Demucs maps the dominant vocal harmonic energy to 'vocals',
        # and residual/overlapping speech and acoustic reflections to 'other' & background stems.
        vocals_idx = source_names.index("vocals") if "vocals" in source_names else 3
        other_idx = source_names.index("other") if "other" in source_names else 2

        vocal_stem = stems_16k[vocals_idx]
        other_stem = stems_16k[other_idx]

        # Primary isolated layers:
        isolated_layers = [vocal_stem, other_stem]
        layer_labels = ["source_0_vocal", "source_1_residual"]

        # Calculate energy distribution
        energies = [float(np.sum(s**2)) for s in isolated_layers]
        total_energy = sum(energies) + 1e-12
        ratios = [e / total_energy for e in energies]

        meta: Dict[str, Any] = {
            "model_name": self.model_name,
            "inference_time_seconds": round(infer_duration, 4),
            "num_stems": len(isolated_layers),
            "stems": [],
        }

        # 6. Save stems if output_dir specified
        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            for i, (stem, label, ratio) in enumerate(zip(isolated_layers, layer_labels, ratios)):
                stem_file = out_p / f"{file_prefix}_{label}.wav"
                save_audio(stem_file, stem, sample_rate=self.target_sr)
                meta["stems"].append({
                    "source_index": i,
                    "label": label,
                    "file_path": str(stem_file),
                    "energy_ratio": round(ratio, 4),
                    "metrics": compute_audio_metrics(stem, sample_rate=self.target_sr),
                })

        return isolated_layers, meta

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
        """Pipeline stage processing method."""
        input_file = state.normalized_audio_path or state.input_audio_path
        base_dir = Path(state.execution_telemetry.get("output_dir", Path("outputs/runs") / state.session_id))
        out_dir = base_dir / "01_separated"

        isolated_stems, meta = self.separate_file(input_file, output_dir=out_dir)

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
            separation_model=self.model_name,
            isolated_tracks=tracks,
            execution_time_seconds=meta["inference_time_seconds"],
        )
        return state
