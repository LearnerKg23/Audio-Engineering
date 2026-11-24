"""Shared diagnostics for separated audio tracks."""

from typing import Dict, List, Sequence, Union

import numpy as np
import torch


class SeparationDiagnosticsMixin:
    """Provide lightweight energy and activity diagnostics for separator output."""

    target_sr: int = 16000
    _ACTIVE_FRAME_RMS_THRESHOLD = 1e-2

    def compute_channel_energy_distribution(
        self,
        separated_wavs: Sequence[Union[np.ndarray, torch.Tensor]],
    ) -> List[Dict[str, Union[int, float]]]:
        """Calculate RMS level and active-frame ratio for each separated track.

        Each track is divided into 25 ms frames using the separator's target
        sample rate. A frame is counted as active when its RMS amplitude exceeds
        0.01 (approximately -40 dBFS). This is a lightweight energy heuristic,
        not a voice activity detector; the overall RMS is useful for spotting
        quiet or phantom tracks while the active ratio shows how consistently
        each track contains above-threshold signal.

        Args:
            separated_wavs: Ordered per-track waveforms, as NumPy arrays or
                Torch tensors. Each track is flattened to mono for measurement.

        Returns:
            One record per input track with its zero-based ``channel_index``,
            linear-amplitude ``rms_energy``, ``rms_energy_dbfs``, and
            ``active_speech_ratio`` in [0, 1].

        Raises:
            ValueError: If any track is empty or contains non-finite samples.
        """
        sample_rate = int(self.target_sr)
        if sample_rate <= 0:
            raise ValueError("target_sr must be a positive sample rate.")
        frame_length = max(1, round(sample_rate * 0.025))
        distribution = []

        for channel_index, waveform in enumerate(separated_wavs):
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            signal = np.asarray(waveform, dtype=np.float64).reshape(-1)
            if signal.size == 0:
                raise ValueError(f"Separated track {channel_index} must not be empty.")
            if not np.all(np.isfinite(signal)):
                raise ValueError(
                    f"Separated track {channel_index} must contain only finite samples."
                )

            rms_energy = float(np.sqrt(np.mean(np.square(signal))))
            frame_rms = [
                float(np.sqrt(np.mean(np.square(signal[start : start + frame_length]))))
                for start in range(0, signal.size, frame_length)
            ]
            active_speech_ratio = float(
                np.mean(np.asarray(frame_rms) > self._ACTIVE_FRAME_RMS_THRESHOLD)
            )
            distribution.append(
                {
                    "channel_index": channel_index,
                    "rms_energy": rms_energy,
                    "rms_energy_dbfs": float(20.0 * np.log10(max(rms_energy, 1e-12))),
                    "active_speech_ratio": active_speech_ratio,
                }
            )

        return distribution
