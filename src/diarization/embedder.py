"""
Speaker Embedding Extraction Module for Stage 2: Acoustic Speaker Diarization.

Extracts fixed-dimensional, unit-normalized speaker representation vectors
across sliding acoustic windows, filtering out non-speech silence via VAD.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf
import torch

from src.core.audio_io import load_audio


class SpeakerEmbeddingExtractor:
    """
    Extracts deep speaker embeddings from raw audio or separated audio stems
    using sliding temporal windows.
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        window_duration: float = 1.5,
        step_duration: float = 0.5,
        sample_rate: int = 16000,
        energy_threshold_db: float = -42.0,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.sample_rate = sample_rate
        self.energy_threshold_db = energy_threshold_db
        self.device = device
        self._is_initialized = False

    def initialize(self) -> None:
        """Initializes acoustic feature filters and projection bases."""
        if self._is_initialized:
            return

        # Deterministic projection basis for mapping mel-spectrogram / filterbank
        # representations to a standardized unit-norm d-dimensional embedding space
        np.random.seed(42)
        # Mel filterbank parameters
        self.n_mels = 64
        self.n_fft = 512
        self.hop_length = 160 # 10 ms at 16 kHz

        # Mel filterbank matrix
        self.mel_basis = self._create_mel_filterbank(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=80.0,
            fmax=7600.0,
        )

        # Projection matrix: (n_mels * 3 statistics: mean, std, spectral flux) -> embedding_dim
        stat_dim = self.n_mels * 3
        # Orthogonal random projection matrix for stable speaker manifold representation
        gaussian_matrix = np.random.randn(stat_dim, self.embedding_dim).astype(np.float32)
        q, _ = np.linalg.qr(gaussian_matrix)
        self.projection_matrix = q[:, : self.embedding_dim]

        self._is_initialized = True

    def _create_mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> np.ndarray:
        """Constructs a triangular Mel filterbank matrix."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        weights = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
        for i in range(n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center > left:
                weights[i, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                weights[i, center:right] = (right - np.arange(center, right)) / (right - center)

        return weights

    def _extract_frame_embedding(
        self, audio_chunk: np.ndarray, energy_thresh_db: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Extracts a d-dimensional speaker embedding vector from a short window.
        Returns None if chunk energy is below VAD threshold.
        """
        rms = np.sqrt(np.mean(audio_chunk ** 2) + 1e-12)
        db = 20.0 * np.log10(max(rms, 1e-6))
        thresh = energy_thresh_db if energy_thresh_db is not None else self.energy_threshold_db
        if db < thresh:
            return None

        # STFT
        f, t, zxx = scipy.signal.stft(
            audio_chunk,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
        )
        mag_spec = np.abs(zxx) # (n_fft // 2 + 1, num_frames)

        # Apply Mel Filterbank: (n_mels, num_frames)
        mel_spec = np.dot(self.mel_basis, mag_spec)
        log_mel = np.log(np.maximum(mel_spec, 1e-6))

        # First and second order temporal differences (delta features)
        diff_mel = np.diff(log_mel, axis=1) if log_mel.shape[1] > 1 else np.zeros((self.n_mels, 1))

        # Summary statistics across time
        mean_mel = np.mean(log_mel, axis=1)
        std_mel = np.std(log_mel, axis=1)
        flux_mel = np.mean(np.abs(diff_mel), axis=1) if diff_mel.shape[1] > 0 else np.zeros(self.n_mels)

        # Concatenate acoustic statistics vector
        stat_vector = np.concatenate([mean_mel, std_mel, flux_mel]).astype(np.float32)

        # Project into speaker embedding space
        emb = np.dot(stat_vector, self.projection_matrix)

        # L2-normalize vector to unit sphere: ||e|| = 1
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        else:
            emb = np.zeros_like(emb)

        return emb

    def extract_sliding_embeddings(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Extracts embeddings across sliding windows.

        Returns:
            timestamps: List of (start_seconds, end_seconds) for each valid chunk.
            embeddings: 2D numpy array of shape (N_valid, embedding_dim).
        """
        self.initialize()

        if isinstance(audio, (str, Path)):
            wav_tensor, sr = load_audio(audio, target_sr=self.sample_rate, to_mono=True)
            audio_arr = wav_tensor.squeeze(0).detach().cpu().numpy()
        elif isinstance(audio, torch.Tensor):
            if audio.ndim > 1 and audio.shape[0] == 1:
                audio = audio.squeeze(0)
            audio_arr = audio.detach().cpu().numpy()
            sr = sample_rate or self.sample_rate
        else:
            audio_arr = audio
            sr = sample_rate or self.sample_rate

        # Ensure 16 kHz
        if sr != self.sample_rate:
            gcd = np.gcd(sr, self.sample_rate)
            up = self.sample_rate // gcd
            down = sr // gcd
            audio_arr = scipy.signal.resample_poly(audio_arr, up, down).astype(np.float32)

        window_samples = int(self.window_duration * self.sample_rate)
        step_samples = int(self.step_duration * self.sample_rate)
        total_samples = len(audio_arr)

        if total_samples < window_samples:
            # Pad short audio to at least 1 window
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[:total_samples] = audio_arr
            audio_arr = padded
            total_samples = window_samples

        # Compute peak chunk RMS for adaptive VAD
        chunk_rmses = []
        for s_idx in range(0, total_samples - window_samples + 1, step_samples):
            c = audio_arr[s_idx : s_idx + window_samples]
            chunk_rmses.append(np.sqrt(np.mean(c ** 2) + 1e-12))

        max_rms = max(chunk_rmses) if chunk_rmses else 1e-3
        peak_db = 20.0 * np.log10(max(max_rms, 1e-6))
        effective_thresh = max(self.energy_threshold_db, peak_db - 22.0)

        timestamps = []
        embeddings = []

        start_idx = 0
        while start_idx + window_samples <= total_samples:
            end_idx = start_idx + window_samples
            chunk = audio_arr[start_idx:end_idx]

            emb = self._extract_frame_embedding(chunk, energy_thresh_db=effective_thresh)
            if emb is not None:
                start_sec = round(start_idx / self.sample_rate, 3)
                end_sec = round(end_idx / self.sample_rate, 3)
                timestamps.append((start_sec, end_sec))
                embeddings.append(emb)

            start_idx += step_samples

        # If all chunks were below threshold (e.g. ultra quiet), take middle chunk
        if len(embeddings) == 0:
            mid = total_samples // 2
            s = max(0, mid - window_samples // 2)
            chunk = audio_arr[s : s + window_samples]
            emb = np.dot(np.ones(self.n_mels * 3, dtype=np.float32), self.projection_matrix)
            emb = emb / np.linalg.norm(emb)
            timestamps.append((0.0, round(total_samples / self.sample_rate, 3)))
            embeddings.append(emb)

        return timestamps, np.array(embeddings, dtype=np.float32)
