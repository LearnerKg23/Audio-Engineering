"""
Acoustic Speaker Diarization Module (Stage 2).

Conforms to BaseModule interface, orchestrating:
1. Voice Activity Detection & Sliding-Window Embedding Extraction.
2. Unsupervised Clustering (Spectral with Eigengap auto-k / GMM).
3. Pipeline integration with Stage 1 (Separation-aware diarization).
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.audio_io import load_audio
from src.core.base_module import BaseModule
from src.core.state import DiarizationSegment, DiarizationTimeline, PipelineState
from src.diarization.clusterer import SpeakerClusterer
from src.diarization.embedder import SpeakerEmbeddingExtractor


class SpeakerDiarizer(BaseModule):
    """
    Unified Stage 2 Diarization Engine.
    Answers 'Who spoke when' chronologically without enrolled speaker templates.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "auto",
        clustering_method: str = "spectral",
        window_duration: float = 1.5,
        step_duration: float = 0.5,
        leakage_margin_db: float = 7.0,
    ):
        super().__init__(config=config, device=device)
        diar_cfg = self.config.get("diarization", {})

        self.clustering_method = diar_cfg.get("clustering_method", clustering_method)
        self.window_duration = diar_cfg.get("window_duration_seconds", window_duration)
        self.step_duration = diar_cfg.get("step_duration_seconds", step_duration)
        self.min_speakers = diar_cfg.get("min_speakers", 1)
        self.max_speakers = diar_cfg.get("max_speakers", 4)
        self.collar_seconds = diar_cfg.get("collar_seconds", 0.25)
        self.leakage_margin_db = diar_cfg.get("leakage_margin_db", leakage_margin_db)
        self.target_sr = self.config.get("target_sample_rate", 16000)

        self.extractor: Optional[SpeakerEmbeddingExtractor] = None
        self.clusterer: Optional[SpeakerClusterer] = None

    def initialize(self) -> None:
        """Initializes embedding extractor and clustering components."""
        if self._is_initialized:
            return

        self.extractor = SpeakerEmbeddingExtractor(
            embedding_dim=192,
            window_duration=self.window_duration,
            step_duration=self.step_duration,
            sample_rate=self.target_sr,
            device=self.device,
        )
        self.extractor.initialize()

        self.clusterer = SpeakerClusterer(
            clustering_method=self.clustering_method,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            min_segment_duration=self.window_duration * 0.5,
            collar_seconds=self.collar_seconds,
        )
        self._is_initialized = True

    def diarize_audio(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ) -> DiarizationTimeline:
        """
        Diarizes a single audio file.

        Args:
            audio_path: Path to .wav audio file.
            num_speakers: Optional ground-truth speaker count. If None,
                          estimated via the Eigengap heuristic.

        Returns:
            DiarizationTimeline with sorted chronological speaker segments.
        """
        self.initialize()
        audio_path_str = str(audio_path)

        timestamps, embeddings = self.extractor.extract_sliding_embeddings(audio_path_str)

        timeline = self.clusterer.cluster(
            timestamps=timestamps,
            embeddings=embeddings,
            audio_file=audio_path_str,
            num_speakers=num_speakers,
        )
        return timeline

    def diarize_separated_stems(
        self,
        stem_paths: List[Union[str, Path]],
        reference_audio_path: str,
    ) -> DiarizationTimeline:
        """
        Pipeline Diarization (Condition 2):
        Diarizes isolated single-speaker audio stems from Stage 1 source separation.
        Implements Cross-Stem Relative Energy Gating to eliminate cross-talk bleed
        artifacts and false-alarm speaker attribution during silent intervals.
        """
        self.initialize()
        if not stem_paths:
            return self.diarize_audio(reference_audio_path)

        if len(stem_paths) == 1:
            return self.diarize_audio(stem_paths[0])

        # Load all stem waveforms
        stem_waveforms: List[np.ndarray] = []
        for p in stem_paths:
            wav_tensor, _ = load_audio(p, target_sr=self.target_sr, to_mono=True)
            stem_waveforms.append(wav_tensor.squeeze(0).detach().cpu().numpy())

        min_len = min(len(w) for w in stem_waveforms)
        window_samples = int(self.window_duration * self.target_sr)
        step_samples = int(self.step_duration * self.target_sr)

        # Pre-compute peak RMS per stem for calibrated local VAD thresholding
        peak_dbs: List[float] = []
        for w in stem_waveforms:
            rmses = [
                np.sqrt(np.mean(w[s : s + window_samples] ** 2) + 1e-12)
                for s in range(0, min_len - window_samples + 1, step_samples)
            ]
            max_rms = max(rmses) if rmses else 1e-3
            peak_dbs.append(20.0 * np.log10(max(max_rms, 1e-6)))

        effective_threshs = [
            max(self.extractor.energy_threshold_db, p_db - 22.0)
            for p_db in peak_dbs
        ]

        raw_segments_by_stem: List[List[Tuple[float, float]]] = [[] for _ in stem_waveforms]

        start_idx = 0
        while start_idx + window_samples <= min_len:
            end_idx = start_idx + window_samples
            start_sec = round(start_idx / self.target_sr, 3)
            end_sec = round(end_idx / self.target_sr, 3)

            # Measure energy across all stems for this window
            chunk_dbs: List[float] = []
            for w in stem_waveforms:
                chunk = w[start_idx:end_idx]
                rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
                chunk_dbs.append(20.0 * np.log10(max(rms, 1e-6)))

            max_db = max(chunk_dbs)

            for stem_idx, db in enumerate(chunk_dbs):
                # VAD criterion 1: Must exceed stem's local noise floor threshold
                is_speech = db >= effective_threshs[stem_idx]

                # VAD criterion 2 (Cross-Stem Gating): Must not be residual cross-talk bleed
                # from a significantly louder concurrent stem
                if is_speech and (max_db - db > self.leakage_margin_db):
                    is_speech = False

                if is_speech:
                    raw_segments_by_stem[stem_idx].append((start_sec, end_sec))

            start_idx += step_samples

        # Smooth and merge contiguous segments per stem
        combined_segments: List[DiarizationSegment] = []
        for stem_idx, raw_segs in enumerate(raw_segments_by_stem):
            spk_tag = f"SPEAKER_{stem_idx:02d}"
            if not raw_segs:
                continue

            merged: List[Tuple[float, float]] = []
            cur_s, cur_e = raw_segs[0]
            for s, e in raw_segs[1:]:
                if s <= cur_e + 0.3:
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))

            for s, e in merged:
                combined_segments.append(
                    DiarizationSegment(
                        segment_id=0,
                        speaker_id=spk_tag,
                        start_time=s,
                        end_time=e,
                        confidence=0.96,
                    )
                )

        # Sort chronologically by start time
        combined_segments.sort(key=lambda s: (s.start_time, s.end_time))

        # Re-index segment IDs
        for idx, seg in enumerate(combined_segments):
            seg.segment_id = idx

        total_speakers = len(set(s.speaker_id for s in combined_segments))

        return DiarizationTimeline(
            audio_file=reference_audio_path,
            algorithm=f"{self.clustering_method}_separation_gated",
            total_speakers_detected=total_speakers,
            segments=combined_segments,
        )

    def process(self, state: PipelineState) -> PipelineState:
        """
        Executes Diarization stage on the incoming PipelineState.
        If Stage 1 produced separated tracks, leverages separation-aware diarization;
        otherwise runs on input/normalized audio.
        """
        self.initialize()

        # Check if Stage 1 Separation produced isolated tracks
        if (
            state.separation_result
            and state.separation_result.isolated_tracks
            and len(state.separation_result.isolated_tracks) > 1
        ):
            stem_paths = [t.track_path for t in state.separation_result.isolated_tracks]
            timeline = self.diarize_separated_stems(
                stem_paths=stem_paths,
                reference_audio_path=state.input_audio_path,
            )
        else:
            target_audio = state.normalized_audio_path or state.input_audio_path
            timeline = self.diarize_audio(target_audio)

        state.diarization_timeline = timeline
        state.execution_telemetry["SpeakerDiarizer_metrics"] = {
            "clustering_method": self.clustering_method,
            "speakers_detected": timeline.total_speakers_detected,
            "total_segments": len(timeline.segments),
        }
        return state

    def forward(self, state: PipelineState) -> PipelineState:
        """Alias for process for PyTorch-like consistency."""
        return self(state)
