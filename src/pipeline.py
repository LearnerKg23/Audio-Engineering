"""
Master End-to-End Speech Engineering Pipeline Orchestrator (Milestone M8).

Unifies Stages 0 through 4 into a cohesive, production-ready pipeline:
Stage 0: Audio Ingestion, Resampling (16 kHz), Peak/Loudness Normalization, and VAD.
Stage 1: Blind Source Separation (Conv-TasNet / Demucs) for decoupling overlapping speech.
Stage 2: Unsupervised Speaker Diarization (Mel/Spectral Embeddings + Eigengap Clustering).
Stage 3: Regional & Code-Switched ASR (AI4Bharat IndicWav2Vec / Whisper).
Stage 4: Semantic Post-Processing & Structural LLM Guardrails (Contract 5 Report).

Features:
- Strong typing via Pydantic PipelineState and inter-stage Contracts 1-5.
- Multi-tier intermediate artifact caching to avoid expensive re-computation.
- Graceful single-speaker bypass for Stage 1.
- Complete execution telemetry (RTF, wall-clock latency, peak memory).
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from src.asr.asr_engine import ASREngine
from src.core.audio_io import compute_audio_metrics, load_audio
from src.core.base_module import resolve_device
from src.core.state import (
    AudioMetadata,
    DiarizationSegment,
    DiarizationTimeline,
    PipelineState,
    PostProcessedReport,
    RawTranscript,
    SeparatedTrack,
    SeparationResult,
    Utterance,
)
from src.diarization.diarizer import SpeakerDiarizer
from src.llm.llm_engine import LLMEngine
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.separation.conv_tasnet_wrapper import ConvTasNetSeparator
from src.separation.demucs_wrapper import DemucsSeparator

logger = logging.getLogger("speech_pipeline")


class SpeechEngineeringPipeline:
    """
    Master pipeline orchestrating end-to-end multi-speaker Indian conversational speech processing.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "auto",
        enable_caching: bool = True,
        mock_mode: bool = False,
    ):
        self.config = self._load_configuration(config_path, config_dict)
        self.device = resolve_device(device or self.config.get("system", {}).get("device", "auto"))
        self.enable_caching = enable_caching
        self.mock_mode = mock_mode or self.config.get("asr", {}).get("mock_mode", False)
        self.target_sr = self.config.get("system", {}).get("target_sample_rate", 16000)

        # Initialize submodules
        self._init_submodules()
        self._is_initialized = False

    def _load_configuration(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Loads and merges configuration YAML and dictionary overrides."""
        cfg = {}
        default_yaml = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
        target_path = Path(config_path) if config_path else default_yaml

        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load config file at {target_path}: {e}")

        if config_dict:
            # Deep merge config_dict over cfg
            for k, v in config_dict.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v

        return cfg

    def _init_submodules(self) -> None:
        """Instantiates all submodules conforming to the configuration."""
        prep_cfg = self.config.get("preprocessing", {})
        self.preprocessor = AudioPreprocessor(
            target_sr=self.target_sr,
            target_rms_db=prep_cfg.get("target_rms_db", -20.0),
            max_peak_amplitude=prep_cfg.get("target_peak_db", -1.0),
        )

        sep_model = self.config.get("separation", {}).get("model_name", "conv_tasnet").lower()
        if "demucs" in sep_model:
            self.separator = DemucsSeparator(config=self.config, device=str(self.device))
        else:
            self.separator = ConvTasNetSeparator(config=self.config, device=str(self.device))

        self.diarizer = SpeakerDiarizer(config=self.config, device=str(self.device))
        self.asr_engine = ASREngine(config=self.config, device=str(self.device), mock_mode=self.mock_mode)
        self.llm_engine = LLMEngine(config=self.config, device=str(self.device))

    def initialize(self) -> None:
        """Explicitly initializes and allocates model weights across stages."""
        if self._is_initialized:
            return
        t0 = time.perf_counter()
        logger.info("Initializing SpeechEngineeringPipeline stages...")
        if self.config.get("separation", {}).get("enabled", True):
            self.separator.initialize()
        if self.config.get("diarization", {}).get("enabled", True):
            self.diarizer.initialize()
        if self.config.get("asr", {}).get("enabled", True):
            self.asr_engine.initialize()
        if self.config.get("llm", {}).get("enabled", True):
            self.llm_engine.initialize()
        self._is_initialized = True
        logger.info(f"SpeechEngineeringPipeline initialized in {time.perf_counter() - t0:.2f}s")

    def run(
        self,
        input_audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        bypass_separation: bool = False,
        bypass_diarization: bool = False,
        bypass_asr: bool = False,
        bypass_llm: bool = False,
        force_recompute: bool = False,
        save_artifacts: bool = True,
    ) -> PipelineState:
        """
        Executes the full end-to-end speech engineering pipeline on the input audio file.

        Args:
            input_audio_path: Path to input audio (.wav, .mp3, etc.)
            output_dir: Directory to save intermediate stems, manifests, and reports
            session_id: Optional tracking identifier for this pipeline run
            bypass_separation: If True, bypasses Stage 1 BSS (single-speaker or raw baseline)
            bypass_diarization: If True, bypasses Stage 2 Diarization
            bypass_asr: If True, bypasses Stage 3 ASR
            bypass_llm: If True, bypasses Stage 4 LLM semantic structuring
            force_recompute: If True, ignores cached artifacts and runs fresh inference
            save_artifacts: If True, serializes all Contract 1-5 JSONs and markdown report
        """
        if not self._is_initialized:
            self.initialize()

        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        session_id = session_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{input_path.stem}"
        base_output_dir = Path(output_dir or Path(self.config.get("system", {}).get("output_base_dir", "outputs/runs")) / session_id)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        state = PipelineState(
            session_id=session_id,
            input_audio_path=str(input_path),
        )
        state.execution_telemetry["output_dir"] = str(base_output_dir)
        pipeline_start_time = time.perf_counter()

        logger.info(f"=== Starting Pipeline Run [{session_id}] on {input_path.name} ===")

        # ======================================================================
        # STAGE 0: Ingestion & Acoustic Preprocessing
        # ======================================================================
        t_stage0 = time.perf_counter()
        norm_audio_path = base_output_dir / "00_preprocessed" / f"{input_path.stem}_normalized.wav"
        meta_json_path = base_output_dir / "metadata.json"

        if self.enable_caching and not force_recompute and norm_audio_path.exists() and meta_json_path.exists():
            logger.info("  [Cache Hit] Reusing preprocessed audio and metadata.")
            with open(meta_json_path, "r", encoding="utf-8") as f:
                state.audio_metadata = AudioMetadata(**json.load(f))
            state.normalized_audio_path = str(norm_audio_path)
        else:
            norm_audio_path.parent.mkdir(parents=True, exist_ok=True)
            prep_res = self.preprocessor.preprocess_file(input_path, output_path=norm_audio_path)
            state.normalized_audio_path = str(norm_audio_path)
            state.audio_metadata = AudioMetadata(
                source_file=str(input_path),
                sample_rate_hz=self.preprocessor.target_sr,
                channels=1,
                duration_seconds=prep_res["preprocessed_metrics"]["duration_seconds"],
                bit_depth=16,
                rms_energy_db=prep_res["preprocessed_metrics"]["rms_energy_db"],
                snr_estimate_db=prep_res["preprocessed_metrics"]["snr_estimate_db"],
            )

        duration_sec = state.audio_metadata.duration_seconds if state.audio_metadata else 1.0
        state.execution_telemetry["stage0_preprocessing"] = {
            "elapsed_seconds": round(time.perf_counter() - t_stage0, 4),
            "duration_seconds": duration_sec,
            "sample_rate": self.target_sr,
            "status": "SUCCESS",
        }

        # ======================================================================
        # STAGE 1: Blind Source Separation (BSS)
        # ======================================================================
        t_stage1 = time.perf_counter()
        sep_manifest_path = base_output_dir / "separation_manifest.json"

        if bypass_separation or not self.config.get("separation", {}).get("enabled", True):
            logger.info("  [Bypassed] Stage 1 Blind Source Separation bypassed.")
            state.separation_result = None
            state.execution_telemetry["stage1_separation"] = {
                "bypassed": True,
                "reason": "Explicit flag or single-speaker bypass",
                "elapsed_seconds": 0.0,
            }
        elif self.enable_caching and not force_recompute and sep_manifest_path.exists():
            logger.info("  [Cache Hit] Reusing Stage 1 Separation Result.")
            with open(sep_manifest_path, "r", encoding="utf-8") as f:
                state.separation_result = SeparationResult(**json.load(f))
            state.execution_telemetry["stage1_separation"] = {
                "cached": True,
                "elapsed_seconds": round(time.perf_counter() - t_stage1, 4),
            }
        else:
            logger.info("  [Executing] Stage 1 Blind Source Separation...")
            state = self.separator.process(state)
            state.execution_telemetry["stage1_separation"] = {
                "elapsed_seconds": round(time.perf_counter() - t_stage1, 4),
                "num_sources": state.separation_result.num_sources_detected if state.separation_result else 0,
                "rtf": round((time.perf_counter() - t_stage1) / duration_sec, 4),
                "status": "SUCCESS",
            }

        # ======================================================================
        # STAGE 2: Acoustic Speaker Diarization
        # ======================================================================
        t_stage2 = time.perf_counter()
        diar_path = base_output_dir / "diarization_timeline.json"

        if bypass_diarization or not self.config.get("diarization", {}).get("enabled", True):
            logger.info("  [Bypassed] Stage 2 Speaker Diarization bypassed (defaulting to single speaker).")
            state.diarization_timeline = DiarizationTimeline(
                audio_file=state.normalized_audio_path or str(input_path),
                algorithm="single_speaker_default",
                total_speakers_detected=1,
                segments=[
                    DiarizationSegment(
                        segment_id=0,
                        speaker_id="SPEAKER_00",
                        start_time=0.0,
                        end_time=duration_sec,
                        confidence=1.0,
                    )
                ],
            )
            state.execution_telemetry["stage2_diarization"] = {
                "bypassed": True,
                "elapsed_seconds": 0.0,
            }
        elif self.enable_caching and not force_recompute and diar_path.exists():
            logger.info("  [Cache Hit] Reusing Stage 2 Diarization Timeline.")
            with open(diar_path, "r", encoding="utf-8") as f:
                state.diarization_timeline = DiarizationTimeline(**json.load(f))
            state.execution_telemetry["stage2_diarization"] = {
                "cached": True,
                "elapsed_seconds": round(time.perf_counter() - t_stage2, 4),
            }
        else:
            logger.info("  [Executing] Stage 2 Speaker Diarization...")
            state = self.diarizer.process(state)
            state.execution_telemetry["stage2_diarization"] = {
                "elapsed_seconds": round(time.perf_counter() - t_stage2, 4),
                "speakers_detected": state.diarization_timeline.total_speakers_detected if state.diarization_timeline else 1,
                "num_segments": len(state.diarization_timeline.segments) if state.diarization_timeline else 0,
                "rtf": round((time.perf_counter() - t_stage2) / duration_sec, 4),
                "status": "SUCCESS",
            }

        # ======================================================================
        # STAGE 3: Regional & Code-Switched ASR
        # ======================================================================
        t_stage3 = time.perf_counter()
        asr_path = base_output_dir / "raw_transcript.json"

        if bypass_asr or not self.config.get("asr", {}).get("enabled", True):
            logger.info("  [Bypassed] Stage 3 Automatic Speech Recognition bypassed.")
            state.raw_transcript = None
            state.execution_telemetry["stage3_asr"] = {
                "bypassed": True,
                "elapsed_seconds": 0.0,
            }
        elif self.enable_caching and not force_recompute and asr_path.exists():
            logger.info("  [Cache Hit] Reusing Stage 3 Raw Transcript.")
            with open(asr_path, "r", encoding="utf-8") as f:
                state.raw_transcript = RawTranscript(**json.load(f))
            state.execution_telemetry["stage3_asr"] = {
                "cached": True,
                "elapsed_seconds": round(time.perf_counter() - t_stage3, 4),
            }
        else:
            logger.info("  [Executing] Stage 3 Automatic Speech Recognition...")
            state = self.asr_engine.process(state)
            num_utterances = len(state.raw_transcript.utterances) if state.raw_transcript else 0
            state.execution_telemetry["stage3_asr"] = {
                "elapsed_seconds": round(time.perf_counter() - t_stage3, 4),
                "num_utterances": num_utterances,
                "rtf": round((time.perf_counter() - t_stage3) / duration_sec, 4),
                "status": "SUCCESS",
            }

        # ======================================================================
        # STAGE 4: Semantic Post-Processing & Guardrails
        # ======================================================================
        t_stage4 = time.perf_counter()
        report_json_path = base_output_dir / "final_report.json"

        if bypass_llm or not self.config.get("llm", {}).get("enabled", True):
            logger.info("  [Bypassed] Stage 4 LLM Semantic Post-Processing bypassed.")
            state.final_report = None
            state.execution_telemetry["stage4_llm"] = {
                "bypassed": True,
                "elapsed_seconds": 0.0,
            }
        elif self.enable_caching and not force_recompute and report_json_path.exists():
            logger.info("  [Cache Hit] Reusing Stage 4 Final Report.")
            with open(report_json_path, "r", encoding="utf-8") as f:
                state.final_report = PostProcessedReport(**json.load(f))
            state.execution_telemetry["stage4_llm"] = {
                "cached": True,
                "elapsed_seconds": round(time.perf_counter() - t_stage4, 4),
            }
        else:
            logger.info("  [Executing] Stage 4 LLM Semantic Post-Processing & Guardrails...")
            state = self.llm_engine.process(state)
            state.execution_telemetry["stage4_llm"] = {
                "elapsed_seconds": round(time.perf_counter() - t_stage4, 4),
                "status": "SUCCESS" if state.final_report else "FAILED",
            }

        # ======================================================================
        # Cumulative Pipeline Telemetry & Serialization
        # ======================================================================
        total_time = time.perf_counter() - pipeline_start_time
        total_rtf = total_time / duration_sec if duration_sec > 0 else 0.0

        state.execution_telemetry["total_pipeline_time_seconds"] = round(total_time, 4)
        state.execution_telemetry["audio_duration_seconds"] = round(duration_sec, 4)
        state.execution_telemetry["cumulative_rtf"] = round(total_rtf, 4)
        state.execution_telemetry["throughput_ratio"] = round(1.0 / total_rtf if total_rtf > 0 else 0.0, 2)
        state.execution_telemetry["completed_at"] = datetime.now(timezone.utc).isoformat()

        if save_artifacts:
            self.export_all_artifacts(state, base_output_dir)

        logger.info(
            f"=== Pipeline Finished [{session_id}] in {total_time:.2f}s "
            f"(RTF: {total_rtf:.4f}, Speed: {state.execution_telemetry['throughput_ratio']}x) ==="
        )

        return state

    def export_all_artifacts(self, state: PipelineState, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Exports all five formal contract artifacts and final markdown report.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        # 1. Metadata (Contract 1)
        if state.audio_metadata:
            meta_path = out_p / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(state.audio_metadata.model_dump_json(indent=2))
            paths["metadata"] = str(meta_path)

        # 2. Separation Manifest (Contract 2)
        if state.separation_result:
            sep_path = out_p / "separation_manifest.json"
            with open(sep_path, "w", encoding="utf-8") as f:
                f.write(state.separation_result.model_dump_json(indent=2))
            paths["separation_manifest"] = str(sep_path)

        # 3. Diarization Timeline (Contract 3)
        if state.diarization_timeline:
            diar_path = out_p / "diarization_timeline.json"
            with open(diar_path, "w", encoding="utf-8") as f:
                f.write(state.diarization_timeline.model_dump_json(indent=2))
            paths["diarization_timeline"] = str(diar_path)

        # 4. Raw Transcript (Contract 4)
        if state.raw_transcript:
            asr_path = out_p / "raw_transcript.json"
            with open(asr_path, "w", encoding="utf-8") as f:
                f.write(state.raw_transcript.model_dump_json(indent=2))
            paths["raw_transcript"] = str(asr_path)

        # 5. Final Report (Contract 5 - JSON & Markdown)
        if state.final_report:
            json_p, md_p = self.llm_engine.save_report_artifacts(state.final_report, out_p, file_prefix="final_report")
            paths["final_report_json"] = str(json_p)
            paths["final_report_md"] = str(md_p)

        # Master State Snapshot
        state_path = out_p / "pipeline_state.json"
        state.save_json(str(state_path))
        paths["pipeline_state"] = str(state_path)

        return paths

