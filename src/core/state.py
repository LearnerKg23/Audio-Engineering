"""
Pipeline State & Data Contracts for the Speech Engineering Architecture.

Enforces strongly typed Pydantic v2 models for all inter-stage artifacts
as defined in 02_SYSTEM_ARCHITECTURE.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioMetadata(BaseModel):
    """Metadata describing an ingested and normalized audio asset."""

    model_config = ConfigDict(strict=False, extra="forbid")

    source_file: str
    sample_rate_hz: int = 16000
    channels: int = 1
    duration_seconds: float
    bit_depth: int = 16
    rms_energy_db: Optional[float] = None
    snr_estimate_db: Optional[float] = None


class SeparatedTrack(BaseModel):
    """Individual audio track isolated by the Blind Source Separation module."""

    model_config = ConfigDict(strict=False, extra="forbid")

    source_index: int
    track_path: str
    energy_ratio: float = Field(ge=0.0, le=1.0)
    peak_amplitude: float = 0.0


class SeparationResult(BaseModel):
    """Output artifact manifest for Stage 1: Blind Source Separation."""

    model_config = ConfigDict(
        protected_namespaces=(),
        strict=False,
        extra="forbid",
    )

    input_audio_path: str
    num_sources_detected: int
    separation_model: str = "htdemucs"
    isolated_tracks: List[SeparatedTrack] = Field(default_factory=list)
    execution_time_seconds: float = 0.0


class DiarizationSegment(BaseModel):
    """Homogeneous speaker segment with start/end boundaries."""

    model_config = ConfigDict(strict=False, extra="forbid")

    segment_id: int
    speaker_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> DiarizationSegment:
        """Ensure segment end time does not precede start time.

        Returns:
            DiarizationSegment: Validated segment instance.

        Raises:
            ValueError: If ``end_time`` is strictly less than ``start_time``.
        """
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})."
            )
        return self

    @property
    def duration(self) -> float:
        """Compute non-negative duration of the diarization segment in seconds.

        Returns:
            float: Segment duration in seconds.
        """
        return max(0.0, self.end_time - self.start_time)


class DiarizationTimeline(BaseModel):
    """Output artifact for Stage 2: Acoustic Speaker Diarization."""

    model_config = ConfigDict(strict=False, extra="forbid")

    audio_file: str
    algorithm: str = "pyannote_spectral"
    total_speakers_detected: int = 0
    segments: List[DiarizationSegment] = Field(default_factory=list)


class Utterance(BaseModel):
    """Speaker-attributed speech-to-text transcription segment."""

    model_config = ConfigDict(strict=False, extra="forbid")

    utterance_id: int
    speaker_id: str
    start_time: float
    end_time: float
    detected_language: str = "hi"
    raw_text: str
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)


class RawTranscript(BaseModel):
    """Output artifact for Stage 3: Regional & Code-Switched ASR."""

    model_config = ConfigDict(
        protected_namespaces=(),
        strict=False,
        extra="forbid",
    )

    pipeline_stage: str = "ASR"
    model_name: str = "indic_wav2vec"
    utterances: List[Utterance] = Field(default_factory=list)


class KeyFinding(BaseModel):
    """Structured actionable finding extracted by the LLM."""

    model_config = ConfigDict(strict=False, extra="forbid")

    topic: str
    detail: str
    speaker_attribution: str


class SpeakerDialogue(BaseModel):
    """Polished, disfluency-cleaned dialogue turn."""

    model_config = ConfigDict(strict=False, extra="forbid")

    speaker: str
    timestamp: str
    cleaned_text: str


class PostProcessedReport(BaseModel):
    """Output artifact for Stage 4: Semantic Post-Processing & Structuring."""

    model_config = ConfigDict(strict=False, extra="forbid")

    metadata: Dict[str, Any] = Field(default_factory=dict)
    executive_summary: str
    primary_topics: List[str] = Field(default_factory=list)
    keyword_tags: List[str] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    speaker_separated_dialogue: List[SpeakerDialogue] = Field(default_factory=list)


class RAGAnswer(BaseModel):
    """Output artifact for Stage 4 Retrieval-Augmented Generation (RAG) query over conversation."""

    model_config = ConfigDict(strict=False, extra="forbid")

    query: str
    answer: str
    is_grounded: bool = True
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    cited_speakers: List[str] = Field(default_factory=list)
    relevant_timestamps: List[str] = Field(default_factory=list)
    guardrail_status: str = "PASSED"


class PipelineState(BaseModel):
    """
    Pipeline Context passed through all modules.
    Tracks state transitions, artifacts, telemetry, and error events.
    """

    model_config = ConfigDict(strict=False, extra="forbid")

    session_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    input_audio_path: str
    normalized_audio_path: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    separation_result: Optional[SeparationResult] = None
    diarization_timeline: Optional[DiarizationTimeline] = None
    raw_transcript: Optional[RawTranscript] = None
    final_report: Optional[PostProcessedReport] = None
    rag_answers: List[RAGAnswer] = Field(default_factory=list)
    execution_telemetry: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    def save_json(self, output_path: str) -> None:
        """Serialize the complete pipeline state to a UTF-8 JSON file.

        Args:
            output_path: Destination filesystem path for the serialized JSON.

        Returns:
            None: Writes serialized state directly to disk.

        Raises:
            OSError: If parent directories cannot be created or the file cannot be written.
        """
        path: Path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized: str = self.model_dump_json(indent=2)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(serialized)

    @classmethod
    def load_json(cls, input_path: str) -> PipelineState:
        """Deserialize a ``PipelineState`` instance from a JSON file.

        Args:
            input_path: Source filesystem path of the JSON file.

        Returns:
            PipelineState: Validated ``PipelineState`` instance loaded from disk.

        Raises:
            OSError: If the file does not exist or cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            pydantic.ValidationError: If the JSON payload violates schema constraints.
        """
        path: Path = Path(input_path)
        with open(path, "r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
        return cls.model_validate(data)
