"""
Pipeline State & Data Contracts for the Speech Engineering Architecture.

Enforces strongly typed Pydantic models for all inter-stage artifacts
as defined in 02_SYSTEM_ARCHITECTURE.md.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AudioMetadata(BaseModel):
    """Metadata describing an ingested and normalized audio asset."""
    source_file: str
    sample_rate_hz: int = 16000
    channels: int = 1
    duration_seconds: float
    bit_depth: int = 16
    rms_energy_db: Optional[float] = None
    snr_estimate_db: Optional[float] = None


class SeparatedTrack(BaseModel):
    """Individual audio track isolated by the Blind Source Separation module."""
    source_index: int
    track_path: str
    energy_ratio: float = Field(ge=0.0, le=1.0)
    peak_amplitude: float = 0.0


class SeparationResult(BaseModel):
    """Output artifact manifest for Stage 1: Blind Source Separation."""
    model_config = ConfigDict(protected_namespaces=())

    input_audio_path: str
    num_sources_detected: int
    separation_model: str = "htdemucs"
    isolated_tracks: List[SeparatedTrack] = Field(default_factory=list)
    execution_time_seconds: float = 0.0


class DiarizationSegment(BaseModel):
    """Homogeneous speaker segment with start/end boundaries."""
    segment_id: int
    speaker_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class DiarizationTimeline(BaseModel):
    """Output artifact for Stage 2: Acoustic Speaker Diarization."""
    audio_file: str
    algorithm: str = "pyannote_spectral"
    total_speakers_detected: int = 0
    segments: List[DiarizationSegment] = Field(default_factory=list)


class Utterance(BaseModel):
    """Speaker-attributed speech-to-text transcription segment."""
    utterance_id: int
    speaker_id: str
    start_time: float
    end_time: float
    detected_language: str = "hi"
    raw_text: str
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)


class RawTranscript(BaseModel):
    """Output artifact for Stage 3: Regional & Code-Switched ASR."""
    model_config = ConfigDict(protected_namespaces=())

    pipeline_stage: str = "ASR"
    model_name: str = "indic_wav2vec"
    utterances: List[Utterance] = Field(default_factory=list)


class KeyFinding(BaseModel):
    """Structured actionable finding extracted by the LLM."""
    topic: str
    detail: str
    speaker_attribution: str


class SpeakerDialogue(BaseModel):
    """Polished, disfluency-cleaned dialogue turn."""
    speaker: str
    timestamp: str
    cleaned_text: str


class PostProcessedReport(BaseModel):
    """Output artifact for Stage 4: Semantic Post-Processing & Structuring."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    executive_summary: str
    primary_topics: List[str] = Field(default_factory=list)
    keyword_tags: List[str] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    speaker_separated_dialogue: List[SpeakerDialogue] = Field(default_factory=list)


class RAGAnswer(BaseModel):
    """Output artifact for Stage 4 Retrieval-Augmented Generation (RAG) query over conversation."""
    query: str
    answer: str
    is_grounded: bool = True
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    cited_speakers: List[str] = Field(default_factory=list)
    relevant_timestamps: List[str] = Field(default_factory=list)
    guardrail_status: str = "PASSED"


class PipelineState(BaseModel):
    """
    Immutable Pipeline Context passed through all modules.
    Tracks state transitions, artifacts, telemetry, and error events.
    """
    session_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
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
        """Serializes complete pipeline state to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, input_path: str) -> "PipelineState":
        """Deserializes pipeline state from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
