"""
Failure Case Taxonomy & Diagnostic Engine (Milestone M9).

Formalizes the systematic taxonomy of failure modes across all 4 stages:
1. Stage 1 BSS Failures: Musical Noise, Phantom Voice Channels, Source Bleed-Through.
2. Stage 2 Diarization Failures: Speaker Ping-Ponging, Cluster Collapse, Oversegmentation.
3. Stage 3 ASR Failures: CTC Repetition Loops, Devanagari Encoding Mismatch, Code-Switch Dropping.
4. Stage 4 LLM Failures: Numerical Hallucination, Speaker Misattribution, Semantic Extrapolation.

Provides automated diagnostic routines to detect these failure modes within PipelineState.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from src.core.state import PipelineState

logger = logging.getLogger("failure_taxonomy")


class FailureCategory(str, Enum):
    BSS_FAILURE = "Stage 1: Blind Source Separation"
    DIARIZATION_FAILURE = "Stage 2: Speaker Diarization"
    ASR_FAILURE = "Stage 3: Regional ASR"
    LLM_FAILURE = "Stage 4: LLM Post-Processing"


@dataclass
class FailureTaxonomyItem:
    """Detailed definition of an individual failure mode, root cause, and mitigation."""
    failure_code: str
    category: FailureCategory
    name: str
    observable_phenomenon: str
    algorithmic_root_cause: str
    mitigation_strategy: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_code": self.failure_code,
            "category": self.category.value,
            "name": self.name,
            "observable_phenomenon": self.observable_phenomenon,
            "algorithmic_root_cause": self.algorithmic_root_cause,
            "mitigation_strategy": self.mitigation_strategy,
            "severity": self.severity,
        }


# Canonical Project Taxonomy Registry
CANONICAL_FAILURE_TAXONOMY: List[FailureTaxonomyItem] = [
    # Stage 1: BSS
    FailureTaxonomyItem(
        failure_code="ERR_BSS_01",
        category=FailureCategory.BSS_FAILURE,
        name="Musical Noise & Phase Inconsistency",
        observable_phenomenon="High-frequency chirping, metallic phase artifacts, or swirling watery timbre in separated stems.",
        algorithmic_root_cause="STFT phase cancellation or time-frequency mask discontinuities in Demucs/TasNet latent space.",
        mitigation_strategy="Apply Wiener filtering post-masking and smooth spectrogram magnitude thresholds with an energy floor.",
        severity="MEDIUM",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_BSS_02",
        category=FailureCategory.BSS_FAILURE,
        name="Phantom Voice Channel (Over-Allocation)",
        observable_phenomenon="A single active speaker is artificially split into two isolated tracks, one containing muffled echoes.",
        algorithmic_root_cause="Fixed source count K=2 forced onto single-speaker monologue recordings.",
        mitigation_strategy="Single-speaker bypass detection (--bypass-separation) and energy ratio thresholding (<-40 dB relative to peak).",
        severity="HIGH",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_BSS_03",
        category=FailureCategory.BSS_FAILURE,
        name="Severe Acoustic Bleed-Through",
        observable_phenomenon="Stem 1 retains loud residual background vocals from Speaker 2 during high overlap (>50%).",
        algorithmic_root_cause="Acoustic overlap complexity exceeds separation receptive field; negative SI-SDR improvement.",
        mitigation_strategy="Cascaded multi-pass separation or fallback to direct mixture diarization with overlap masking.",
        severity="CRITICAL",
    ),

    # Stage 2: Diarization
    FailureTaxonomyItem(
        failure_code="ERR_DIAR_01",
        category=FailureCategory.DIARIZATION_FAILURE,
        name="Speaker Ping-Ponging (Label Oscillation)",
        observable_phenomenon="A continuous uninterrupted sentence is rapidly fragmented into alternating SPEAKER_00 / SPEAKER_01 labels.",
        algorithmic_root_cause="Sliding window step duration too short (<0.5s) combined with natural pitch/formant inflections.",
        mitigation_strategy="Temporal boundary hysteresis, minimum turn-duration constraint (1.2s), and collar masking (0.25s).",
        severity="HIGH",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_DIAR_02",
        category=FailureCategory.DIARIZATION_FAILURE,
        name="Eigengap Cluster Collapse (K=1 Underestimation)",
        observable_phenomenon="Two distinct conversational speakers are lumped into a single SPEAKER_00 identity.",
        algorithmic_root_cause="Affinity matrix Laplacian eigenvalues lack sharp gap due to room reverberation or low SNR (<10 dB).",
        mitigation_strategy="Enforce minimum speaker prior (K_min=2) when Stage 1 detects multiple distinct acoustic sources.",
        severity="CRITICAL",
    ),

    # Stage 3: ASR
    FailureTaxonomyItem(
        failure_code="ERR_ASR_01",
        category=FailureCategory.ASR_FAILURE,
        name="CTC Silence Repetition Loop",
        observable_phenomenon="The ASR engine outputs the same Hindi word repeatedly (e.g., 'है है है है') over background static.",
        algorithmic_root_cause="Connectionist Temporal Classification (CTC) blank-token failure during unvoiced or reverberant audio slices.",
        mitigation_strategy="Energy-based VAD trimming before ASR decoding and repetition deduplication filters.",
        severity="HIGH",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_ASR_02",
        category=FailureCategory.ASR_FAILURE,
        name="Devanagari Unicode & Matra Inconsistency",
        observable_phenomenon="Phonetically correct Hindi words score artificially high WER due to Nukta or ZWJ glyph divergence.",
        algorithmic_root_cause="Mismatch between CTC character vocabulary canonicalization and reference Unicode encoding.",
        mitigation_strategy="Strict IndicTextNormalizer NFKC normalization and nukta canonicalization before evaluation.",
        severity="LOW",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_ASR_03",
        category=FailureCategory.ASR_FAILURE,
        name="Code-Switched Token Deletion",
        observable_phenomenon="English technical loanwords (e.g., 'rehab', 'fitness test') omitted or severely distorted into Hindi homophones.",
        algorithmic_root_cause="Monolingual Hindi acoustic prior suppressing Latin token emissions.",
        mitigation_strategy="Bilingual hybrid vocabulary decoding (IndicWav2Vec Hindi/English acoustic bundle).",
        severity="MEDIUM",
    ),

    # Stage 4: LLM Guardrails
    FailureTaxonomyItem(
        failure_code="ERR_LLM_01",
        category=FailureCategory.LLM_FAILURE,
        name="Numerical & Rate Extrapolation (Hallucination)",
        observable_phenomenon="LLM invents specific prices, interest rates, or percentages not spoken in the acoustic transcript.",
        algorithmic_root_cause="Autoregressive next-token completion hallucinating missing financial details on rural audio prompts.",
        mitigation_strategy="Explicit Negative Constraint Guardrail 1 (Rule 2) + Automated Factual Hallucination Auditor.",
        severity="CRITICAL",
    ),
    FailureTaxonomyItem(
        failure_code="ERR_LLM_02",
        category=FailureCategory.LLM_FAILURE,
        name="Dialogue Speaker Misattribution",
        observable_phenomenon="A statement uttered by Speaker 1 is attributed to Speaker 2 in the executive summary.",
        algorithmic_root_cause="Loose context window attention failing to track speaker prefixes across fragmented turns.",
        mitigation_strategy="Strict Pydantic schema validation requiring explicit speaker attribution keys aligned to Contract 3.",
        severity="HIGH",
    ),
]


class FailureDiagnosticEngine:
    """
    Scans a completed PipelineState to detect anomalies matching known failure taxonomy modes.
    """

    @staticmethod
    def inspect_state(state: PipelineState) -> List[Dict[str, Any]]:
        """
        Inspects PipelineState and returns a list of detected failure warnings.
        """
        detected: List[Dict[str, Any]] = []

        # 1. Check for BSS Phantom Channel (ERR_BSS_02)
        if state.separation_result and len(state.separation_result.isolated_tracks) >= 2:
            energies = [t.energy_ratio for t in state.separation_result.isolated_tracks]
            max_e = max(energies) if energies else 1.0
            min_e = min(energies) if energies else 0.0
            if max_e > 0 and (min_e / max_e) < 0.02:  # Less than 2% energy in track 2
                detected.append({
                    "failure_code": "ERR_BSS_02",
                    "severity": "HIGH",
                    "message": "Potential Phantom Voice Channel: Track 2 contains negligible acoustic energy relative to Track 1.",
                    "mitigation": "Recommend single-speaker bypass (--bypass-separation) for this recording.",
                })

        # 2. Check for Diarization Ping-Ponging (ERR_DIAR_01)
        if state.diarization_timeline and len(state.diarization_timeline.segments) > 4:
            segs = state.diarization_timeline.segments
            rapid_switches = 0
            for i in range(len(segs) - 1):
                if segs[i].speaker_id != segs[i + 1].speaker_id:
                    dur = segs[i].end_time - segs[i].start_time
                    if dur < 0.8:
                        rapid_switches += 1
            if rapid_switches >= 3:
                detected.append({
                    "failure_code": "ERR_DIAR_01",
                    "severity": "MEDIUM",
                    "message": f"Detected {rapid_switches} rapid speaker oscillations under 0.8s (Speaker Ping-Ponging).",
                    "mitigation": "Increase window step duration or apply temporal boundary smoothing.",
                })

        # 3. Check for ASR Silence Loops (ERR_ASR_01)
        if state.raw_transcript and state.raw_transcript.utterances:
            for u in state.raw_transcript.utterances:
                words = u.raw_text.split()
                if len(words) >= 6:
                    # Check if 70% of words are identical
                    most_common_count = max(words.count(w) for w in set(words))
                    if most_common_count / len(words) > 0.6:
                        detected.append({
                            "failure_code": "ERR_ASR_01",
                            "severity": "HIGH",
                            "message": f"ASR Repetition Loop detected in utterance {u.utterance_id}: '{u.raw_text[:40]}...'",
                            "mitigation": "Check VAD trimming or CTC silence blank thresholds.",
                        })

        return detected

