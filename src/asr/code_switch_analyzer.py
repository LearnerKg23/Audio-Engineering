"""Code-switching and script-mix diagnostics for bilingual Indic/Latin transcripts.

Computes script distribution, code-switch boundary counts, hyphenated token
counts, and zero-width joiner (ZWJ/ZWNJ) occurrences across utterances or
normalized strings to support Stage 3/4 error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, List, Optional, Sequence, Union

from src.core.state import PipelineState, RawTranscript, Utterance


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9\u0966-\u096F]")
_ZWJ_ZWNJ_RE = re.compile(r"[\u200c\u200d]")


@dataclass(frozen=True)
class CodeSwitchStats:
    """Summary statistics for script mixing and code-switching in text."""

    total_tokens: int
    devanagari_tokens: int
    latin_tokens: int
    mixed_script_tokens: int
    hyphenated_tokens: int
    code_switch_transitions: int
    zwj_zwnj_count: int
    latin_ratio: float
    code_switch_rate: float
    sample_switch_pairs: List[tuple[str, str]] = field(default_factory=list)


def _classify_token_script(token: str) -> str:
    """Classify a whitespace-delimited token by script composition."""
    has_dev = bool(_DEVANAGARI_RE.search(token))
    has_lat = bool(_LATIN_RE.search(token))
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "latin"
    if _DIGIT_RE.search(token):
        return "digit"
    return "other"


def analyze_code_switching(
    text: str,
    max_samples: int = 5,
) -> CodeSwitchStats:
    """Analyze code-switch transitions and script balance in a transcript string.

    Args:
        text: Input transcript or normalized text string.
        max_samples: Maximum number of adjacent script-switch token pairs to record.

    Returns:
        CodeSwitchStats: Aggregated token counts, script ratios, and transition samples.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If ``max_samples`` is negative.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str for text, got {type(text).__name__}")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")

    zwj_zwnj_count = len(_ZWJ_ZWNJ_RE.findall(text))
    raw_tokens = [tok for tok in text.strip().split() if tok]

    if not raw_tokens:
        return CodeSwitchStats(
            total_tokens=0,
            devanagari_tokens=0,
            latin_tokens=0,
            mixed_script_tokens=0,
            hyphenated_tokens=0,
            code_switch_transitions=0,
            zwj_zwnj_count=zwj_zwnj_count,
            latin_ratio=0.0,
            code_switch_rate=0.0,
            sample_switch_pairs=[],
        )

    dev_count = 0
    lat_count = 0
    mixed_count = 0
    hyphen_count = 0
    transitions = 0
    switch_samples: List[tuple[str, str]] = []

    prev_script: Optional[str] = None
    prev_token: Optional[str] = None

    for tok in raw_tokens:
        if "-" in tok and len(tok) > 1:
            hyphen_count += 1

        script = _classify_token_script(tok)
        if script == "devanagari":
            dev_count += 1
        elif script == "latin":
            lat_count += 1
        elif script == "mixed":
            mixed_count += 1

        # Track transitions across lexical scripts (excluding pure digits/symbols)
        if script in {"devanagari", "latin", "mixed"}:
            if prev_script is not None and prev_script != script:
                transitions += 1
                if prev_token is not None and len(switch_samples) < max_samples:
                    switch_samples.append((prev_token, tok))
            prev_script = script
            prev_token = tok

    total = len(raw_tokens)
    latin_ratio = round((lat_count + mixed_count) / total, 4)
    possible_steps = max(1, total - 1)
    cs_rate = round(transitions / possible_steps, 4)

    return CodeSwitchStats(
        total_tokens=total,
        devanagari_tokens=dev_count,
        latin_tokens=lat_count,
        mixed_script_tokens=mixed_count,
        hyphenated_tokens=hyphen_count,
        code_switch_transitions=transitions,
        zwj_zwnj_count=zwj_zwnj_count,
        latin_ratio=latin_ratio,
        code_switch_rate=cs_rate,
        sample_switch_pairs=switch_samples,
    )


def analyze_transcript_corpus(
    source: Union[PipelineState, RawTranscript, Sequence[Utterance], Iterable[str]],
) -> CodeSwitchStats:
    """Compute corpus-level code-switch statistics across utterances or text items.

    Args:
        source: A ``PipelineState``, ``RawTranscript``, sequence of ``Utterance``
            objects, or an iterable of text strings.

    Returns:
        CodeSwitchStats: Combined statistics across all provided utterances/texts.

    Raises:
        TypeError: If ``source`` is not a supported transcript container or iterable.
    """
    texts: List[str] = []

    if isinstance(source, PipelineState):
        if source.raw_transcript:
            texts.extend(utt.raw_text for utt in source.raw_transcript.utterances)
    elif isinstance(source, RawTranscript):
        texts.extend(utt.raw_text for utt in source.utterances)
    elif isinstance(source, str):
        texts.append(source)
    elif isinstance(source, Iterable):
        for item in source:
            if isinstance(item, Utterance):
                texts.append(item.raw_text)
            elif isinstance(item, str):
                texts.append(item)
            else:
                raise TypeError(
                    f"Unsupported item type in iterable: {type(item).__name__}"
                )
    else:
        raise TypeError(f"Unsupported transcript source type: {type(source).__name__}")

    combined = "\n".join(t for t in texts if t and t.strip())
    return analyze_code_switching(combined)
