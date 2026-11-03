"""Evaluation Metrics Package."""

__all__ = [
    "calculate_sisdr",
    "calculate_sdr",
    "evaluate_separation_pit",
    "DiarizationEvaluator",
    "ASREvaluator",
    "compute_wer",
    "compute_cer",
    "LLMMetricsEvaluator",
]

# LLM metrics (pure Python, zero heavy DL dependencies)
try:
    from evaluation.llm_metrics import LLMMetricsEvaluator
except ImportError:
    pass

# BSS metrics
try:
    from evaluation.bss_metrics import (
        calculate_sdr,
        calculate_sisdr,
        evaluate_separation_pit,
    )
except ImportError:
    pass

# Diarization metrics
try:
    from evaluation.diarization_metrics import DiarizationEvaluator
except ImportError:
    pass

# ASR metrics
try:
    from evaluation.asr_metrics import ASREvaluator, compute_cer, compute_wer
except ImportError:
    pass
