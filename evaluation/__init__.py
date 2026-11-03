"""Evaluation Metrics & Scientific Ablation Package (Milestone M9)."""

__all__ = [
    # BSS Metrics
    "calculate_sisdr",
    "calculate_sdr",
    "evaluate_separation_pit",
    # Diarization Metrics
    "DiarizationEvaluator",
    "DiarizationMetricsEvaluator",
    # ASR Metrics
    "ASREvaluator",
    "compute_wer",
    "compute_cer",
    "IndicTextNormalizer",
    # LLM Metrics
    "LLMMetricsEvaluator",
    # M9 Ablation & Error Propagation
    "AblationExperimentRunner",
    "AblationMetrics",
    "PipelineVariant",
    "ErrorCascadingAnalyzer",
    "ErrorCascadeDataPoint",
    "CorrelationSummary",
    "CANONICAL_FAILURE_TAXONOMY",
    "FailureDiagnosticEngine",
    "FailureTaxonomyItem",
]

# LLM metrics
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
    from evaluation.diarization_metrics import (
        DiarizationEvaluator,
        DiarizationMetricsEvaluator,
    )
except ImportError:
    pass

# ASR metrics
try:
    from evaluation.asr_metrics import (
        ASREvaluator,
        IndicTextNormalizer,
        compute_cer,
        compute_wer,
    )
except ImportError:
    pass

# M9 Ablations, Error Cascading & Failure Taxonomy
try:
    from evaluation.ablation_runner import (
        AblationExperimentRunner,
        AblationMetrics,
        PipelineVariant,
    )
    from evaluation.error_propagation import (
        CorrelationSummary,
        ErrorCascadeDataPoint,
        ErrorCascadingAnalyzer,
    )
    from evaluation.failure_taxonomy import (
        CANONICAL_FAILURE_TAXONOMY,
        FailureDiagnosticEngine,
        FailureTaxonomyItem,
    )
except ImportError:
    pass
