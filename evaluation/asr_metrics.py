"""
ASR Evaluation Metrics (Stage 3).

Implements rigorous Word Error Rate (WER) and Character Error Rate (CER) calculations
using dynamic programming Levenshtein distance with complete error breakdown
(Substitutions, Deletions, Insertions), as mandated by Milestone M6 and 08_EVALUATION_STRATEGY.md.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from src.asr.text_normalizer import IndicTextNormalizer, normalize_indic_text


@dataclass
class ErrorBreakdown:
    """Detailed error counts from Levenshtein sequence alignment."""
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    error_rate: float
    error_percent: float


def compute_levenshtein_alignment(
    ref_tokens: List[str],
    hyp_tokens: List[str],
) -> Tuple[int, int, int, int, List[Tuple[str, Optional[str], Optional[str]]]]:
    """
    Computes exact Wagner-Fischer dynamic programming edit distance.
    
    Returns:
        (total_errors, substitutions, deletions, insertions, alignment_ops)
        where alignment_ops is a list of (op_type, ref_token, hyp_token).
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    if n == 0:
        ops = [("insert", None, token) for token in hyp_tokens]
        return m, 0, 0, m, ops
    if m == 0:
        ops = [("delete", token, None) for token in ref_tokens]
        return n, 0, n, 0, ops

    # dp[i][j] stores (cost, substitutions, deletions, insertions)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                cost_sub = dp[i - 1][j - 1][0] + 1
                state_sub = (
                    cost_sub,
                    dp[i - 1][j - 1][1] + 1,
                    dp[i - 1][j - 1][2],
                    dp[i - 1][j - 1][3],
                )

                cost_del = dp[i - 1][j][0] + 1
                state_del = (
                    cost_del,
                    dp[i - 1][j][1],
                    dp[i - 1][j][2] + 1,
                    dp[i - 1][j][3],
                )

                cost_ins = dp[i][j - 1][0] + 1
                state_ins = (
                    cost_ins,
                    dp[i][j - 1][1],
                    dp[i][j - 1][2],
                    dp[i][j - 1][3] + 1,
                )

                best = min([state_sub, state_del, state_ins], key=lambda x: x[0])
                dp[i][j] = best

    total_cost, s_count, d_count, i_count = dp[n][m]

    # Backtrack alignment path
    i, j = n, m
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            ops.append(("match", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif (
            i > 0
            and j > 0
            and dp[i][j][0] == dp[i - 1][j - 1][0] + 1
            and dp[i][j][1] == dp[i - 1][j - 1][1] + 1
        ):
            ops.append(("substitute", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j][0] == dp[i - 1][j][0] + 1 and dp[i][j][2] == dp[i - 1][j][2] + 1:
            ops.append(("delete", ref_tokens[i - 1], None))
            i -= 1
        else:
            ops.append(("insert", None, hyp_tokens[j - 1]))
            j -= 1

    ops.reverse()
    return total_cost, s_count, d_count, i_count, ops


def compute_wer(
    reference: str,
    hypothesis: str,
    normalizer: Optional[IndicTextNormalizer] = None,
) -> Dict[str, Any]:
    """
    Computes Word Error Rate (WER) between reference and hypothesis text.
    
    WER = (S + D + I) / N
    
    Args:
        reference: Ground truth transcript.
        hypothesis: Predicted ASR transcript.
        normalizer: IndicTextNormalizer instance. If None, default normalizer is used.
        
    Returns:
        Dictionary with wer, wer_percent, substitutions, deletions, insertions, hits.
    """
    norm = normalizer or IndicTextNormalizer()
    ref_words = norm.tokenize_words(reference)
    hyp_words = norm.tokenize_words(hypothesis)

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        wer_val = 1.0 if m > 0 else 0.0
        return {
            "wer": wer_val,
            "wer_percent": round(wer_val * 100.0, 2),
            "substitutions": 0,
            "deletions": 0,
            "insertions": m,
            "hits": 0,
            "ref_words": 0,
            "hyp_words": m,
        }

    total_cost, s, d, i, _ = compute_levenshtein_alignment(ref_words, hyp_words)
    hits = n - (s + d)
    wer_val = round(float(total_cost) / float(n), 4)

    return {
        "wer": wer_val,
        "wer_percent": round(wer_val * 100.0, 2),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": max(0, hits),
        "ref_words": n,
        "hyp_words": m,
    }


def compute_cer(
    reference: str,
    hypothesis: str,
    normalizer: Optional[IndicTextNormalizer] = None,
) -> Dict[str, Any]:
    """
    Computes Character Error Rate (CER) between reference and hypothesis text.
    Critical for Indic languages where matras and conjuncts make WER overly punitive.
    
    CER = (S_c + D_c + I_c) / N_c
    """
    norm = normalizer or IndicTextNormalizer()
    ref_chars = norm.tokenize_characters(reference)
    hyp_chars = norm.tokenize_characters(hypothesis)

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        cer_val = 1.0 if m > 0 else 0.0
        return {
            "cer": cer_val,
            "cer_percent": round(cer_val * 100.0, 2),
            "substitutions": 0,
            "deletions": 0,
            "insertions": m,
            "hits": 0,
            "ref_chars": 0,
            "hyp_chars": m,
        }

    total_cost, s, d, i, _ = compute_levenshtein_alignment(ref_chars, hyp_chars)
    hits = n - (s + d)
    cer_val = round(float(total_cost) / float(n), 4)

    return {
        "cer": cer_val,
        "cer_percent": round(cer_val * 100.0, 2),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": max(0, hits),
        "ref_chars": n,
        "hyp_chars": m,
    }


def compute_corpus_metrics(
    references: List[str],
    hypotheses: List[str],
    normalizer: Optional[IndicTextNormalizer] = None,
) -> Dict[str, Any]:
    """
    Computes both Micro (aggregated tokens) and Macro (average per sample) WER and CER
    across a multi-utterance benchmark suite.
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses lists must have identical lengths.")

    total_ref_words = 0
    total_word_errors = 0
    total_ref_chars = 0
    total_char_errors = 0

    sample_wers: List[float] = []
    sample_cers: List[float] = []

    for ref, hyp in zip(references, hypotheses):
        w_res = compute_wer(ref, hyp, normalizer)
        c_res = compute_cer(ref, hyp, normalizer)

        total_ref_words += w_res["ref_words"]
        total_word_errors += w_res["substitutions"] + w_res["deletions"] + w_res["insertions"]

        total_ref_chars += c_res["ref_chars"]
        total_char_errors += c_res["substitutions"] + c_res["deletions"] + c_res["insertions"]

        sample_wers.append(w_res["wer_percent"])
        sample_cers.append(c_res["cer_percent"])

    micro_wer = round(total_word_errors / max(1, total_ref_words), 4)
    micro_cer = round(total_char_errors / max(1, total_ref_chars), 4)
    macro_wer = round(sum(sample_wers) / max(1, len(sample_wers)), 2)
    macro_cer = round(sum(sample_cers) / max(1, len(sample_cers)), 2)

    return {
        "total_samples": len(references),
        "micro_wer": micro_wer,
        "micro_wer_percent": round(micro_wer * 100.0, 2),
        "macro_wer_percent": macro_wer,
        "micro_cer": micro_cer,
        "micro_cer_percent": round(micro_cer * 100.0, 2),
        "macro_cer_percent": macro_cer,
        "total_ref_words": total_ref_words,
        "total_ref_chars": total_ref_chars,
    }
