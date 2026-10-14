"""
Diarization Evaluation Metrics (DER and JER).

Implements:
1. NIST-standard Diarization Error Rate (DER) with boundary collar tolerance.
   DER = (Missed_Speech + False_Alarm + Speaker_Confusion) / Total_Reference_Speech
2. Jaccard Error Rate (JER) providing equal-weight speaker error calculation.
3. Optimal speaker assignment using the Hungarian algorithm (linear_sum_assignment).
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.state import DiarizationSegment, DiarizationTimeline


class DiarizationEvaluator:
    """
    Evaluates speaker diarization hypotheses against ground-truth timelines.
    """

    def __init__(self, collar_seconds: float = 0.25, time_step: float = 0.05):
        self.collar_seconds = collar_seconds
        self.time_step = time_step

    def _get_time_grid(
        self,
        ref_segments: List[Tuple[str, float, float]],
        hyp_segments: List[Tuple[str, float, float]],
    ) -> np.ndarray:
        """Constructs discrete time points spanning both reference and hypothesis."""
        max_time = 0.0
        for _, s, e in ref_segments:
            max_time = max(max_time, e)
        for _, s, e in hyp_segments:
            max_time = max(max_time, e)

        if max_time <= 0.0:
            return np.array([])
        return np.arange(0.0, max_time + self.time_step, self.time_step)

    def _build_collar_mask(
        self, ref_segments: List[Tuple[str, float, float]], time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Marks time grid points that fall within collar_seconds of any reference
        turn boundary (speech onset or offset).
        """
        collar_mask = np.zeros(len(time_grid), dtype=bool)
        if self.collar_seconds <= 0:
            return collar_mask

        for _, s, e in ref_segments:
            # Mask around start boundary [s - collar, s + collar]
            collar_mask |= (time_grid >= s - self.collar_seconds) & (time_grid <= s + self.collar_seconds)
            # Mask around end boundary [e - collar, e + collar]
            collar_mask |= (time_grid >= e - self.collar_seconds) & (time_grid <= e + self.collar_seconds)

        return collar_mask

    def _align_speakers(
        self,
        ref_segments: List[Tuple[str, float, float]],
        hyp_segments: List[Tuple[str, float, float]],
    ) -> Dict[str, str]:
        """
        Computes the optimal 1-to-1 mapping from hypothesis speaker labels
        to reference speaker labels that maximizes co-occurrence duration.
        """
        ref_speakers = sorted(list(set(s[0] for s in ref_segments)))
        hyp_speakers = sorted(list(set(s[0] for s in hyp_segments)))

        if not ref_speakers or not hyp_speakers:
            return {}

        ref_map = {spk: idx for idx, spk in enumerate(ref_speakers)}
        hyp_map = {spk: idx for idx, spk in enumerate(hyp_speakers)}

        overlap_matrix = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=float)

        for ref_spk, r_s, r_e in ref_segments:
            for hyp_spk, h_s, h_e in hyp_segments:
                overlap = max(0.0, min(r_e, h_e) - max(r_s, h_s))
                if overlap > 0:
                    overlap_matrix[ref_map[ref_spk], hyp_map[hyp_spk]] += overlap

        # Minimize negative overlap to maximize co-occurrence
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

        hyp_to_ref = {}
        for r_idx, h_idx in zip(row_ind, col_ind):
            hyp_to_ref[hyp_speakers[h_idx]] = ref_speakers[r_idx]

        # For any unassigned hypothesis speaker, assign default tag
        for h_spk in hyp_speakers:
            if h_spk not in hyp_to_ref:
                hyp_to_ref[h_spk] = f"UNMAPPED_{h_spk}"

        return hyp_to_ref

    def compute_der(
        self,
        reference: List[Tuple[str, float, float]],
        hypothesis: List[Tuple[str, float, float]],
    ) -> Dict[str, float]:
        """
        Calculates Diarization Error Rate (DER):
        DER = (T_miss + T_fa + T_conf) / T_ref

        Args:
            reference: List of (speaker_id, start_time, end_time) tuples.
            hypothesis: List of (speaker_id, start_time, end_time) tuples.

        Returns:
            Dictionary with der, der_percent, missed_speech_seconds,
            false_alarm_seconds, confusion_seconds, and total_ref_seconds.
        """
        if not reference:
            return {
                "der": 1.0 if hypothesis else 0.0,
                "der_percent": 100.0 if hypothesis else 0.0,
                "missed_seconds": 0.0,
                "false_alarm_seconds": sum(e - s for _, s, e in hypothesis),
                "confusion_seconds": 0.0,
                "total_ref_seconds": 0.0,
            }

        time_grid = self._get_time_grid(reference, hypothesis)
        if len(time_grid) == 0:
            return {"der": 0.0, "der_percent": 0.0}

        collar_mask = self._build_collar_mask(reference, time_grid)
        hyp_to_ref = self._align_speakers(reference, hypothesis)

        # Build frame-level presence sets
        ref_frames: List[Set[str]] = [set() for _ in range(len(time_grid))]
        hyp_frames: List[Set[str]] = [set() for _ in range(len(time_grid))]

        for spk, s, e in reference:
            indices = np.where((time_grid >= s) & (time_grid < e))[0]
            for idx in indices:
                ref_frames[idx].add(spk)

        for spk, s, e in hypothesis:
            mapped_spk = hyp_to_ref.get(spk, spk)
            indices = np.where((time_grid >= s) & (time_grid < e))[0]
            for idx in indices:
                hyp_frames[idx].add(mapped_spk)

        missed_frames = 0
        fa_frames = 0
        conf_frames = 0
        total_ref_frames = 0

        for i in range(len(time_grid)):
            # Skip collar regions around turn boundaries
            if collar_mask[i]:
                continue

            r_set = ref_frames[i]
            h_set = hyp_frames[i]
            n_ref = len(r_set)
            n_hyp = len(h_set)

            total_ref_frames += n_ref

            if n_ref == 0 and n_hyp > 0:
                fa_frames += n_hyp
            elif n_ref > 0 and n_hyp == 0:
                missed_frames += n_ref
            elif n_ref > 0 and n_hyp > 0:
                correct = len(r_set.intersection(h_set))
                conf_frames += min(n_ref, n_hyp) - correct
                if n_ref > n_hyp:
                    missed_frames += n_ref - n_hyp
                elif n_hyp > n_ref:
                    fa_frames += n_hyp - n_ref

        dt = self.time_step
        total_ref_sec = total_ref_frames * dt
        missed_sec = missed_frames * dt
        fa_sec = fa_frames * dt
        conf_sec = conf_frames * dt

        if total_ref_sec == 0:
            der = 0.0 if (missed_sec + fa_sec + conf_sec) == 0 else 1.0
        else:
            der = (missed_sec + fa_sec + conf_sec) / total_ref_sec

        der = min(der, 1.0)
        return {
            "der": round(der, 4),
            "der_percent": round(der * 100.0, 2),
            "missed_seconds": round(missed_sec, 2),
            "false_alarm_seconds": round(fa_sec, 2),
            "confusion_seconds": round(conf_sec, 2),
            "total_ref_seconds": round(total_ref_sec, 2),
            "collar_seconds": self.collar_seconds,
            "speaker_alignment": hyp_to_ref,
        }

    def compute_jer(
        self,
        reference: List[Tuple[str, float, float]],
        hypothesis: List[Tuple[str, float, float]],
    ) -> Dict[str, float]:
        """
        Calculates Jaccard Error Rate (JER):
        JER = (1 / |S|) * sum_{s in S} (1 - overlap(s, hyp_s) / union(s, hyp_s))
        """
        ref_speakers = sorted(list(set(s[0] for s in reference)))
        if not ref_speakers:
            return {"jer": 0.0, "jer_percent": 0.0}

        hyp_to_ref = self._align_speakers(reference, hypothesis)
        # Invert to map ref -> hyp
        ref_to_hyp = {r: h for h, r in hyp_to_ref.items()}

        jer_list = []
        speaker_jers = {}

        for ref_spk in ref_speakers:
            hyp_spk = ref_to_hyp.get(ref_spk)

            # Ref intervals for this speaker
            r_spans = [(s, e) for spk, s, e in reference if spk == ref_spk]
            # Hyp intervals mapped to this speaker
            h_spans = [(s, e) for spk, s, e in hypothesis if spk == hyp_spk] if hyp_spk else []

            # Compute overlap and union
            overlap = 0.0
            for r_s, r_e in r_spans:
                for h_s, h_e in h_spans:
                    ov = max(0.0, min(r_e, h_e) - max(r_s, h_s))
                    overlap += ov

            total_r = sum(e - s for s, e in r_spans)
            total_h = sum(e - s for s, e in h_spans)
            union = total_r + total_h - overlap

            if union == 0:
                spk_jer = 0.0
            else:
                spk_jer = 1.0 - (overlap / union)

            spk_jer = max(0.0, min(1.0, spk_jer))
            jer_list.append(spk_jer)
            speaker_jers[ref_spk] = round(spk_jer * 100.0, 2)

        mean_jer = round(float(np.mean(jer_list)), 4)
        return {
            "jer": mean_jer,
            "jer_percent": round(mean_jer * 100.0, 2),
            "per_speaker_jer": speaker_jers,
        }

    def evaluate_timeline(
        self,
        reference_timeline: List[Tuple[str, float, float]],
        hypothesis_timeline: Union[DiarizationTimeline, List[DiarizationSegment], List[Tuple[str, float, float]]],
    ) -> Dict[str, Any]:
        """
        Unified helper to evaluate both DER and JER from timeline objects.
        """
        hyp_tuples: List[Tuple[str, float, float]] = []
        if isinstance(hypothesis_timeline, DiarizationTimeline):
            hyp_tuples = [(seg.speaker_id, seg.start_time, seg.end_time) for seg in hypothesis_timeline.segments]
        elif isinstance(hypothesis_timeline, list) and hypothesis_timeline and isinstance(hypothesis_timeline[0], DiarizationSegment):
            hyp_tuples = [(seg.speaker_id, seg.start_time, seg.end_time) for seg in hypothesis_timeline] # type: ignore
        else:
            hyp_tuples = hypothesis_timeline # type: ignore

        der_res = self.compute_der(reference_timeline, hyp_tuples)
        jer_res = self.compute_jer(reference_timeline, hyp_tuples)

        return {
            "der": der_res["der"],
            "der_percent": der_res["der_percent"],
            "jer": jer_res["jer"],
            "jer_percent": jer_res["jer_percent"],
            "missed_seconds": der_res["missed_seconds"],
            "false_alarm_seconds": der_res["false_alarm_seconds"],
            "confusion_seconds": der_res["confusion_seconds"],
            "total_ref_seconds": der_res["total_ref_seconds"],
            "collar_seconds": self.collar_seconds,
            "speaker_alignment": der_res.get("speaker_alignment", {}),
            "per_speaker_jer": jer_res.get("per_speaker_jer", {}),
        }
