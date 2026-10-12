"""
Acoustic Speaker Clustering Module for Stage 2: Speaker Diarization.

Implements:
1. Cosine similarity affinity graph computation.
2. Normalized Graph Laplacian construction.
3. Automated speaker count estimation (K*) via the Eigengap heuristic.
4. Spectral Clustering and Gaussian Mixture Model (GMM) clustering.
5. Temporal segment smoothing, boundary collation, and DiarizationTimeline formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.core.state import DiarizationSegment, DiarizationTimeline


class SpeakerClusterer:
    """
    Clusters frame-level speaker embeddings into discrete speaker timelines.
    """

    def __init__(
        self,
        clustering_method: str = "spectral",
        min_speakers: int = 1,
        max_speakers: int = 4,
        min_segment_duration: float = 0.25,
        collar_seconds: float = 0.25,
    ):
        self.clustering_method = clustering_method.lower()
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.min_segment_duration = min_segment_duration
        self.collar_seconds = collar_seconds

    def compute_affinity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity affinity matrix with non-negative thresholding:
        A_ij = max(0, cos_sim(e_i, e_j)).
        """
        # Ensure unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        norm_embs = embeddings / norms

        affinity = np.dot(norm_embs, norm_embs.T)
        # Zero out negative correlations and self-loops
        affinity = np.maximum(0.0, affinity)
        np.fill_diagonal(affinity, 1.0)
        return affinity

    def estimate_speaker_count_eigengap(
        self, affinity: np.ndarray, min_k: Optional[int] = None, max_k: Optional[int] = None
    ) -> int:
        """
        Estimates the number of speakers K* using the Eigengap Heuristic
        on the normalized symmetric Graph Laplacian:
        L_sym = I - D^(-1/2) A D^(-1/2)

        K* = argmax_{k} (lambda_{k+1} - lambda_k)
        """
        n_samples = affinity.shape[0]
        if n_samples <= 2:
            return 1

        min_k = min_k or self.min_speakers
        max_k = min(max_k or self.max_speakers, n_samples - 1)
        if min_k >= max_k:
            return min_k

        # Degree matrix
        degree = np.sum(affinity, axis=1)
        inv_sqrt_d = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        d_mat = np.diag(inv_sqrt_d)

        # Normalized symmetric Laplacian: L = I - D^(-1/2) A D^(-1/2)
        laplacian = np.eye(n_samples) - np.dot(d_mat, np.dot(affinity, d_mat))

        # Eigenvalues in ascending order
        eigenvalues = linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)

        # Eigengap search across valid cluster counts [min_k, max_k]
        best_k = min_k
        max_gap = -1.0

        for k in range(min_k, max_k + 1):
            if k < len(eigenvalues) - 1:
                gap = float(eigenvalues[k] - eigenvalues[k - 1]) if k > 0 else float(eigenvalues[1] - eigenvalues[0])
                if gap > max_gap:
                    max_gap = gap
                    best_k = k

        return max(1, best_k)

    def cluster_spectral(
        self,
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Performs Spectral Clustering on speaker embeddings.
        """
        n_samples = embeddings.shape[0]
        if n_samples == 1:
            return np.zeros(1, dtype=int), 1

        affinity = self.compute_affinity_matrix(embeddings)

        k = num_speakers or self.estimate_speaker_count_eigengap(affinity)
        k = max(1, min(k, n_samples))

        if k == 1:
            return np.zeros(n_samples, dtype=int), 1

        # Normalized Laplacian
        degree = np.sum(affinity, axis=1)
        inv_sqrt_d = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        d_mat = np.diag(inv_sqrt_d)
        laplacian = np.eye(n_samples) - np.dot(d_mat, np.dot(affinity, d_mat))

        # Eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eigh(laplacian)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        # Top k eigenvectors
        top_vecs = eigvecs[:, :k]
        # Normalize rows to unit length for spherical k-means
        row_norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
        top_vecs = top_vecs / np.maximum(row_norms, 1e-12)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(top_vecs)
        return labels, k

    def cluster_gmm(
        self,
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Performs Gaussian Mixture Model (GMM) clustering with BIC selection.
        """
        n_samples = embeddings.shape[0]
        if n_samples == 1:
            return np.zeros(1, dtype=int), 1

        if num_speakers is not None:
            k = max(1, min(num_speakers, n_samples))
        else:
            # Automatic K selection via Bayesian Information Criterion (BIC)
            best_bic = np.inf
            k = self.min_speakers
            max_cand = min(self.max_speakers, n_samples)
            for cand_k in range(self.min_speakers, max_cand + 1):
                try:
                    gmm_cand = GaussianMixture(
                        n_components=cand_k,
                        covariance_type="diag",
                        random_state=42,
                        max_iter=50,
                    )
                    gmm_cand.fit(embeddings)
                    bic = gmm_cand.bic(embeddings)
                    if bic < best_bic:
                        best_bic = bic
                        k = cand_k
                except Exception:
                    continue

        k = max(1, min(k, n_samples))
        if k == 1:
            return np.zeros(n_samples, dtype=int), 1

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            max_iter=100,
        )
        labels = gmm.fit_predict(embeddings)
        return labels, k

    def smooth_and_collate_timeline(
        self,
        timestamps: List[Tuple[float, float]],
        labels: np.ndarray,
        audio_file: str,
        total_speakers: int,
        algorithm_name: str,
    ) -> DiarizationTimeline:
        """
        Collates contiguous frame-level cluster predictions into continuous
        DiarizationSegment intervals, pruning sub-threshold transient spikes.
        """
        if len(timestamps) == 0:
            return DiarizationTimeline(
                audio_file=audio_file,
                algorithm=algorithm_name,
                total_speakers_detected=0,
                segments=[],
            )

        raw_segments = []
        curr_spk = labels[0]
        curr_start = timestamps[0][0]
        curr_end = timestamps[0][1]

        for i in range(1, len(timestamps)):
            spk = labels[i]
            t_start, t_end = timestamps[i]

            # If same speaker and overlapping/adjacent window, extend segment
            if spk == curr_spk and t_start <= curr_end + 0.1:
                curr_end = max(curr_end, t_end)
            else:
                raw_segments.append((curr_spk, curr_start, curr_end))
                curr_spk = spk
                curr_start = t_start
                curr_end = t_end

        raw_segments.append((curr_spk, curr_start, curr_end))

        # Filter out micro-spurious bursts (< min_segment_duration) if multiple segments exist
        filtered_segments = []
        for spk, start, end in raw_segments:
            duration = end - start
            if len(raw_segments) > 1 and duration < self.min_segment_duration:
                # Merge into previous segment if available
                if filtered_segments:
                    prev_spk, prev_start, prev_end = filtered_segments[-1]
                    filtered_segments[-1] = (prev_spk, prev_start, max(prev_end, end))
                continue
            filtered_segments.append((spk, start, end))

        if not filtered_segments and raw_segments:
            filtered_segments = raw_segments

        # Map cluster integers to standard IDs: SPEAKER_00, SPEAKER_01, etc.
        unique_labels = sorted(list(set(s[0] for s in filtered_segments)))
        spk_id_map = {lbl: f"SPEAKER_{idx:02d}" for idx, lbl in enumerate(unique_labels)}

        structured_segments = []
        for idx, (spk_lbl, s_time, e_time) in enumerate(filtered_segments):
            structured_segments.append(
                DiarizationSegment(
                    segment_id=idx,
                    speaker_id=spk_id_map[spk_lbl],
                    start_time=round(s_time, 2),
                    end_time=round(e_time, 2),
                    confidence=0.92,
                )
            )

        return DiarizationTimeline(
            audio_file=audio_file,
            algorithm=algorithm_name,
            total_speakers_detected=len(unique_labels),
            segments=structured_segments,
        )

    def cluster(
        self,
        timestamps: List[Tuple[float, float]],
        embeddings: np.ndarray,
        audio_file: str = "audio.wav",
        num_speakers: Optional[int] = None,
    ) -> DiarizationTimeline:
        """
        Runs clustering and returns standardized DiarizationTimeline.
        """
        if self.clustering_method == "gmm":
            labels, k = self.cluster_gmm(embeddings, num_speakers=num_speakers)
            algo = "gmm_clustering"
        else:
            labels, k = self.cluster_spectral(embeddings, num_speakers=num_speakers)
            algo = "spectral_clustering_eigengap"

        return self.smooth_and_collate_timeline(
            timestamps=timestamps,
            labels=labels,
            audio_file=audio_file,
            total_speakers=k,
            algorithm_name=algo,
        )
