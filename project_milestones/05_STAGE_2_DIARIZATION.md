# 05 — Stage 2: Acoustic Speaker Diarization

## 1\. Core Objective & Conceptual Formulation

The fundamental acoustic task of Stage 2 is to answer:

> *“Who spoke when throughout the recording, without prior enrolled speaker profiles or identity templates?”*

While Stage 1 isolates overlapping acoustic waveforms into independent tracks, Stage 2 solves the temporal clustering and speaker tracking problem. Together with Automatic Speech Recognition (Stage 3), they form the full conversational matrix:

*   **Diarization**: *“Who spoke when?”* (Temporal boundaries $[t_{\text{start}}, t_{\text{end}}]$ $\rightarrow$ Speaker Cluster $C_k$)
*   **ASR**: *“What did they say?”* (Acoustic segment $\rightarrow$ Text tokens $W$)
*   **Unified Output**: *“Who said what and when?”*

 *

## 2\. Diarization Pipeline Architecture

```mermaid
flowchart LR
    A[Acoustic Audio Stream] --> B[Voice Activity Detection VAD]
    B -->|Speech Segments| C[Sliding Window Segmentation 1.5s - 3.0s]
    C --> D[Deep Embedding Extractor pyannote.audio SincNet / TDNN]
    D -->|d-dimensional Embeddings| E[Affinity Matrix Computation Cosine Similarity]
    E --> F[Unsupervised Clustering Spectral vs. GMM]
    F -->|Cluster IDs| G[Temporal Boundary Smoothing & Merging]
    G --> H[Chronological Timeline JSON]
```

### Key Stages:

1. Voice Activity Detection (VAD): Prunes non-speech silence, microphone rumble, and acoustic voids to prevent noise vectors from poisoning speaker embedding spaces.
2. Feature Extraction / Embedding Generation: Uses pyannote.audio state-of-the-art neural feature extractors (e.g., ECAPA-TDNN or SincNet-based backbones) trained on metric learning objectives (Additive Angular Margin Loss / ArcFace). Converts variable-length speech chunks into fixed-length unit-norm representations e∈Rd (typically d=192 or 512).
3. Unsupervised Clustering:Method A: Spectral Clustering: Constructs an affinity graph using cosine similarity:

$$
A_{ij} = \max\left(0, \frac{\mathbf{e}_i^\top \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}\right)
$$

```
 Computes the normalized graph Laplacian <span class="va-math-inline" data-latex="L = I - D^{-1/2} A D^{-1/2}" contenteditable="false"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="katex-base"><span class="katex-strut" style="height:0.6833em;"></span><span class="mord mathnormal">L</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="katex-base"><span class="katex-strut" style="height:0.7667em;vertical-align:-0.0833em;"></span><span class="mord mathnormal" style="margin-right:0.0785em;">I</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="katex-base"><span class="katex-strut" style="height:0.888em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right:0.0278em;">D</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.888em;"><span style="top:-3.063em;margin-right:0.05em;"><span class="pstrut" style="height:2.7em;"></span><span class="katex-sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">−</span><span class="mord mtight">1/2</span></span></span></span></span></span></span></span></span><span class="mord mathnormal">A</span><span class="mord"><span class="mord mathnormal" style="margin-right:0.0278em;">D</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.888em;"><span style="top:-3.063em;margin-right:0.05em;"><span class="pstrut" style="height:2.7em;"></span><span class="katex-sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">−</span><span class="mord mtight">1/2</span></span></span></span></span></span></span></span></span></span></span></span></span> and performs <span class="va-math-inline" data-latex="k" contenteditable="false"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="katex-base"><span class="katex-strut" style="height:0.6944em;"></span><span class="mord mathnormal" style="margin-right:0.0315em;">k</span></span></span></span></span>-means on the top eigenvectors.
```

*   **Method B: Gaussian Mixture Models (GMMs)**: Models the embedding space as a mixture of $K$ multivariate Gaussian distributions with expectation-maximization (EM) fitting.

4. Automated Speaker Count Estimation (K): In real-world radio panels, the exact number of speakers is not known a priori. We implement the Eigengap Heuristic on the sorted eigenvalues λ1​≤λ2​≤…λn​ of the graph Laplacian:

$$
K^* = \arg\max_{k} (\lambda_{k+1} - \lambda_k)
$$

 *

## 3\. Evaluation Metrics: Mathematical Rigor

### Diarization Error Rate (DER)

DER is the standard NIST metric for speaker diarization. It measures the percentage of time that speech is incorrectly attributed:

$$
\text{DER} = \frac{\text{Duration of Missed Speech} + \text{Duration of False Alarm Speech} + \text{Duration of Speaker Confusion}}{\text{Total Ground-Truth Speech Duration}}
$$

where:

*   **Missed Speech ($T_{\text{miss}}$)**: Ground-truth speaker is active, but system predicts silence.
*   **False Alarm ($T_{\text{fa}}$)**: Ground truth is silent, but system predicts speaker activity.
*   **Speaker Confusion ($T_{\text{conf}}$)**: Speech is detected, but attributed to Speaker $A$ instead of Speaker $B$.

A standard collar (typically $250\text{ ms}$) is applied around reference boundaries to accommodate human labeling tolerance.

### Jaccard Error Rate (JER)

JER provides an equal-weight evaluation across all speakers regardless of total speaking time (preventing dominant talkers from skewing the metric):

$$
\text{JER} = \frac{1}{|S|} \sum_{s \in S} \left( 1 - \frac{\text{overlap}(s, \hat{s})}{\text{union}(s, \hat{s})} \right)
$$

where $S$ is the set of speakers, and $\hat{s}$ is the optimal matching hypothesis speaker.

 *

## 4\. Key Experimental Study: Diarization With vs. Without Source Separation

A core contribution of this project is to quantitatively measure the impact of upstream source separation on diarization:

*   **Condition 1 (Baseline Diarization)**: Feed raw multi-speaker mixed audio directly to `pyannote.audio` diarizer. Measure DER on overlapping speech segments.
*   **Condition 2 (Pipeline Diarization)**: Pass audio through Stage 1 Demucs separator first, then diarize isolated tracks and reconcile timestamps.
*   **Hypothesis**: Stage 1 separation will significantly reduce Speaker Confusion ($T_{\text{conf}}$) and Missed Speech ($T_{\text{miss}}$) during overlapping intervals ($\Omega \ge 25\%$).

 *

## 5\. Implementation Architecture (`src/diarization/`)

```python
class Diarizer(ABC):
    @abstractmethod
    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> DiarizationTimeline:
        pass

class PyannoteSpectralDiarizer(Diarizer):
    def __init__(self, clustering_method: str = "spectral"):
        self.embedding_model = load_embedding_model()
        self.clustering_method = clustering_method
        
    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> DiarizationTimeline:
        # 1. Voice activity detection
        # 2. Extract sliding window embeddings
        # 3. If num_speakers is None, run Eigengap heuristic
        # 4. Perform Spectral Clustering or GMM fitting
        # 5. Temporal collation and collar smoothing
        # 6. Return DiarizationTimeline object
        pass
```

 *

## 6\. Definition of Done (DoD) for Stage 2

- [x] Embedding extraction from `pyannote.audio` executes reliably without out-of-memory faults.
- [x] Both Spectral Clustering and GMM clustering implementations are functional and selectable via config.
- [x] Automated speaker count estimation ($K$ estimation via eigengap heuristic) is verified on synthetic multi-speaker audio.
- [x] Output JSON timeline strictly conforms to Contract 3 (`diarization_timeline.json`).
- [x] DER and JER evaluation script compares hypotheses against ground-truth synthetic annotations.
- [x] Ablation study comparing Diarization on Raw Audio vs. Diarization on Separated Audio is completed with recorded DER metrics.
- [x] Latency, processing time, RTF, RAM, and GPU memory benchmarks are captured.