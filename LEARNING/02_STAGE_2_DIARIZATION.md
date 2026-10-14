# Stage 2: Acoustic Speaker Diarization — Deep Dive

---

## 1. The Problem We Are Solving

### 1.1 Real-World Scenario: "Who Spoke When?"
In any conversation, business meeting, medical consultation, or panel discussion, multiple people speak over time. Even if audio is clean or has been separated into stems, we do not know:
1. **Who is speaking** at any given millisecond (speaker identity label).
2. **When each turn begins and ends** (temporal boundaries / timestamps).
3. **How many distinct speakers** participated in the conversation ($K^*$).

Without speaker diarization, an ASR transcript is a monotonous wall of text with no speaker attribution. A medical consultation transcript without speaker labels cannot distinguish whether the doctor said *"Take 50mg daily"* or the patient asked *"Should I take 50mg daily?"*.

### 1.2 Why Raw Mixtures Break Traditional Diarization
Standard diarization algorithms assume that in any short time frame (e.g., 200ms–1.5s), only **one speaker** is talking. When speakers talk simultaneously (cross-talk / overlap):
- The acoustic feature vector extracted from the mixture contains blended formants from two different vocal tracts.
- The clustering algorithm cannot place this vector into Cluster A or Cluster B. It either creates an erroneous phantom third speaker cluster or randomly assigns the frame, causing catastrophic **Speaker Confusion Errors**.
- Overlap of just 25%–50% causes raw mixture Diarization Error Rate (DER) to jump above **40%–60%**.

### 1.3 Formal Problem Statement

> **Given**: An audio signal $x(t)$ of duration $T$, containing speech from an unknown number of speakers $K \in \{1, \dots, K_{\max}\}$.
>
> **Goal**: Segment $x(t)$ into $N$ contiguous non-silent turns:
> $$\mathcal{T} = \{(t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}, \text{speaker\_id}^{(i)})\}_{i=1}^N$$
> such that each time slice $t \in [t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}]$ is correctly attributed to the true speaker identity with minimum Diarization Error Rate (DER).

---

## 2. How We Are Solving It — Two Diarization Paradigms

Our architecture supports both **Raw Audio Diarization** and **Separation-Aware Diarization (Pipeline Integration)**.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: SPEAKER DIARIZATION                            │
│                                                                               │
│  Input Audio / Stems                                                          │
│        │                                                                      │
│        ▼                                                                      │
│  ┌───────────────────────────────┐                                            │
│  │  Voice Activity Detector     │  Removes silence (< -42 dB threshold)       │
│  │  (VAD) + Sliding Windows     │  Window = 1.5s, Step = 0.5s                 │
│  └──────────────┬────────────────┘                                            │
│                 │ Valid speech frames                                         │
│                 ▼                                                             │
│  ┌───────────────────────────────┐                                            │
│  │  Speaker Embedding Extractor  │  Log-Mel Filterbank (64 mels, 10ms hop)     │
│  │  (Unit-Normalized 192-dim)    │  → Orthonormal Projection → L2 Normalization│
│  └──────────────┬────────────────┘                                            │
│                 │ {e_i} ∈ ℝ^192, ||e_i|| = 1                                  │
│                 ▼                                                             │
│  ┌───────────────────────────────┐                                            │
│  │  Cosine Affinity Matrix       │  A_ij = max(0, cos_sim(e_i, e_j))           │
│  └──────────────┬────────────────┘                                            │
│                 │ Affinity Graph A                                            │
│                 ▼                                                             │
│  ┌───────────────────────────────┐                                            │
│  │  Eigengap Heuristic on        │  L_sym = I - D^(-1/2) A D^(-1/2)           │
│  │  Normalized Laplacian         │  K* = argmax_k (λ_{k+1} - λ_k)              │
│  └──────────────┬────────────────┘                                            │
│                 │ Optimal speaker count K*                                    │
│                 ▼                                                             │
│  ┌───────────────────────────────┐                                            │
│  │  Spectral Clustering / GMM    │  Eigenvectors U_K* → K-Means               │
│  └──────────────┬────────────────┘                                            │
│                 │ Frame labels {y_i}                                          │
│                 ▼                                                             │
│  ┌───────────────────────────────┐                                            │
│  │  Temporal Boundary Collation  │  Merge adjacent frames, suppress collars,  │
│  │  & Segment Smoothing          │  filter short bursts (< 0.25s)              │
│  └──────────────┬────────────────┘                                            │
│                 │                                                             │
│                 ▼                                                             │
│  Output: DiarizationTimeline [ {start, end, speaker_id}, ... ]                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Why Separation-Aware Diarization Is Superior
When combined with Stage 1 (Conv-TasNet):
1. **Raw Mixture Mode**: The diarizer processes the mixed waveform directly. Where overlap occurs, embeddings become corrupted, yielding a high DER of **40.52%**.
2. **Separation-Aware Mode**: The diarizer takes Stem 1 and Stem 2 produced by Conv-TasNet. Because the voices are physically unmixed into separate audio tracks:
   - Voice Activity Detection (VAD) is executed on each stem independently with a **leakage suppression margin** (7.0 dB).
   - Speaker A's stem has clean vocal characteristics with zero interference from Speaker B.
   - When both speakers talk at once, both stems register active speech independently with distinct speaker IDs.
   - **DER drops from 40.52% to 23.97%** — a massive **16.55% absolute reduction in error!**

---

## 3. Tech Stack & Software Architecture

| Technology | What It Is | What It Does In Stage 2 | Why We Use It |
|---|---|---|---|
| **Python 3.10+** | Core programming language | Implements the pipeline, matrix algebra, and evaluation suite | Ubiquitous in speech science, rich ecosystem for numerical arrays |
| **PyTorch (`torch`)** | Deep learning framework | Runs GPU/CPU tensor operations for acoustic feature transformations | Fast batched linear algebra and hardware auto-selection (`cuda` / `cpu`) |
| **NumPy (`numpy`)** | Numerical computing library | Vectorized operations, cosine affinity matrices, and collar masks | Highly optimized C-level BLAS/LAPACK execution for distance calculations |
| **SciPy (`scipy.linalg`, `scipy.signal`)** | Scientific computing library | Computes exact symmetric eigendecompositions (`eigh`) and Hungarian assignment (`linear_sum_assignment`) | Robust, deterministic linear algebra and graph theory solvers |
| **Scikit-Learn (`sklearn`)** | Machine learning library | Provides `KMeans` initialization and `GaussianMixture` (GMM) with covariance modeling | Production-tested clustering algorithms with configurable random seeds for reproducibility |
| **SoundFile (`soundfile`) / Librosa** | Audio I/O and DSP library | Reads WAV files and calculates accurate sample counts and sample rates | Zero-loss audio reading, consistent 16 kHz resampled floating point arrays |

---

## 4. Mathematical Foundations

### 4.1 Voice Activity Detection (VAD) & Energy Thresholding
Before attempting to cluster speaker identities, we must discard silent or background-noise frames.

For an acoustic window $w(t)$ of length $N = f_s \cdot T_{\text{win}}$ samples (where $f_s = 16000\text{ Hz}$, $T_{\text{win}} = 1.5\text{ s}$, $N = 24000$ samples):
The Root-Mean-Square (RMS) energy is:
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^N w[n]^2}$$

Converted to decibels full-scale (dBFS):
$$E_{\text{dB}} = 20 \log_{10}(\text{RMS} + \epsilon), \quad \epsilon = 10^{-9}$$

A frame is classified as **speech** if and only if:
$$E_{\text{dB}} \ge \theta_{\text{VAD}} \quad (\text{default: } -42.0\text{ dBFS})$$

Frames failing this condition are excluded from embedding extraction, preventing noise from creating spurious speaker clusters.

---

### 4.2 Log-Mel Filterbanks & Speaker Embeddings
Human voice timbre (pitch, formant frequencies $F_1, F_2, F_3$, vocal tract shape) is captured across the Mel frequency scale.

The Mel scale approximates the human ear's non-linear frequency resolution:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

1. We compute Short-Time Fourier Transform (STFT) with $N_{\text{fft}} = 512$ ($32\text{ ms}$) and hop size $H = 160$ ($10\text{ ms}$).
2. Pass the power spectrum through a 64-band triangular Mel filterbank matrix spanning $80\text{ Hz} \le f \le 7600\text{ Hz}$.
3. Take the log energy: $S_{\text{mel}} = \log(M \cdot |X|^2 + 10^{-6})$.
4. The temporal mean and standard deviation of $S_{\text{mel}}$ across the $1.5\text{ s}$ window form a feature vector $\mathbf{v} \in \mathbb{R}^{128}$.
5. Project $\mathbf{v}$ through an orthonormal projection matrix $W \in \mathbb{R}^{128 \times 192}$ and apply $L_2$ normalization:
$$\mathbf{e} = \frac{W^T \mathbf{v}}{\|W^T \mathbf{v}\|_2} \in \mathbb{R}^{192}, \quad \|\mathbf{e}\|_2 = 1$$

Now, every $1.5\text{ s}$ window of speech is represented as a point on the unit hypersphere $\mathbb{S}^{191}$.

---

### 4.3 Cosine Affinity Matrix
Because all embeddings lie on the unit hypersphere, the cosine similarity between two frame embeddings $\mathbf{e}_i$ and $\mathbf{e}_j$ is simply their inner product:
$$\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\|_2 \|\mathbf{e}_j\|_2} = \mathbf{e}_i^T \mathbf{e}_j$$

We construct the **Affinity Matrix** $A \in \mathbb{R}^{M \times M}$ across all $M$ active speech frames with non-negative thresholding:
$$A_{ij} = \begin{cases} 
1.0 & \text{if } i = j \\
\max(0.0, \mathbf{e}_i^T \mathbf{e}_j) & \text{if } i \ne j 
\end{cases}$$

- If frame $i$ and frame $j$ are from the same speaker, their embeddings align closely: $\mathbf{e}_i^T \mathbf{e}_j \approx 0.7 - 0.95$.
- If they are from different speakers, $\mathbf{e}_i^T \mathbf{e}_j \le 0.2$, which is suppressed toward zero.

---

### 4.4 Normalized Graph Laplacian & The Eigengap Heuristic
How do we know how many speakers spoke ($K$) without human intervention? We use **Spectral Graph Theory**.

Define the Degree Matrix $D \in \mathbb{R}^{M \times M}$ as a diagonal matrix where:
$$D_{ii} = \sum_{j=1}^M A_{ij}$$

The **Normalized Symmetric Graph Laplacian** is defined as:
$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$$

Properties of $L_{\text{sym}}$:
1. $L_{\text{sym}}$ is symmetric positive semi-definite.
2. Its eigenvalues satisfy $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_M \le 2$.
3. The multiplicity of eigenvalue $0$ equals the number of connected components in the affinity graph.

In real acoustic data with noise, the eigenvalues do not equal zero, but there is an abrupt "jump" (eigengap) after the true number of clusters $K^*$.

The **Eigengap Heuristic** chooses:
$$K^* = \operatorname{argmax}_{k \in [K_{\min}, K_{\max}]} (\lambda_{k+1} - \lambda_k)$$

```
Eigenvalue λ_k
  ▲
  │                                     ●  λ₄ = 0.85
  │                              ●  λ₃ = 0.81
  │                      ┌──────────────────────
  │                      │  EIGENGAP Δ = 0.62!
  │                      └──────────────────────
  │               ●  λ₂ = 0.19
  │        ●  λ₁ = 0.00
  └────────┼──────┼──────┼──────┼──────────────► k (Cluster index)
           k=1    k=2    k=3    k=4
           
  Conclusion: K* = 2 Speakers!
```

---

### 4.5 Spectral Clustering (Ng-Jordan-Weiss Algorithm)
Once $K^*$ is determined:
1. Compute the first $K^*$ eigenvectors $\mathbf{u}_1, \dots, \mathbf{u}_{K^*}$ corresponding to the smallest eigenvalues of $L_{\text{sym}}$.
2. Form matrix $U = [\mathbf{u}_1, \dots, \mathbf{u}_{K^*}] \in \mathbb{R}^{M \times K^*}$.
3. Renormalize each row of $U$ to unit length:
$$Y_{ij} = \frac{U_{ij}}{\sqrt{\sum_{k=1}^{K^*} U_{ik}^2}}$$
4. Run standard $K$-Means clustering on the rows of $Y$.
5. Assign original frame $i$ to cluster $k$ if row $i$ of $Y$ was assigned to cluster $k$.

*Why this works*: In the original 192-dimensional space, speaker clusters may be non-linearly separated or curved manifolds. The graph Laplacian maps these manifolds into an orthogonal coordinate system where clusters become tightly grouped, spherical points easily separable by $K$-Means.

---

### 4.6 Diarization Evaluation Metrics: DER and JER

#### 1. NIST Diarization Error Rate (DER)
The official metric defined by the National Institute of Standards and Technology (NIST) Rich Transcription evaluations:

$$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{total\_ref}}}$$

Where:
- **$T_{\text{miss}}$ (Missed Speech)**: Time duration where the ground truth has speech, but the hypothesis predicted silence.
- **$T_{\text{fa}}$ (False Alarm)**: Time duration where the hypothesis predicted speech, but the ground truth is silence.
- **$T_{\text{conf}}$ (Speaker Confusion)**: Time duration where speech is correctly detected, but assigned to the wrong speaker.
- **$T_{\text{total\_ref}}$**: Total duration of speech across all speakers in the ground truth.

#### 2. Collar Tolerance ($\tau = 250\text{ ms}$)
Human annotators cannot mark the start and end of a spoken word with millisecond precision. Natural vocal attacks and trailing breath sounds introduce ambiguity.

To avoid penalizing algorithms for human annotation fuzziness, NIST defines a **collar** of $\pm 250\text{ ms}$ around every reference turn boundary:
- If a turn begins at $t = 3.0\text{ s}$, the window $[2.75\text{ s}, 3.25\text{ s}]$ is masked out.
- Any discrepancy occurring inside this collar is ignored in the DER calculation.

```
Reference:   ──────────[SPEAKER_00 SPEECH]───────────
Boundary:             ▲                             ▲
                      3.0s                          5.0s
                      
Collar (±250ms):   ░░░░░                           ░░░░░
                 [2.75, 3.25]                    [4.75, 5.25]
                 (Errors Ignored)                (Errors Ignored)
```

#### 3. Optimal Speaker Matching via the Hungarian Algorithm
Hypothesis speaker labels (`SPEAKER_00`, `SPEAKER_01`) are arbitrary permutations of reference speaker names (`spk_A`, `spk_B`).

We construct a cost matrix $C \in \mathbb{R}^{|R| \times |H|}$ where $C_{ij}$ is the negative duration of temporal overlap between reference speaker $r_i$ and hypothesis speaker $h_j$:
$$C_{ij} = - \int_{0}^T \mathbb{I}(r(t) = r_i) \cdot \mathbb{I}(h(t) = h_j) \, dt$$

The **Hungarian Algorithm** (Kuhn-Munkres, implemented via `scipy.optimize.linear_sum_assignment`) finds the bijective mapping $\pi: R \to H$ that minimizes total cost in $\mathcal{O}(K^3)$ time:
$$\pi^* = \operatorname{argmin}_{\pi} \sum_{i} C_{i, \pi(i)}$$

#### 4. Jaccard Error Rate (JER)
While DER weights errors by total speech duration (giving long speakers more weight), JER calculates error per speaker equally:
$$\text{JER}_s = 1 - \frac{|R_s \cap H_{\pi(s)}|}{|R_s \cup H_{\pi(s)}|}$$
$$\text{JER} = \frac{1}{|S|} \sum_{s \in S} \text{JER}_s$$

JER ensures that short, infrequent turns by secondary speakers are not masked by a dominant speaker who talks 90% of the time.

---

## 5. Experimental Results & In-Depth Interpretation

Our diarization engine was benchmarked across 9 controlled multi-speaker configurations spanning 25% to 75% overlap, varying SIR ratios, and background noise.

### 5.1 Benchmark Results Table

| Mixture ID | Overlap | Raw Spectral DER | Raw GMM DER | Separation Pipeline DER | Separation Pipeline JER | Absolute DER Gain |
|---|---|---|---|---|---|---|
| `mix_01` (Pair AD) | 25% | 42.53% | 42.53% | **9.20%** | 10.47% | **-33.33%** |
| `mix_02` (Pair AD) | 50% | 44.25% | 66.67% | **9.20%** | 11.70% | **-35.05%** |
| `mix_03` (Pair AD, SIR +6) | 50% | 44.25% | 66.67% | **9.20%** | 11.70% | **-35.05%** |
| `mix_04` (Pair BC) | 25% | 30.00% | 58.75% | **37.50%** | 27.53% | -7.50% |
| `mix_05` (Pair BC) | 50% | 30.00% | 56.25% | **30.00%** | 26.61% | 0.00% |
| `mix_06` (Pair BC, SIR -6) | 50% | 30.00% | 68.75% | **16.25%** | 18.80% | **-13.75%** |
| `mix_07` (Pair AE) | 75% | 35.91% | 49.55% | **8.18%** | 11.59% | **-27.73%** |
| `mix_08` (Trio ADF Panel) | 50% | 62.07% | 45.26% | **49.57%** | 60.03% | **-12.50%** |
| `mix_09` (Pair BG, Noise) | 50% | 45.63% | 45.15% | **46.60%** | 31.31% | +0.97% |
| **Overall Average** | — | **40.52%** | **55.51%** | **23.97%** | **23.30%** | **-16.55%** |

### 5.2 Scientific Interpretation of Results

1. **The Separation Dividend (-16.55% Average DER Reduction)**:
   - On raw audio, Spectral Clustering produces an average DER of **40.52%**, and GMM is even worse at **55.51%**.
   - With Stage 1 Conv-TasNet running beforehand, average DER plummets to **23.97%**.
   - On high-overlap two-speaker mixes (`mix_01`, `mix_02`, `mix_03`, `mix_07`), the DER drops to **8% – 9%**, achieving production-grade accuracy even under **75% speech overlap** (`mix_07`).

2. **Why Raw GMM Fails (55.51% DER)**:
   - Gaussian Mixture Models fit elliptical Gaussian distributions in the 192-dimensional space.
   - When speakers overlap, cross-talk frames occupy intermediate spaces between speaker clusters. GMM inflates its covariance matrices to capture these outliers, merging the two speakers into a single wide cluster or splitting them arbitrarily.

3. **Spectral vs. GMM**:
   - Spectral clustering operates on affinity graph connectivity rather than parametric distributions. By zeroing out negative similarities and using graph Laplacian eigenvectors, it separates manifolds cleanly. Spectral clustering outperforms GMM on almost every mixture.

4. **Speed & Efficiency**:
   - The entire diarization pipeline runs with an average **Real-Time Factor (RTF) of 0.006**.
   - This means **10 seconds of audio is diarized in just 60 milliseconds** on CPU — over **166× faster than real-time**.

---

## 6. How Stage 3 (ASR) Builds on Diarization

Stage 2 answers **"WHO spoke WHEN"**, producing a list of timestamped speaker turns:
```json
[
  {"start": 0.00, "end": 2.45, "speaker_id": "SPEAKER_00"},
  {"start": 2.30, "end": 4.80, "speaker_id": "SPEAKER_01"}
]
```

However, Stage 2 has zero semantic understanding. It does not know what words were uttered.

**In Stage 3 (Automatic Speech Recognition)**:
1. We take each segment's $[t_{\text{start}}, t_{\text{end}}]$ and slice the audio corresponding to that exact interval.
2. If Stage 1 separation was performed, we slice the audio from the **clean stem corresponding to that speaker**.
3. We feed this isolated speaker slice into our ASR models (OpenAI Whisper & AI4Bharat Vakyansh).
4. The result is an attributed transcript:
   - `[0.00s - 2.45s] SPEAKER_00: "उन्हें दस दिन तक rehab करना होगा"`
   - `[2.30s - 4.80s] SPEAKER_01: "यह एक वॉइस ऑडियो डेटासेट है"`

If we bypassed Stage 2 and fed raw overlapping audio directly into ASR, the ASR model would encounter colliding phonemes and collapse into hallucinations or repetition loops.

---

## 7. Key Viva / Defense Talking Points

1. **Why not just use an energy-based VAD for diarization?**
   > *Energy-based VAD can only detect speech vs. non-speech. It cannot differentiate between Speaker A and Speaker B. Diarization requires acoustic feature embeddings (timbre/vocal tract geometry) and clustering in high-dimensional space.*

2. **Why use Spectral Clustering instead of simple K-Means?**
   > *Standard K-Means assumes spherical clusters in Euclidean space. Speaker embeddings often lie on curved sub-manifolds of the hypersphere. Spectral clustering uses the Graph Laplacian to unroll these manifolds into linear subspaces, making clustering trivial.*

3. **How does your system automatically determine the number of speakers?**
   > *We use the Eigengap Heuristic on the normalized Laplacian $L_{\text{sym}}$. The difference between consecutive eigenvalues $\lambda_{k+1} - \lambda_k$ exhibits a sharp maximum at the true number of clusters $K^*$.*

4. **Why is the 250ms collar used in NIST DER?**
   > *Human annotators disagree on word onset/offset boundaries by 100–300ms due to breathing and low-energy fricatives. The 250ms collar forgives edge discrepancies, focusing evaluation on true speaker confusion and missed speech.*

5. **What is the most impactful engineering decision in Stage 2?**
   > *Integrating upstream Blind Source Separation (Stage 1). Upstream separation reduced Diarization Error Rate by 16.55% absolute, proving that resolving the acoustic cocktail party problem is foundational to multi-speaker conversation tracking.*
