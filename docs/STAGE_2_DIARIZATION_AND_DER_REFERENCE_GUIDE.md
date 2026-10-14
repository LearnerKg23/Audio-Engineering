# Stage 2: Acoustic Speaker Diarization & DER Reference Guide
**Document**: Mathematical Formulations, Clustering Architecture, Cross-Stem Gating, and Benchmark Results  
**Milestone**: Milestone M5 (Stage 2 — Diarization)  
**Location**: `docs/STAGE_2_DIARIZATION_AND_DER_REFERENCE_GUIDE.md`  

---

## 1. Overview: The Diarization Challenge ("Who Spoke When?")

Speaker Diarization partitions an audio stream into homogeneous segments attributed to individual speaker identities without prior enrollment templates.

In conversational Indian audio (panel discussions, multi-speaker debates, and telephony calls), diarization faces **two catastrophic failure modes**:
1. **The Overlap Dilemma**: When two or more people speak concurrently, standard single-speaker clustering models assign the entire overlapping frame to only one speaker, causing severe **Missed Speech** and **Speaker Confusion**.
2. **Ghost Speaker Hallucination**: Acoustic cross-talk corrupts voice embeddings, causing clustering algorithms to falsely detect non-existent "ghost speakers."

Our system addresses this through **Separation-Aware Diarization**: passing audio through Stage 1 Conv-TasNet first, and diarizing the isolated stems with **Cross-Stem Relative Energy Gating**.

---

## 2. Mathematical Formulations & Metric Derivations

### A. NIST Diarization Error Rate (DER)

Diarization Error Rate (DER) is the international NIST RT-09 standard metric measuring the fraction of time attributed to the wrong speaker.

$$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{total\_ref}}}$$

Where:
* $T_{\text{miss}}$ (**Missed Speech**): Audio where reference speech is present, but the hypothesis predicted silence.
* $T_{\text{fa}}$ (**False Alarm Speech**): Audio where reference is silent, but the hypothesis falsely predicted speech.
* $T_{\text{conf}}$ (**Speaker Confusion**): Audio where speech is present, but assigned to the wrong speaker cluster ID.
* $T_{\text{total\_ref}}$: Total duration of ground-truth reference speech across all speakers.

#### The 250ms Collar Tolerance
Human turn-taking boundaries and phonetic stop closures exhibit natural transition variance. Following the NIST evaluation standard, we apply a **$\pm 250\text{ ms}$ collar** centered at every reference segment boundary. Any discrepancy within this collar is excluded from the error calculation to avoid penalizing micro-boundary alignment noise.

---

### B. Optimal Permutation Matching (The Hungarian Algorithm)

Diarization models assign anonymous cluster IDs (`SPEAKER_00`, `SPEAKER_01`). To evaluate against ground truth without false penalties from label reordering, we compute optimal bipartite matching using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`).

Given $R$ reference speakers and $H$ hypothesis clusters, we construct a cost matrix $C \in \mathbb{R}^{R \times H}$ where:

$$C_{i,j} = - \text{OverlapDuration}(\text{Ref}_i, \text{Hyp}_j)$$

The optimal mapping $\pi^*$ maximizes total co-occurring speech duration:

$$\pi^* = \arg\min_{\pi} \sum_{i=1}^R C_{i, \pi(i)}$$

---

### C. Jaccard Error Rate (JER)

Jaccard Error Rate evaluates speaker-level segmentation quality independently of speaker dominance, preventing a talkative speaker from drowning out short interjections:

$$\text{JER} = \frac{1}{K} \sum_{k=1}^K \left( 1 - \frac{|\text{Ref}_k \cap \text{Hyp}_{\pi(k)}|}{|\text{Ref}_k \cup \text{Hyp}_{\pi(k)}|} \right)$$

---

### D. Hyperspherical Acoustic Embeddings & Normalized Graph Laplacian

Each audio window ($T_w = 1.5\text{ s}$, step $T_s = 0.5\text{ s}$) is mapped to a unit-norm embedding vector $\mathbf{e} \in \mathbb{R}^{192}$ on a hypersphere ($\|\mathbf{e}\|_2 = 1$).

We construct the **Cosine Similarity Affinity Matrix** $A$:

$$A_{ij} = \max(0, \mathbf{e}_i^\top \mathbf{e}_j), \quad A_{ii} = 1.0$$

The **Normalized Symmetric Graph Laplacian** $L_{\text{sym}}$ is defined as:

$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$$

Where $D$ is the diagonal degree matrix $D_{ii} = \sum_j A_{ij}$.

---

### E. Automated Speaker Count Estimation ($K^*$) via Eigengap Heuristic

Rather than hardcoding the number of speakers, the system computes the eigenvalues $\lambda_1 \le \lambda_2 \le \dots \le \lambda_N$ of $L_{\text{sym}}$ and estimates $K^*$ at the maximum spectral gap:

$$K^* = \arg\max_{k \in [K_{\min}, K_{\max}]} (\lambda_{k+1} - \lambda_k)$$

The top $K^*$ eigenvectors are then projected onto the unit sphere and clustered via $K$-Means.

---

### F. Cross-Stem Relative Energy Gating

When diarizing separated audio stems from Stage 1, minor acoustic leakage (bleed) can trigger false speech detections during silent intervals.

To eliminate this, we compute chunk-level RMS energy across all $S$ stems simultaneously:

$$\text{RMS}_s = \sqrt{\frac{1}{N} \sum_{n=1}^N s[n]^2}, \quad \text{dB}_s = 20 \log_{10}(\text{RMS}_s)$$

A stem $s$ is marked as active speech **if and only if**:
1. $\text{dB}_s \ge \text{Threshold}_{\text{VAD}}$ (exceeds local noise floor).
2. $\max_j(\text{dB}_j) - \text{dB}_s \le \text{LeakageMargin}_{\text{dB}}$ (not acoustic bleed from a concurrently louder talker).

We calibrate $\text{LeakageMargin}_{\text{dB}} = 7.0\text{ dB}$, successfully suppressing phantom cross-talk.

---

## 3. Empirical Results: Raw vs. Separation-Aware Diarization

Evaluated across all 9 controlled mixtures in the benchmark suite:

| Mixture ID | Overlap ($\Omega$) | Raw Spectral DER | Raw GMM DER | Separation Pipeline DER | Pipeline JER |
|---|:---:|:---:|:---:|:---:|:---:|
| `controlled_mix_01` (Pair AD) | 25% | 42.53% | 42.53% | **9.20%** | 10.47% |
| `controlled_mix_02` (Pair AD) | 50% | 44.25% | 66.67% | **9.20%** | 11.70% |
| `controlled_mix_03` (Pair AD) | 50% | 44.25% | 66.67% | **9.20%** | 11.70% |
| `controlled_mix_04` (Pair BC) | 25% | 30.00% | 58.75% | **37.50%** | 27.53% |
| `controlled_mix_05` (Pair BC) | 50% | 30.00% | 56.25% | **30.00%** | 26.61% |
| `controlled_mix_06` (Pair BC) | 50% | 30.00% | 68.75% | **16.25%** | 18.80% |
| `controlled_mix_07` (Pair AE) | 75% | 35.91% | 49.55% | **8.18%** | 11.59% |
| `controlled_mix_08` (Trio ADF) | 50% | 62.07% | 45.26% | **49.57%** | 60.03% |
| `controlled_mix_09` (Pair BG) | 50% | 45.63% | 45.15% | **46.60%** | 31.31% |
| **Overall Dataset Average** | — | **40.52%** | **55.51%** | **23.97%** ⭐ | **23.30%** |

---

## 4. Key Scientific Insights

1. **Upstream Separation Rescues Diarization**:
   * Raw mixture diarization suffers a severe **40.52% mean DER** because overlapping frames confuse the clustering boundary.
   * Passing audio through Conv-TasNet first drops mean DER to **23.97%** (**-16.55% absolute DER improvement**).
   * On clean 2-speaker mixtures (`mix_01`, `mix_02`, `mix_03`), separation drops DER from **44.25% down to 9.20%**!
2. **Spectral Clustering vs. GMM**:
   * Spectral Clustering with the Eigengap heuristic achieves **40.52% DER** vs. GMM's **55.51% DER** (a **15% advantage**). GMM assumes elliptical Gaussian cluster distributions which break down in high-dimensional speech manifolds.
3. **Computational Efficiency**:
   * Average Diarization Real-Time Factor (RTF) is **0.006** (processing audio **166.7x faster than real-time**).

---

## 5. Viva / Defense Q&A Cheat Sheet

| Question | Winning Technical Answer |
|---|---|
| **Q: "Why does raw audio diarization fail on overlapping speech?"** | *"Standard diarization classifiers assign each temporal window to exactly one speaker. During overlap, the acoustic embeddings are a linear blend of both speakers, which creates either heavy confusion error or splits into hallucinated ghost clusters."* |
| **Q: "What is the purpose of the 250ms collar in DER?"** | *"The 250ms collar is the official NIST RT-09 standard. It accounts for human turn-taking transition variance and prevents micro-boundary alignment noise from distorting the overall metric."* |
| **Q: "How does the Eigengap heuristic find the speaker count?"** | *"It calculates the eigenvalues of the normalized Graph Laplacian ($L_{\text{sym}}$). The number of near-zero eigenvalues corresponds to the number of connected components in the affinity graph. The largest gap between consecutive eigenvalues indicates the optimal cluster count $K^*$."* |
| **Q: "What is Cross-Stem Energy Gating?"** | *"Because neural separation models can have minor acoustic bleed during silent intervals, cross-stem gating compares the RMS energy across all channels simultaneously and suppresses secondary channels that are more than 7 dB quieter than the dominant channel."* |
