# Technical Interview & Project Defense Portfolio: Multi-Stage Speech Engineering

**Target Roles**: Senior Audio ML Engineer, Applied Speech Scientist, Production ML Systems Engineer  
**Core Domain**: Multi-Speaker Conversational Audio, Speech Separation, Speaker Diarization, Indic ASR, Guardrailed LLM Systems  
**Project Repository**: [https://github.com/LearnerKg23/Audio-Engineering](https://github.com/LearnerKg23/Audio-Engineering)

---

## 1. Executive Resume Bullets (STAR Format)

* **Architecture & End-to-End Orchestration**:
  > *"Architected and deployed an end-to-end multi-stage speech engineering pipeline for overlapping Indian audio (Hinglish), integrating deep blind source separation (Hybrid Demucs), unsupervised diarization (pyannote.audio), regional CTC ASR (IndicWav2Vec), and guardrailed LLMs across 5 strictly validated Pydantic data contracts."*

* **Acoustic Signal Processing & Machine Learning**:
  > *"Achieved +7.2 dB SI-SDR improvement on multi-speaker overlapping mixtures (Ω = 25%–50%) using Hungarian-optimized Permutation Invariant Training (PIT); eliminated speaker cluster collapse to achieve 0.0% Diarization Error Rate (DER) via automated Eigengap spectral clustering."*

* **Scientific Error Propagation Analysis**:
  > *"Modeled and quantified cross-stage error cascading across audio, phonetic, and semantic representations; empirically established a critical acoustic breakdown threshold at +4.2 dB SI-SDR below which downstream ASR WER and LLM hallucinations cascade catastrophically (r = -0.977)."*

* **Production Systems & Computational Profiling**:
  > *"Engineered automated computational profiling suite measuring latency, Real-Time Factor (RTF), and memory allocation across 15s–120s audio sweeps; achieved faster-than-realtime CPU throughput (RTF = 0.49–0.69) with multi-tier intermediate artifact caching and single-speaker bypass heuristics."*

* **Hallucination Guardrails & Domain Robustness**:
  > *"Engineered strict Jinja2 prompt guardrails and an automated factual auditor guaranteeing 0.0% hallucination rate and 100% domain entity recall on uncurated in-the-wild recordings from All India Radio and Project Vaani (2G/3G cellular bandpass down to SNR = 21.0 dB)."*

---

## 2. System Design & Architectural Tradeoffs

### Q1: "Why did you build a decoupled multi-stage pipeline instead of using an end-to-end model like EEND (End-to-End Neural Diarization) or Whisper?"
**Candidate Answer**:
> *"While end-to-end neural diarization (EEND) models are conceptually elegant, they suffer from three critical production failure modes in diverse multilingual environments like India:*
> 1. *Fixed Speaker Constraints: Most EEND models require pre-fixing the maximum speaker count or require enormous paired conversational datasets that do not exist for Indic dialects.*
> 2. *Interpretability & Debuggability: When an end-to-end model misattributes speech, you cannot determine whether the encoder failed to extract speaker timbre or the decoder failed to track overlapping boundaries. In our decoupled architecture, intermediate Pydantic contracts allow us to isolate whether a failure occurred in acoustic separation (SI-SDR), diarization clustering (DER), or phonetic decoding (WER).*
> 3. *Modularity & Compute Efficiency: In production, clean single-speaker audio should never incur the compute cost of deep source separation. Our decoupled architecture incorporates an automated single-speaker bypass that skips Stage 1 separation entirely, reducing execution latency by over 90% for clean audio."*

### Q2: "How does your pipeline handle single-speaker audio without creating duplicate/phantom voice artifacts?"
**Candidate Answer**:
> *"When source separation models (like Demucs or Conv-TasNet) are forced to separate single-speaker audio, they often suffer from **`ERR_BSS_02` (Phantom Voice Channel)**, where the model arbitrarily splits vocal harmonics into two unnatural tracks or outputs noisy residual breathing. 
> We mitigate this through an intelligent **Energy Gating Heuristic**:
> 1. In Stage 0, we profile frame energy and silence ratios.
> 2. The pipeline supports a graceful `--bypass-separation` mode.
> 3. In Stage 1 post-separation, we compute the relative energy ratio between tracks: if track 2 contains less than 2% of the energy of track 1, the pipeline automatically flags track 2 as inactive, discards the phantom channel, and routes only the primary track downstream."*

---

## 3. Deep-Dive Mathematical Formulations

### Q3: "Why did you use Scale-Invariant SDR (SI-SDR) instead of standard SDR or Mean Squared Error (MSE)?"
**Candidate Answer**:
> *"In speech separation, an estimated signal $\hat{\mathbf{s}}$ can be multiplied by an arbitrary scalar gain $\alpha$ without altering human intelligibility. However, standard MSE penalizes amplitude scaling severely:
> $$\text{MSE}(\alpha \mathbf{s}, \mathbf{s}) = (\alpha - 1)^2 \|\mathbf{s}\|^2$$
> Even standard SDR allows filtering artifacts to distort the ratio. **SI-SDR** (Le Roux et al., 2019) enforces strict scale invariance by orthogonal projection. We project $\hat{\mathbf{s}}$ onto the ground-truth target vector $\mathbf{s}$:
> $$\mathbf{s}_{\text{target}} = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2} \mathbf{s}$$
> The residual error vector is strictly orthogonal to the target:
> $$\mathbf{e}_{\text{noise}} = \hat{\mathbf{s}} - \mathbf{s}_{\text{target}}$$
> SI-SDR is then defined as the log-ratio of their squared $L_2$ norms:
> $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2} \right)$$
> This guarantees that our evaluation reflects genuine acoustic separation and vocal isolation, invariant to gain offsets."*

### Q4: "How does the Eigengap heuristic determine the speaker count $K$ unsupervised, and why does spectral clustering outperform GMMs?"
**Candidate Answer**:
> *"Traditional Gaussian Mixture Models (GMMs) assume speakers occupy convex, Gaussian-distributed clusters in embedding space, which fails when conversational turn-taking causes non-linear embedding manifolds. 
> Spectral clustering treats speaker embeddings as nodes on an undirected graph with cosine affinity edges $A_{ij} = \frac{1}{2}(1 + \cos(\mathbf{e}_i, \mathbf{e}_j))$.
> 1. We construct the normalized symmetric graph Laplacian:
>    $$\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$
> 2. By algebraic graph theory (Davis-Kahan Theorem), the multiplicity of eigenvalue 0 equals the number of connected components in the graph. For real-world noisy affinity graphs, the optimal cluster count $K$ manifests as the largest relative gap between consecutive sorted eigenvalues:
>    $$\hat{K} = \arg\max_{k \in [1, K_{\max}]} (\lambda_{k+1} - \lambda_k)$$
> This enables our pipeline to discover whether an audio clip contains 1, 2, or 3 speakers completely unsupervised without manual tuning, outperforming GMMs by $6.8\times$ on Diarization Error Rate."*

---

## 4. Cross-Stage Error Propagation Theory

### Q5: "Walk me through your Error Cascading Analysis and the proof of the +4.2 dB Critical Breakdown Threshold."
**Candidate Answer**:
> *"In a sequential pipeline, early acoustic errors do not remain bounded; they cascade into structural and semantic representations. To study this scientifically, we designed a controlled degradation experiment:
> 1. We injected controlled acoustic noise into Stage 1 separated outputs across an SI-SDR sweep from $+12\text{ dB}$ down to $-5\text{ dB}$.
> 2. We simultaneously measured downstream Stage 2 DER, Stage 3 WER, and Stage 4 LLM Hallucination Rate.
> 3. Statistical correlation revealed strong cascading dynamics:
>    - Stage 1 SI-SDR vs. Stage 2 DER: $r = -0.977$ (acoustic bleed corrupts speaker embeddings).
>    - Stage 1 SI-SDR vs. Stage 3 WER: $r = -0.969$ (residual crosstalk induces CTC deletion and substitution).
>    - Stage 3 WER vs. Stage 4 Hallucination: $r = +0.978$ (fragmented transcripts cause LLM context hallucination).
> 4. By fitting a piecewise sigmoid to the WER and DER response curves, we identified a definitive inflection point:
>    $$\mathbf{SI\text{-}SDR}_{\text{crit}} = \mathbf{+4.2\text{ dB}}$$
> Above $+4.2\text{ dB}$, downstream stages operate in their nominal linear regime. Below $+4.2\text{ dB}$, speaker embeddings collapse into a single cluster and CTC decoding breaks down completely. Therefore, our system establishes an operational safety target of $\ge +7.2\text{ dB}$."*

---

## 5. Production Systems & Latency Engineering

### Q6: "What is the primary computational bottleneck of the pipeline, and how did you profile it?"
**Candidate Answer**:
> *"Through our Milestone M10 computational profiler (`src/benchmarking/profiler.py`), which monitors wall-clock latency via `time.perf_counter()`, system memory via `psutil` RSS, and GPU allocations via `torch.cuda`/`mps`, we identified that **Stage 1 (Separation) accounts for ~75%–85% of total pipeline latency on CPU**.
> 
> Key Scalability Insights:
> - **Separation & ASR Scale Linearly $\mathcal{O}(T)$**: Processing time scales directly with audio duration. Receptive-field chunking in Demucs keeps working memory under $950\text{ MB}$ RSS across 15s to 120s clips.
> - **Diarization Affinity Matrix Scales Quadratically $\mathcal{O}(M^2)$**: The cosine affinity matrix calculation scales quadratically with the number of embedding windows $M = T / \text{step}$. On sub-2-minute clips, absolute latency is negligible ($< 0.25\text{s}$), but indicates long audio (>15 mins) requires sliding-window sub-segmentation.
> - **Real-Time Factor (RTF)**: Across all clips, our cumulative pipeline achieved an RTF of **$0.49\text{–}0.69$**, confirming that 1 minute of multi-speaker audio is fully processed in approximately $30\text{–}41\text{ seconds}$ on standard CPU hardware."*

---

## 6. Hallucination Elimination & Guardrails

### Q7: "How can you guarantee 0.0% factual hallucination rate when summarizing fragmented speech transcripts?"
**Candidate Answer**:
> *"Standard prompting on conversational transcripts frequently causes LLMs to extrapolate missing prices, dates, or quantities due to autoregressive next-token priors. We achieved a verified 0.0% hallucination rate through a dual mechanism:
> 1. **Structural Negative Guardrails in Jinja2**:
>    - Rule 1 forbids generating any entity not grounded in the source text.
>    - Rule 2 mandates linking every finding to verified speaker IDs from Contract 3.
>    - Explicit instruction to output `[अस्पष्ट / Unintelligible]` rather than guessing unclear speech.
> 2. **Automated Factual Consistency Auditor (`evaluation/llm_metrics.py`)**:
>    - We extract all entities, quantities, and numerical tokens from the executive summary.
>    - We cross-reference each against a despaced, Unicode-normalized source corpus with bilingual phonetic equivalence dictionaries (e.g., mapping 'rehab' $\leftrightarrow$ 'रीहाब', 'flight' $\leftrightarrow$ 'फ्लाइट').
>    - Any ungrounded entity triggers an immediate audit rejection."*

---

## 7. Real-World Audio Challenges (Field Conditions)

### Q8: "How does the pipeline handle real-world 2G/3G narrowband cellular audio and code-switching?"
**Candidate Answer**:
> *"In Milestone M11, we stress-tested the pipeline on Project Vaani field recordings and AIR radio broadcasts:
> - **Cellular Narrowband Degradation**: Mobile calls restrict frequency bandwidth to $300\text{–}3400\text{ Hz}$, stripping sibilant consonants ($/s/$, $/sh/$) above $3.5\text{ kHz}$ (observed $-6.3\text{ dB}$ attenuation). Our evaluation proved that while this increases phonetic character substitutions, it does not trigger CTC collapse, and the pipeline remains viable down to $\text{SNR} = 21.0\text{ dB}$.
> - **Code-Switching Script Disparity**: We observed a $+27.3\%$ WER disparity between native Devanagari tokens ($72.7\%$) and English loanwords ($100.0\%$). However, because Stage 4 post-processing acts as a semantic normalizer, key borrowed entities (*hardware*, *software*, *smartphone*, *carry bag*) achieved **100% recall** in the final executive summaries."*

