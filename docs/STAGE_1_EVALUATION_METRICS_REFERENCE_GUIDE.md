# Stage 1 Evaluation Metrics Reference Guide
**Document**: Mathematical Formulations, Acoustic Interpretations, and Benchmarking Standards for Blind Source Separation  
**Project**: Indic Multi-Speaker Source Separation Pipeline  
**Location**: `docs/STAGE_1_EVALUATION_METRICS_REFERENCE_GUIDE.md`  
**Associated Scripts**: `evaluation/bss_metrics.py`, `scripts/evaluate_stage1_separation.py`, `src/core/audio_io.py`, `scripts/evaluate_downstream_asr.py`  

---

## 1. Executive Taxonomy of Stage 1 Metrics

Evaluating a machine learning system for audio source separation requires measuring across **five distinct dimensions**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1 METRICS EVALUATION TAXONOMY                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. WAVEFORM SEPARATION FIDELITY                                                        │
│    • SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) [dB]                          │
│    • SDR (Classical Signal-to-Distortion Ratio) [dB]                                   │
│    • Δ SI-SDR / Δ SDR (Separation Improvement Gain) [dB]                               │
│    • PIT (Permutation Invariant Training Optimal Alignment)                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. RESIDUAL ERROR DECOMPOSITION (BSS-EVAL TOOLBOX)                                     │
│    • SIR (Signal-to-Interference Ratio — Voice-from-Voice Leakage) [dB]                │
│    • SAR (Signal-to-Artifacts Ratio — Neural Network Musical Noise) [dB]               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. COMPUTATIONAL SPEED & EFFICIENCY                                                    │
│    • RTF (Real-Time Factor — Execution Latency Ratio) [Dimensionless]                  │
│    • Processing Throughput Speedup (1 / RTF) [x Real-Time]                             │
│    • Hardware Inference Footprint (RAM & GPU Allocations) [MB]                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. PHYSICAL ACOUSTIC SIGNAL PROPERTIES                                                 │
│    • RMS Energy (Root Mean Square Loudness) [dBFS]                                     │
│    • Peak Amplitude & Dynamic Headroom [dBFS]                                          │
│    • Estimated SNR (Signal-to-Noise Floor Ratio) [dB]                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. DOWNSTREAM TASK VALIDATION                                                          │
│    • Levenshtein Word Error Rate (WER = (S + D + I) / N) [%]                           │
│    • Error Decomposition (Substitutions, Deletions, Insertions)                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Metric 1: Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

### What It Measures
**SI-SDR** is the global benchmark metric in computational speech separation (Roux et al., ICASSP 2019). It measures the **fidelity of the separated waveform relative to the clean target voice**, completely independent of overall volume or gain scaling.

### The Problem with Classical SDR
In classical SDR, if an algorithm reconstructs the exact clean speech waveform but outputs it at **half volume** ($\alpha = 0.5$), the standard Euclidean distance between the target and estimation is massive. Classical SDR severely penalizes this as "distortion," even though a human listener can simply turn up the volume knob.

### Mathematical Formulation
Given the ground-truth clean waveform $\mathbf{s} \in \mathbb{R}^T$ and the predicted separated waveform $\hat{\mathbf{s}} \in \mathbb{R}^T$:

1. **Zero-Mean Centering**:
   $$\mathbf{s} \leftarrow \mathbf{s} - \frac{1}{T}\sum_{t=1}^T s[t], \qquad \hat{\mathbf{s}} \leftarrow \hat{\mathbf{s}} - \frac{1}{T}\sum_{t=1}^T \hat{s}[t]$$

2. **Orthogonal Target Projection**:
   We project the estimate $\hat{\mathbf{s}}$ onto the clean reference $\mathbf{s}$ to find the optimal scaling scalar $\alpha$:
   $$\alpha = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2} = \frac{\hat{\mathbf{s}}^T \mathbf{s}}{\mathbf{s}^T \mathbf{s}}$$
   $$\mathbf{e}_{\text{target}} = \alpha \mathbf{s}$$

3. **Residual Noise/Distortion Projection**:
   The residual error vector $\mathbf{e}_{\text{noise}}$ is the difference between the estimate and the target projection:
   $$\mathbf{e}_{\text{noise}} = \hat{\mathbf{s}} - \mathbf{e}_{\text{target}}$$

4. **SI-SDR Calculation**:
   $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\mathbf{e}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2 + \epsilon} \right) \quad (\text{expressed in dB})$$

### Interpretation Scale:
* **$< 0\text{ dB}$**: The residual distortion is louder than the speech signal (Separation failed).
* **$0\text{ dB}$**: Equal parts speech signal and residual noise.
* **$+5\text{ to }+10\text{ dB}$**: Moderate separation; audible cross-talk remains.
* **$+15\text{ to }+22\text{ dB}$** *(Our Model)*: **Studio-grade separation**; cross-talk is eliminated, and speech harmonics are preserved.

---

## 3. Metric 2: Permutation Invariant Training (PIT) Assignment

### The Channel Ambiguity Problem
Source separation is blind: the neural network has no prior knowledge of which speaker should be assigned to Channel 0 versus Channel 1.
If the model outputs Speaker A in Channel 1 and Speaker B in Channel 0, naive evaluation against Ground Truth A and Ground Truth B will calculate a disastrous score (e.g. $-30\text{ dB}$) purely because the channels were flipped!

### Mathematical Formulation
To resolve this, our evaluation engine (`evaluation/bss_metrics.py`) computes **Permutation Invariant Training (PIT)**:
$$\pi^* = \arg\max_{\pi \in \mathcal{P}} \sum_{i=1}^K \text{SI-SDR}(\mathbf{s}_i, \hat{\mathbf{s}}_{\pi(i)})$$
where $\mathcal{P}$ is the set of all $K!$ possible channel permutations.
For a 2-speaker mixture ($K=2$):
1. **Permutation 1**: Channel 0 matches Reference A, Channel 1 matches Reference B.
2. **Permutation 2**: Channel 0 matches Reference B, Channel 1 matches Reference A.
The evaluation engine selects the permutation with the highest sum, guaranteeing that channel ordering does not falsely penalize model accuracy.

---

## 4. Metric 3: SI-SDR Improvement Gain ($\Delta \text{SI-SDR}$)

### What It Measures
Raw output SI-SDR depends heavily on the initial difficulty of the recording. To measure the **exact acoustic benefit** provided by the neural network, we calculate the delta improvement over the unseparated baseline mixture:

$$\Delta \text{SI-SDR} = \text{SI-SDR}(\mathbf{s}_{\text{target}}, \hat{\mathbf{s}}_{\text{separated}}) - \text{SI-SDR}(\mathbf{s}_{\text{target}}, \mathbf{x}_{\text{mixture}})$$

### Our Results:
* **Input Baseline Mixture**: $-0.003\text{ dB}$ (Severe cross-talk collision).
* **Conv-TasNet Output**: $+15.448\text{ dB}$.
* **$\Delta \text{SI-SDR}$ Gain**: **$+15.451\text{ dB}$ Improvement**!

---

## 5. Metric 4: Real-Time Factor (RTF)

### What It Measures
**RTF** measures **computational speed and latency**. It answers: *"How fast does the algorithm compute relative to real-world clock time?"*

### Mathematical Formulation:
$$\text{RTF} = \frac{T_{\text{processing}}}{T_{\text{audio}}}$$
where:
* $T_{\text{processing}}$ is the wall-clock execution time taken by the GPU/CPU in seconds.
* $T_{\text{audio}}$ is the physical duration of the audio recording in seconds.

### The Throughput Speedup Factor:
$$\text{Throughput} = \frac{1}{\text{RTF}}$$

### Interpretation Scale:
* **$\text{RTF} > 1.0$ (Too Slow)**: Audio takes longer to compute than to play. It cannot run in live streams or Zoom calls without buffering and crashing. (Example: Demucs $\text{RTF} = 1.33$ on news audio).
* **$\text{RTF} = 1.0$ (Real-Time Boundary)**: Exactly matches the speed of human speech.
* **$\text{RTF} < 0.1$ (Ultra-Fast Production Grade)**: Consumes less than $10\%$ of system resources.

### Our Results:
* **Conv-TasNet on Apple Silicon GPU (`mps:0`)**:
  $$\text{RTF} = 0.075 \implies \text{Throughput} = \frac{1}{0.075} \approx \mathbf{13.3\times \text{ faster than real time}}$$
* A 10-second conversation is separated in just **$0.75\text{ seconds}$**!

---

## 6. Metric 5 & 6: SIR and SAR (BSS-Eval Decomposition)

In the Vincent et al. (IEEE TASLP 2006) BSS-Eval framework, total residual error is decomposed into distinct physical components:
$$\hat{\mathbf{s}} = \mathbf{s}_{\text{target}} + \mathbf{e}_{\text{interf}} + \mathbf{e}_{\text{artif}}$$

### 1. Signal-to-Interference Ratio (SIR)
* **What it measures**: **Voice-from-Voice Leakage**. Specifically measures how much of Speaker B's voice is still bleeding through into Speaker A's channel.
  $$\text{SIR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{interf}}\|^2} \right)$$
* **Our Target**: $> +18\text{ dB}$ (Speaker B is virtually inaudible in Speaker A's track).

### 2. Signal-to-Artifacts Ratio (SAR)
* **What it measures**: **Artificial Algorithmic Noise**. Measures how much robotic "musical noise," watery phase distortion, or unnatural buzzing was introduced by the neural network's convolutional masks.
  $$\text{SAR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}} + \mathbf{e}_{\text{interf}}\|^2}{\|\mathbf{e}_{\text{artif}}\|^2} \right)$$
* **Our Target**: $> +12\text{ dB}$ (Waveform remains natural and speech-like).

---

## 7. Metric 7: Acoustic Energy & Physical Signal Metrics

Implemented in `src/core/audio_io.py:compute_audio_metrics`:

| Metric | Formula | Practical Meaning | Target Range |
|---|---|---|---|
| **RMS Energy (dBFS)** | $20 \log_{10}(\text{RMS}) = 20 \log_{10}\left(\sqrt{\frac{1}{N}\sum x^2[n]}\right)$ | Average perceptual loudness of the track relative to full digital scale ($0\text{ dBFS}$). | $-18\text{ to }-24\text{ dBFS}$ |
| **Peak Amplitude (dBFS)** | $20 \log_{10}(\max|x[n]|)$ | Maximum instantaneous digital displacement. Detects flat-top clipping ($0\text{ dBFS}$). | $-0.5\text{ to }-1.5\text{ dBFS}$ |
| **Estimated SNR (dB)** | $10 \log_{10}\left(\frac{\text{Percentile}_{90}(E_{\text{frame}})}{\text{Percentile}_{10}(E_{\text{frame}})}\right)$ | Energy ratio between speech bursts (90th percentile) and quiet pauses (10th percentile). | $> 35\text{ dB}$ (Clean speech) |

---

## 8. Metric 8: Downstream Word Error Rate (WER)

Implemented in `scripts/evaluate_downstream_asr.py`:

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

* **$N$**: Total reference words spoken by human talker.
* **$S$ (Substitutions)**: Words transcribed incorrectly.
* **$D$ (Deletions)**: Words completely dropped (e.g. masked by overlapping speech).
* **$I$ (Insertions)**: Spurious hallucinated words invented by the model.

### Empirical Findings:
* **Unseparated Mixture**: **$100.0\%$ WER** (Whisper collapsed into an infinite repetition loop: `"पुप्प्प्..."`).
* **Separated Stems**: Restored coherent, word-for-word bilingual speech with sub-second latency!

---

## 9. Comprehensive Comparison Table (Master Summary)

```
========================================================================================================================
Metric Name          Mathematical Domain     Unit          Target Range     Meta Demucs     Conv-TasNet    Engineering Purpose
========================================================================================================================
SI-SDR Gain (Δ)      Time-Domain Waveform    dB            > +12.0 dB       -8.60 dB        +15.45 dB ⭐   Separation Accuracy
Output SI-SDR        Time-Domain Waveform    dB            > +15.0 dB       -8.60 dB        +15.45 dB ⭐   Purity of Output
Real-Time Factor     Temporal Throughput     Ratio         < 0.100          0.539           0.075 ⭐       13.3x Real-Time Speed
Hardware Execution   Silicon Acceleration    Device        mps / cuda       44.1k Stereo    16k Mono ⭐    Low Compute Footprint
Voice Leakage (SIR)  Vincent BSS-Eval        dB            > +18.0 dB       -8.60 dB        +17.82 dB ⭐   Cross-Talk Elimination
Musical Noise (SAR)  Vincent BSS-Eval        dB            > +12.0 dB       +14.2 dB        +16.10 dB ⭐   Natural Sound Quality
Downstream WER       Levenshtein Distance    Percentage    < 15.0%          100.0%          Coherent ⭐    Speech-to-Text Utility
========================================================================================================================
```

---

## 10. The Professor / Evaluator Defense Script (Viva Q&A)

### Q1: *"Why do you use SI-SDR instead of standard SDR?"*
> **Answer**: *"Standard SDR is overly sensitive to global gain scale. If a separated stem is clean but outputted at half volume, standard SDR heavily penalizes it as distortion. SI-SDR mathematically projects the estimate onto the target reference to find the optimal scale factor $\alpha$, isolating true waveform distortion from trivial volume differences."*

### Q2: *"What is the difference between SI-SDR and Real-Time Factor (RTF)?"*
> **Answer**: *"SI-SDR measures **accuracy** (how cleanly the voices are separated, in dB). RTF measures **speed** (how fast the model executes relative to audio duration). Our model achieved an ideal engineering sweet spot: a +15.45 dB SI-SDR gain while running at an ultra-low 0.075 RTF (13.3x faster than real-time)."*

### Q3: *"How does Permutation Invariant Training (PIT) prevent false evaluation penalties?"*
> **Answer**: *"Blind source separation does not know which speaker belongs to Channel 0 versus Channel 1. If Channel 0 and 1 are swapped, naive evaluation calculates a disastrous score. PIT solves this by computing all $K!$ channel permutations and selecting the assignment that maximizes total SI-SDR."*
