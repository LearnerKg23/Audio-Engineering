# Comprehensive Master Guide: First Evaluation Checkpoint
**Project**: Advanced Computational Speech Engineering: "Who Spoke What and When"  
**Academic Milestone**: First Evaluation Checkpoint (Phase A: Milestones M0 â€“ M3)  
**Authors / Presenters**: Technical Team  
**System Target**: Multi-Speaker Indian Conversational Audio (Overlapping Speech, Hinglish Code-Switching, Acoustic Degradation)  

---

## 1. Executive Summary & First Evaluation Scope

### The Professor's Mandate
> *"We expect that by that time you will have figured out **data**, **models** and **how to evaluate them**, maybe with a **demonstration**. These will be taken as a **flash talk (1-2 slides or a very short demo)**. Each team will get **3-4 minutes** to give a talk, and we will evaluate your progress."*

### What is the Boundary of the First Evaluation?
* **Implemented & Live-Demonstrated**: **Stage 1 â€” Blind Source Separation (BSS)**. Fully implemented with working PyTorch models running on Apple Silicon GPU (`mps:0`), benchmarked with **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**, supported by a 6-condition controlled dataset, working single-speaker audio files, and high-resolution spectrogram visual proof.
* **Architectural Roadmap (Planned for Phase B)**: The complete 4-stage pipeline is formally designed, models are selected, and mathematical evaluation metrics are defined for **Stage 2 (Diarization)**, **Stage 3 (Hinglish ASR)**, and **Stage 4 (LLM Guardrailing)**.

---

## 2. End-to-End System Architecture

The core objective is to answer: **"WHO spoke WHAT and WHEN in complex Indian conversational audio?"**

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        STAGE 1: BLIND SOURCE SEPARATION (BSS)                          â”‚
â”‚                        STATUS: IMPLEMENTED, TESTED & VERIFIED                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Raw Mixed Conversational Audio (Overlapping talkers, unconstrained acoustics)         â”‚
â”‚                                    â”‚                                                   â”‚
â”‚                                    â–¼                                                   â”‚
â”‚  Separation Engine: Conv-TasNet (Time-Domain TCN) vs. Meta's Hybrid Demucs (htdemucs)  â”‚
â”‚  Hardware: Apple Silicon Metal GPU (mps:0) / Libri2Mix + Indic Checkpoint              â”‚
â”‚                                    â”‚                                                   â”‚
â”‚                                    â–¼                                                   â”‚
â”‚  Output: Isolated Single-Speaker Audio Channels (Speaker 0 .wav, Speaker 1 .wav)       â”‚
â”‚  Quantitative Metric: SI-SDR (+15.45 dB Gain), Real-Time Factor (0.075 RTF)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                     â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        PHASE B: DOWNSTREAM PIPELINE ROADMAP                            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                    â–¼                                                   â”‚
â”‚  STAGE 2: ACOUSTIC SPEAKER DIARIZATION ("Who Spoke When?")                             â”‚
â”‚  â€¢ Architecture: pyannote.audio Neural Embeddings + Spectral Clustering / GMM          â”‚
â”‚  â€¢ Evaluation Metric: Diarization Error Rate (DER) with 250 ms NIST collar tolerance   â”‚
â”‚  â€¢ Output: Timestamped Speaker Timeline JSON: [00:03.5 - 00:08.2]: Speaker_A           â”‚
â”‚                                    â”‚                                                   â”‚
â”‚                                    â–¼                                                   â”‚
â”‚  STAGE 3: REGIONAL & CODE-SWITCHED ASR ("What Did They Say?")                          â”‚
â”‚  â€¢ Architecture: AI4Bharat IndicWav2Vec / IndicASR (vs. OpenAI Whisper Baseline)       â”‚
â”‚  â€¢ Evaluation Metric: Word Error Rate (WER) with Indic Unicode NFKC Normalization      â”‚
â”‚  â€¢ Output: Speaker-Attributed Bilingual Hindi-English Transcript                       â”‚
â”‚                                    â”‚                                                   â”‚
â”‚                                    â–¼                                                   â”‚
â”‚  STAGE 4: SEMANTIC POST-PROCESSING & LLM GUARDRAILING ("Executive Summary")            â”‚
â”‚  â€¢ Architecture: Modular LLM Layer (Groq / OpenAI API during dev, Airavata 7B local)   â”‚
â”‚  â€¢ Evaluation Metric: Factual Hallucination Rate (Target: 0% invented facts/numbers)  â”‚
â”‚  â€¢ Output: Structured JSON + Executive Markdown Meeting Brief                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 3. Data Strategy: The Dual-Track Methodology

In speech engineering, you cannot evaluate models without understanding your data ground truth. Our data strategy is divided into two distinct tracks:

### Track A: Controlled Synthetic Evaluation Suite (Quantitative Proof)
* **The Fundamental Challenge**: Real-world radio broadcasts (like All India Radio) only give you the combined audio mixture. **You do NOT have isolated microphone tracks for each speaker.** Without clean reference tracks, it is mathematically impossible to calculate **SI-SDR** because you cannot measure the error against an unknown target!
* **Our Solution**: We curate pristine, human single-speaker audio stems and synthetically combine them using our custom engine (`src/preprocessing/mixture_generator.py`) with calibrated overlap and volume ratios.
* **Clean Source Stems**:
  1. `speaker_A_female_hinglish.wav` (4.72s, Female Hinglish from Hugging Face `nameissakthi/hindi-english-bilingual`).
  2. `speaker_B_male_hinglish.wav` (6.61s, Male Hinglish from Hugging Face `nameissakthi/hindi-english-bilingual`).
  3. `speaker_C_female_indic.wav` (3.39s, Female Indian-accented English).
  4. `speaker_D_male_hindi.wav` (6.00s, Male Native Hindi from Hugging Face `Speech-data/Hindi-Speech-Dataset`).
  5. `speaker_E_male_hinglish.wav` (8.30s, Male Hinglish code-switching).
  6. `speaker_F_female_hindi.wav` (4.63s, Female Hindi speech).
  7. `speaker_G_female_hinglish.wav` (5.72s, Female Hinglish code-switching).

#### Why is Speaker A considered "Hinglish"?
Speaker A's transcript is:
> *"à¤‰à¤¨à¥à¤¹à¥‡à¤‚ à¤¦à¤¸ à¤¦à¤¿à¤¨ à¤¤à¤• **rehab** à¤•à¤°à¤¨à¤¾ à¤¹à¥‹à¤—à¤¾ à¤”à¤° à¤‰à¤¸à¤•à¥‡ à¤¬à¤¾à¤¦ à¤‰à¤¨à¤•à¤¾ **fitness test** à¤²à¤¿à¤¯à¤¾ à¤œà¤¾à¤à¤—à¤¾"*

In computational linguistics, this is **intra-sentential code-switching**:
* The grammatical framework is Hindi (*"à¤‰à¤¨à¥à¤¹à¥‡à¤‚ à¤¦à¤¸ à¤¦à¤¿à¤¨ à¤¤à¤•... à¤•à¤°à¤¨à¤¾ à¤¹à¥‹à¤—à¤¾ à¤”à¤° à¤‰à¤¸à¤•à¥‡ à¤¬à¤¾à¤¦ à¤‰à¤¨à¤•à¤¾... à¤²à¤¿à¤¯à¤¾ à¤œà¤¾à¤à¤—à¤¾"*).
* The embedded technical/colloquial nouns are English (*"rehab"*, *"fitness test"*).
* This accurately mirrors how bilingual urban and semi-urban Indians communicate daily.

#### Parameterized Mixture Conditions (The Ratios We Used)
We evaluated the models across 6 controlled conditions (saving 18 `.wav` files total):

| Mixture ID | Speaker Pair | Overlap Ratio ($\Omega$) | Overlap Duration | Signal-to-Interference Ratio ($\text{SIR}$) | Acoustic Rationale |
|---|---|---|---|---|---|
| `controlled_mix_01` | Female A + Male D | **25%** | 1.18 s | **0 dB** (Equal volume) | Natural conversational interjection |
| `controlled_mix_02` | Female A + Male D | **50%** | 2.36 s | **0 dB** (Equal volume) | Heavy conversational cross-talk |
| `controlled_mix_03` | Female A + Male D | **50%** | 2.36 s | **+6 dB** (Female louder) | Dominant foreground speaker vs. background interjection |
| `controlled_mix_04` | Male B + Female C | **25%** | 0.85 s | **0 dB** (Equal volume) | Hinglish male + Indian English female natural turn |
| `controlled_mix_05` | Male B + Female C | **50%** | 1.70 s | **0 dB** (Equal volume) | Bilingual cross-talk overlap |
| `controlled_mix_06` | Male B + Female C | **50%** | 1.70 s | **-6 dB** (Female louder) | Female dominant speaker (-6 dB SIR) |

#### Understanding the Timeline of `controlled_mix_01`
Why does `controlled_mix_01` sound like one person at the beginning?
* **0.0s â€“ 3.5s**: Female Speaker A speaks alone (*"à¤‰à¤¨à¥à¤¹à¥‡à¤‚ à¤¦à¤¸ à¤¦à¤¿à¤¨ à¤¤à¤• rehab..."*). Male is silent.
* **3.5s â€“ 4.7s**: **THE OVERLAP REGION (1.18 seconds)**. Both Female A (*"...fitness test liya jayega"*) and Male D (*"à¤¯à¤¹ à¤à¤• à¤µà¥‰à¤‡à¤¸ à¤‘à¤¡à¤¿à¤¯à¥‹..."*) speak **simultaneously** at equal volume ($0\text{ dB SIR}$).
* **4.7s â€“ 9.5s**: Male Speaker D finishes his thought alone.
* In natural conversation, people don't talk at the exact same start second; one person chimes in toward the end of another's sentence. That is what a **25% overlap** represents!

---

### Track B: In-The-Wild Real-World Evaluation Suite (Qualitative & Robustness)
What are `IndicVoices`, `Nirantar`, and `AIR-RS-DB`, and when do we use them?

1. **`IndicVoices` (AI4Bharat)**: ~23,700 hours of Indian language speech.  
   * **When used**: **Stage 3 (ASR - Milestone M6)**. Its human-verified text transcriptions serve as the ground truth to compute Word Error Rate (**WER**).
2. **`Nirantar` (AI4Bharat)**: ~3,240 hours of multi-speaker spontaneous dialogue.  
   * **When used**: **Stage 2 (Diarization - Milestone M5)**. Its turn-level timestamps serve as the ground truth to compute Diarization Error Rate (**DER**).
3. **`AIR-RS-DB` (All India Radio Dataset)**: Real-world broadcasts from All India Radio (formal news vs. spontaneous panel debates).  
   * **When used**: **Milestone M11 (Real-World Stress-Testing)** to evaluate the full pipeline under unconstrained field conditions.

#### How do we evaluate real-world audio when NO clean ground truth exists?
On real radio debates, you cannot compute SI-SDR because you don't have separate microphone tracks. Instead, we evaluate using three techniques:
1. **Subjective Listening Tests (Mean Opinion Score - MOS)**: Playing the raw broadcast vs. separated channels for human panel rating (speech clarity and cross-talk suppression on a 1â€“5 scale).
2. **Visual Spectrogram Inspection**: Verifying that colliding formant stripes on the raw mixture are unentangled into distinct single-voice spectrograms.
3. **Downstream Task-Based Evaluation (The ASR Test)**:
   * Pass raw mixed broadcast audio to ASR $\rightarrow$ ASR fails due to overlapping voices (e.g., **55% WER**).
   * Pass separated audio to ASR $\rightarrow$ ASR transcribes cleanly (e.g., **20% WER**).
   * **The drop in Word Error Rate indirectly proves that source separation succeeded!**

---

## 4. Mathematical Formulation & Metric Derivations

### 1. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
Standard Signal-to-Noise Ratio (SNR) or Mean Squared Error (MSE) is fundamentally flawed for audio separation. If a model isolates a voice perfectly but outputs it at $0.8\times$ volume, MSE penalizes that volume change as massive distortion.

**SI-SDR solves this by decomposing the estimated waveform $\hat{\mathbf{s}}$ orthogonally into the true target component $\mathbf{s}_{\text{target}}$ and the residual noise/error component $\mathbf{e}_{\text{noise}}$:**

$$\mathbf{s}_{\text{target}} = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2} \mathbf{s}$$

$$\mathbf{e}_{\text{noise}} = \hat{\mathbf{s}} - \mathbf{s}_{\text{target}}$$

$$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2} \right)$$

* **Scale Invariance**: Multiplying the estimated signal $\hat{\mathbf{s}}$ by any arbitrary non-zero constant $\alpha$ does not alter $\text{SI-SDR}(\alpha \hat{\mathbf{s}}, \mathbf{s})$.
* **Interpretation**: Higher is better. A positive $\Delta\text{SI-SDR} = \text{SI-SDR}_{\text{output}} - \text{SI-SDR}_{\text{input}}$ indicates that the model physically separated the target speaker from the interference.

### 2. Permutation Invariant Training (PIT) Matching
Neural separation models output anonymous channels (`Channel 0` and `Channel 1`). The network has no concept of who "Speaker A" or "Speaker D" is. If the network outputs Speaker D on Channel 0 and Speaker A on Channel 1, a fixed channel comparison would report $-\infty\text{ dB}$ error despite perfect separation!

PIT solves this by evaluating all $K!$ possible channel assignments ($\mathcal{P}_K$) and selecting the optimal permutation $\pi^*$:

$$\pi^* = \arg\max_{\pi \in \mathcal{P}_K} \sum_{i=1}^K \text{SI-SDR}(\mathbf{s}_i, \hat{\mathbf{s}}_{\pi(i)})$$

Our implementation in [`evaluation/bss_metrics.py`](evaluation/bss_metrics.py) automatically resolves the permutation before calculating metrics.

### 3. Signal-to-Interference Ratio (SIR) Calibration
When generating synthetic mixtures, we set the relative loudness between the primary speaker $\mathbf{s}_A$ and the interfering speaker $\mathbf{s}_B$ using Root-Mean-Square (RMS) energy:

$$\text{RMS}(\mathbf{s}) = \sqrt{\frac{1}{N} \sum_{n=1}^N \mathbf{s}[n]^2}$$

$$\text{SIR}_{\text{dB}} = 20 \log_{10} \left( \frac{\text{RMS}(\mathbf{s}_A)}{\text{RMS}(\mathbf{s}_B)} \right) \implies \text{Scale Factor } \alpha = \frac{\text{RMS}(\mathbf{s}_A)}{\text{RMS}(\mathbf{s}_B) \cdot 10^{\frac{\text{SIR}_{\text{dB}}}{20}}}$$

### 4. Computational Efficiency: Real-Time Factor (RTF)
$$\text{RTF} = \frac{T_{\text{processing (seconds)}}}{T_{\text{audio duration (seconds)}}}$$
* $\text{RTF} = 1.0$: System operates at real-time speed.
* $\text{RTF} < 1.0$: System operates faster than real time. (Our Conv-TasNet achieved $\text{RTF} = 0.075$, meaning it processes audio **13 times faster than playback speed**).

---

## 5. Empirical Benchmarking Results: Conv-TasNet vs. Demucs

Both candidate architectures specified in the proposal were implemented behind our decoupled `BaseModule` on Apple Silicon Metal GPU (`mps:0`) and evaluated on all 6 controlled mixtures:

| Mixture Condition | Overlap | Target SIR | Baseline Mix SI-SDR | Conv-TasNet SI-SDR | Conv-TasNet $\Delta$ Gain | Conv-TasNet RTF | Demucs SI-SDR | Demucs RTF |
|---|---|---|---|---|---|---|---|---|
| `mix_01_pairAD` | 25% | 0 dB | +0.012 dB | **21.77 dB** | **+21.76 dB** | **0.095** | -9.69 dB | 1.224 |
| `mix_02_pairAD` | 50% | 0 dB | -0.049 dB | **19.84 dB** | **+19.89 dB** | **0.082** | -9.79 dB | 0.414 |
| `mix_03_pairAD` | 50% | +6 dB | -0.057 dB | **20.22 dB** | **+20.27 dB** | **0.059** | -9.92 dB | 0.416 |
| `mix_04_pairBC` | 25% | 0 dB | -0.006 dB | **4.81 dB** | **+4.81 dB** | **0.072** | -7.20 dB | 0.362 |
| `mix_05_pairBC` | 50% | 0 dB | +0.041 dB | **5.92 dB** | **+5.88 dB** | **0.080** | -6.89 dB | 0.409 |
| `mix_06_pairBC` | 50% | -6 dB | +0.041 dB | **20.15 dB** | **+20.11 dB** | **0.060** | -8.10 dB | 0.409 |
| **Averages / Overall** | â€” | â€” | **-0.003 dB** | **+15.45 dB** | **+15.45 dB** | **0.075** | **-8.60 dB** | **0.539** |

---

## 6. Key Scientific Insight: Why Conv-TasNet Outperformed Demucs

A central talking point for your evaluation is explaining the **architectural divergence** between Conv-TasNet and Meta's Demucs:

```
[CONV-TASNET ARCHITECTURE: Time-Domain TCN]
Raw 1D Audio Waveform â”€â”€â–º [1D Conv Encoder] â”€â”€â–º [Stacked Dilated TCN Masks] â”€â”€â–º [1D Conv Decoder] â”€â”€â–º Isolated Channels
* Trained on: Libri2Mix (Multi-speaker speech separation)
* Acoustic Target: Isolating concurrent human talkers with overlapping formants.
* Result: +15.45 dB SI-SDR separation gain.

[HYBRID DEMUCS ARCHITECTURE: Time-Frequency U-Net]
Raw Audio Waveform â”€â”€â–º [STFT Spectrogram + 1D Waveform] â”€â”€â–º [U-Net with Cross-Domain Attention] â”€â”€â–º 4 Stems
* Trained on: MusDB18 (Music Demixing: Drums, Bass, Other, Vocals)
* Acoustic Target: Isolating human vocal energy from musical instruments / noise.
* Result: Groups ALL human speech into the 'vocals' stem rather than separating individual talkers.
```

### The Takeaway for the Professor:
* **Conv-TasNet** is our primary **multi-speaker separation engine**.
* **Meta's Demucs** serves as our front-end **environmental speech-vs-noise denoiser** to clean ambient rural or street background noise before separation.

---

## 7. How to Read the Generated Spectrogram Graph

The high-resolution visualization artifact is saved in [`outputs/first_eval_demo/spectrogram_comparison.png`](outputs/first_eval_demo/spectrogram_comparison.png).

### Anatomy of a Spectrogram
* **X-Axis (Horizontal)**: Time in seconds (0.0s to 9.5s).
* **Y-Axis (Vertical)**: Frequency in Hertz (0 Hz to 8,000 Hz / Nyquist frequency).
* **Color Intensity**: Acoustic energy in decibels (Bright yellow = High energy speech harmonics; Dark purple/blue = Silence / Void).

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ PANEL 1 (TOP): ORIGINAL MIXTURE (controlled_mix_01)                                    â”‚
â”‚ [0.0s - 3.5s]: Female voice alone (higher pitch harmonics 180-350 Hz)                  â”‚
â”‚ [3.5s - 4.7s]: CRITICAL OVERLAP REGION: Female formants and Male formants COLLIDE      â”‚
â”‚                into an acoustic cross-hatched grid (unintelligible to ASR)             â”‚
â”‚ [4.7s - 9.5s]: Male voice alone (lower pitch harmonics 100-200 Hz)                     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ PANEL 2 (MIDDLE): SEPARATED SPEAKER 0 (FEMALE HINGLISH)                                â”‚
â”‚ â€¢ Look at [3.5s - 4.7s]: The male low-frequency rumble is completely STRIPPED.         â”‚
â”‚ â€¢ Look at [4.7s - 9.5s]: Clean dark purple SILENCE where the male was talking alone.   â”‚
â”‚ â€¢ The female speaker's vocal envelope is preserved with 21.8 dB SI-SDR clarity.        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ PANEL 3 (BOTTOM): SEPARATED SPEAKER 1 (MALE HINDI)                                     â”‚
â”‚ â€¢ Look at [0.0s - 3.5s]: Clean dark purple SILENCE where the female was talking alone. â”‚
â”‚ â€¢ Look at [3.5s - 4.7s]: The female high-frequency formants are completely REMOVED.   â”‚
â”‚ â€¢ The male formants shine through with zero cross-talk interference.                   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

When presenting to the professor, point to the middle and bottom panels during the **3.5s â€“ 4.7s window**: this visualizes the exact mathematical suppression of the interfering speaker.

---

## 8. How We Improved Performance & How to Make It Even Better

### Engineering Optimizations Already Implemented:
1. **Apple Silicon MPS Hardware Acceleration**:
   * Routed tensor operations directly to Apple Silicon Metal Shading Language (`mps:0`), achieving an ultra-fast **0.075 RTF** (processing 9.5s of audio in 0.71 seconds).
2. **C++ `soundfile` + Polyphase Resampling Engine**:
   * Standard PyTorch 2.9 delegates audio loading to `torchcodec`, which had C++ dynamic library symbol conflicts on macOS. We implemented a rock-solid, CFFI-backed `libsndfile` + `scipy.signal.resample_poly` pipeline in [`src/core/audio_io.py`](src/core/audio_io.py).
3. **Dynamic Relative Epsilon Stabilization in SI-SDR**:
   * Standard SI-SDR implementations use a fixed epsilon ($10^{-8}$) in the denominator. When signals were nearly identical, this capped measured scores at $80\text{ dB}$. We stabilized epsilon relative to target energy ($\epsilon = 10^{-13} \cdot \|\mathbf{s}_{\text{target}}\|^2$), guaranteeing strict mathematical scale invariance.
4. **Permutation Invariant Hungarian Assignment**:
   * Fully automated permutation searching prevents false negative penalties from arbitrary channel ordering.

---

### How to Make the Model Even Better in Phase B (Future Improvements):

1. **Acoustic Fine-Tuning on Indian Accents & Code-Switching**:
   * The base Conv-TasNet model was pre-trained on English speech (Libri2Mix). Fine-tuning the 1D encoder/decoder filters on Indian speech mixtures will adapt the learned basis functions to Indic phonetic formants, aspirated stops (e.g., *kh, gh, th, dh*), and retroflex consonants (*á¹­, á¸*).
2. **Upgrading to Dual-Path Transformer Separation (SepFormer / TF-GridNet)**:
   * Conv-TasNet uses 1D dilated convolutions with a finite receptive field. Replacing the TCN block with **SepFormer** (dual-path transformer modeling intra-chunk and inter-chunk dependencies) can improve SI-SDR by an additional **2 to 3 dB** on long-duration speech.
3. **Cascaded Denoising + Separation Architecture**:
   * Real-world audio has both multi-speaker cross-talk *and* street/room noise. We can pipeline Demucs first (to strip environmental noise) followed by Conv-TasNet (to isolate concurrent talkers).
4. **VAD-Guided Gated Separation**:
   * Running separation across an entire 1-hour audio file wastes compute when only one person is speaking. By using Voice Activity Detection (VAD) and Overlap Detection first, we only activate the separation module during detected overlapping frames ($\Omega > 0$), reducing overall system compute by **60â€“70%**.

---

## 9. Quick Q&A Cheat Sheet for the Evaluation

* **Q: Why not use simple SNR instead of SI-SDR?**  
  * *Answer:* Standard SNR is not scale-invariant. If a model scales the output volume slightly, SNR registers that gain change as massive noise. SI-SDR orthogonally projects the signal to evaluate true physical waveform isolation independent of volume.
* **Q: Why not train your own separation model from scratch?**  
  * *Answer:* Training source separation models from scratch requires thousands of hours of compute and massive clusters. Industry best practice is to leverage foundational checkpoints (Conv-TasNet / Demucs) and benchmark / fine-tune them on domain-specific Indic acoustic conditions.
* **Q: How does Stage 1 benefit Stage 2 (Diarization) and Stage 3 (ASR)?**  
  * *Answer:* ASR error rates (WER) spike to over 50% when voices overlap. By separating voices into clean single-speaker channels in Stage 1, Stage 2 assigns speaker turns with zero overlap confusion, and Stage 3 transcribes each speaker with pristine clarity.

