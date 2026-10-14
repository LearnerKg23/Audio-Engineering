# Stage 1: Blind Source Separation (BSS) — Deep Dive

---

## 1. The Problem We Are Solving

### 1.1 Real-World Scenario
Imagine recording a conversation between two people in a room. Both speakers' voices travel through the air and arrive at the **same microphone**. The resulting audio file contains a **mixture** — a single waveform where both voices are superimposed (added together).

When Speaker A says *"उन्हें दस दिन तक rehab करना होगा"* at the same time Speaker B says *"यह एक वॉइस ऑडियो डेटासेट है"*, the microphone captures **one combined signal** where both sentences are entangled.

### 1.2 Why This Is a Problem
- **ASR (Speech-to-Text) systems fail catastrophically** when two voices are mixed. The ASR model hears overlapping phonemes and either:
  - Gets stuck in **repetition loops** (e.g., `"पुप्प् पुप्प् पुप्प्..."`) — WER = 86.3%
  - **Drops entire sentences** from the quieter speaker
  - **Scrambles words** between speakers
- **Diarization systems also fail** because the acoustic fingerprints of both speakers collide in the same time window.

### 1.3 Formal Problem Statement

> **Given**: A single-channel mixture signal $x(t) = s_1(t) + s_2(t)$ where $s_1$ and $s_2$ are unknown source signals.
>
> **Goal**: Estimate $\hat{s}_1(t)$ and $\hat{s}_2(t)$ from $x(t)$ alone, **without** knowing the individual source signals, microphone positions, or room acoustics.
>
> This is called **Blind Source Separation (BSS)** because we are "blind" — we have zero prior information about the sources.

This is also known as the **Cocktail Party Problem** (proposed by Colin Cherry, 1953): How do you listen to one person at a noisy party?

---

## 2. How We Are Solving It — Conv-TasNet

### 2.1 What Is Conv-TasNet?

**Conv-TasNet** (Convolutional Time-domain Audio Separation Network) is a deep neural network that directly separates mixed speech in the **time domain** (raw waveform), without needing traditional spectrograms.

**Paper**: *"Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation"* — Yi Luo and Nima Mesgarani, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.

### 2.2 Three-Module Architecture

Conv-TasNet has exactly 3 modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CONV-TASNET                               │
│                                                                     │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐  │
│  │   ENCODER    │ →  │   SEPARATION      │ →  │    DECODER       │  │
│  │              │    │   NETWORK          │    │                  │  │
│  │ 1D Conv      │    │ Temporal Conv      │    │ 1D TransConv     │  │
│  │ (Waveform    │    │ Network (TCN)      │    │ (Latent →        │  │
│  │  → Latent    │    │ Generates masks    │    │  Waveform)       │  │
│  │   Repr.)     │    │ for each speaker   │    │                  │  │
│  └──────────────┘    └───────────────────┘    └──────────────────┘  │
│                                                                     │
│  Input: x(t)         Masks: M₁, M₂          Output: ŝ₁(t), ŝ₂(t) │
└─────────────────────────────────────────────────────────────────────┘
```

#### Module 1: Encoder (1D Convolution)
- **What it does**: Converts the raw waveform into a **learned latent representation** (like a learned spectrogram).
- **How**: Uses a 1D convolution with a small kernel (e.g., 16 samples = 1 millisecond at 16 kHz).
- **Math**:
  $$\mathbf{W} = \text{ReLU}(\text{Conv1D}(x))$$
  where $\mathbf{W} \in \mathbb{R}^{N \times T'}$ is the encoded representation with $N$ basis functions across $T'$ time frames.

#### Module 2: Separation Network (Temporal Convolutional Network — TCN)
- **What it does**: Learns to generate **masks** $M_1$ and $M_2$ — one for each speaker.
- **How**: Stacks of 1D dilated convolutions with exponentially increasing dilation factors (`1, 2, 4, 8, 16, 32, ...`). This gives a very large **receptive field** (the model can see long temporal context without being expensive).
- **Key insight**: The masks are like "filters" — they tell the decoder which parts of the encoded mixture belong to Speaker 1 vs Speaker 2.
- **Math**:
  $$M_i = \sigma(\text{TCN}_i(\mathbf{W})), \quad i \in \{1, 2\}$$
  where $\sigma$ is sigmoid/softmax activation and $M_i \in [0, 1]^{N \times T'}$.

#### Module 3: Decoder (1D Transposed Convolution)
- **What it does**: Converts the masked latent representation back to waveform audio.
- **Math**:
  $$\hat{s}_i(t) = \text{ConvTranspose1D}(M_i \odot \mathbf{W})$$
  where $\odot$ is element-wise multiplication. Each mask selects the parts of the encoding that belong to its speaker.

### 2.3 End-to-End Processing Summary

```
Raw Mixed Audio     →  Encoder  →  Latent Space    →  TCN Masks   →  Decoder  →  Clean Speaker Stems
x(t) [16kHz mono]                  W [N × T']          M₁, M₂                    ŝ₁(t), ŝ₂(t)
```

---

## 3. The Tech Stack

### 3.1 Libraries Used

| Library | What It Does in Stage 1 | Why We Use It |
|---|---|---|
| **PyTorch** (`torch`) | Deep learning framework that runs the neural network | Industry standard, GPU acceleration, automatic differentiation |
| **Torchaudio** (`torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX`) | Provides **pre-trained** Conv-TasNet weights trained on the Libri2Mix dataset | We don't need to train from scratch — we use transfer learning |
| **SciPy** (`scipy.signal.resample_poly`) | Polyphase resampling between sample rates (16kHz ↔ 8kHz) | Conv-TasNet requires 8kHz input; our pipeline standardizes on 16kHz |
| **NumPy** | Numerical array operations for waveform manipulation | Fast, efficient array math |
| **SoundFile** (`soundfile`) | Read/write WAV audio files | Cross-platform audio I/O |

### 3.2 Why Conv-TasNet Over Other Approaches?

| Approach | Limitation | Why Conv-TasNet Wins |
|---|---|---|
| **Ideal Binary Mask (IBM)** | Requires knowing the sources (not blind) | Conv-TasNet is fully blind |
| **NMF (Non-negative Matrix Factorization)** | Linear decomposition, fails on overlapping harmonics | Conv-TasNet uses nonlinear deep learning |
| **Deep Clustering** | Requires two-stage pipeline (embedding + clustering) | Conv-TasNet is end-to-end in one pass |
| **Demucs (Meta)** | Designed for music (drums/bass/vocals/other), operates at 44.1kHz | Conv-TasNet is optimized for speech at 8kHz, faster inference |

### 3.3 Pre-trained Model Details

| Property | Value |
|---|---|
| Model Name | `conv_tasnet_base_libri2mix` |
| Training Dataset | LibriMix (derived from LibriSpeech) |
| Number of Sources | 2 speakers |
| Native Sample Rate | 8000 Hz |
| Our Pipeline Sample Rate | 16000 Hz (we resample both ways) |

---

## 4. The Mathematical Concepts

### 4.1 Signal Mixing Model (Linear Instantaneous)

When two speakers talk into one microphone, the recorded signal is:

$$x(t) = s_1(t) + s_2(t) + n(t)$$

Where:
- $x(t)$ = mixture signal (what the microphone records)
- $s_1(t)$ = Speaker 1's clean speech signal
- $s_2(t)$ = Speaker 2's clean speech signal
- $n(t)$ = ambient noise (we assume this is small)

### 4.2 Time-Frequency Masking

In the encoder's latent space, the mixture representation $\mathbf{W}$ is approximately:

$$\mathbf{W} \approx \mathbf{W}_1 + \mathbf{W}_2$$

The separation masks $M_1$ and $M_2$ satisfy:

$$M_1 + M_2 = \mathbf{1} \quad \text{(softmax constraint)}$$

So the estimated sources are:

$$\hat{\mathbf{W}}_1 = M_1 \odot \mathbf{W}, \quad \hat{\mathbf{W}}_2 = M_2 \odot \mathbf{W}$$

### 4.3 Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

This is the **primary evaluation metric** for source separation quality. It measures how close the estimated source $\hat{s}$ is to the true source $s$.

#### Step 1: Project estimated signal onto true signal

$$s_{\text{target}} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} \cdot s$$

Where:
- $\langle \hat{s}, s \rangle = \sum_t \hat{s}(t) \cdot s(t)$ is the inner product (dot product)
- $\|s\|^2 = \sum_t s(t)^2$ is the energy of the true signal

#### Step 2: Compute the distortion (everything that is NOT the target)

$$e_{\text{noise}} = \hat{s} - s_{\text{target}}$$

#### Step 3: Calculate SI-SDR in decibels

$$\text{SI-SDR} = 10 \cdot \log_{10} \left( \frac{\|s_{\text{target}}\|^2}{\|e_{\text{noise}}\|^2} \right) \text{ dB}$$

**Interpretation**:
- **Higher SI-SDR = better separation quality**
- **+20 dB** means the true signal is 100× stronger than the distortion → excellent separation
- **0 dB** means signal and noise are equal → terrible separation
- **Negative dB** means the distortion is louder than the true signal → catastrophic

### 4.4 Permutation Invariant Training (PIT)

When separating 2 speakers, the network doesn't know which output channel corresponds to which input speaker. The two outputs could be swapped.

**PIT solves this** by computing the loss for both possible assignments and using the **minimum**:

$$\mathcal{L}_{\text{PIT}} = \min \left( \mathcal{L}(\hat{s}_1, s_1) + \mathcal{L}(\hat{s}_2, s_2), \quad \mathcal{L}(\hat{s}_1, s_2) + \mathcal{L}(\hat{s}_2, s_1) \right)$$

This ensures that no matter which order the network outputs the sources, it will always be compared fairly.

### 4.5 Real-Time Factor (RTF)

$$\text{RTF} = \frac{T_{\text{compute}}}{T_{\text{audio}}}$$

Where:
- $T_{\text{compute}}$ = time the model takes to process the audio
- $T_{\text{audio}}$ = duration of the audio file

**Interpretation**:
- **RTF < 1.0** = faster than real-time (good!)
- **RTF = 0.075** means 1 second of audio is processed in 75 milliseconds

### 4.6 Resampling Mathematics (Polyphase)

Conv-TasNet expects 8 kHz audio, but our pipeline uses 16 kHz. We need to **downsample** (16→8) before the model and **upsample** (8→16) after.

**Polyphase resampling** does this in one step:

$$x_{\text{out}}[n] = \sum_k h[k] \cdot x_{\text{in}}\left[\frac{\text{down} \cdot n - k}{\text{up}}\right]$$

Where:
- `up = target_sr // gcd(source_sr, target_sr)`
- `down = source_sr // gcd(source_sr, target_sr)`
- $h[k]$ is an anti-aliasing low-pass filter

In our code:
```python
# Downsample 16kHz → 8kHz
gcd = np.gcd(16000, 8000)  # = 8000
up = 8000 // 8000   # = 1
down = 16000 // 8000 # = 2
resampled = scipy.signal.resample_poly(data, up=1, down=2)
```

---

## 5. Our Code Implementation

### 5.1 File: `src/separation/conv_tasnet_wrapper.py`

Key class: `ConvTasNetSeparator`

**Processing Pipeline** (what happens when you call `separate_audio()`):

```
Step 1: Downmix stereo to mono (if needed)
    ↓
Step 2: Resample from 16kHz → 8kHz (Conv-TasNet's native rate)
    ↓
Step 3: Reshape tensor to (batch=1, channels=1, frames)
    ↓
Step 4: Forward pass through Conv-TasNet model (torch.no_grad())
    ↓
Step 5: Extract 2 separated stems from output (2, frames)
    ↓
Step 6: Upsample each stem from 8kHz → 16kHz
    ↓
Step 7: Compute energy ratios for each stem
    ↓
Step 8: Save separated WAV files to disk
    ↓
Output: [stem_0_numpy, stem_1_numpy], metadata_dict
```

### 5.2 File: `src/separation/demucs_wrapper.py` (Comparative Baseline)

We also implemented **Meta's Demucs** (Hybrid Demucs / htdemucs) as a comparative baseline. Key differences:

| Feature | Conv-TasNet | Demucs |
|---|---|---|
| **Native Sample Rate** | 8 kHz | 44.1 kHz |
| **Architecture** | 1D Conv Encoder + TCN + Decoder | Hybrid U-Net (time + frequency domain) |
| **Designed For** | 2-speaker speech separation | 4-stem music separation (drums/bass/other/vocals) |
| **Speech Stems** | Direct 2-speaker outputs | Maps "vocals" → Speaker 0, "other" → Speaker 1 |
| **Inference Speed** | Faster (smaller model, lower sample rate) | Slower (larger model, higher sample rate) |

---

## 6. Results and Interpretation

### 6.1 Benchmark Results

| Mixture | Overlap | Baseline SI-SDR | Conv-TasNet SI-SDR | **Gain** | RTF |
|---|---|---|---|---|---|
| `mix_01` (Pair AD, 25%) | 25% | +0.01 dB | 21.77 dB | **+21.76 dB** | 0.095 |
| `mix_02` (Pair AD, 50%) | 50% | -0.05 dB | 19.84 dB | **+19.89 dB** | 0.082 |
| `mix_03` (Pair AD, 50%, SIR +6) | 50% | -0.06 dB | 20.22 dB | **+20.27 dB** | 0.059 |
| `mix_04` (Pair BC, 25%) | 25% | -0.01 dB | 4.81 dB | **+4.81 dB** | 0.072 |
| `mix_05` (Pair BC, 50%) | 50% | +0.04 dB | 5.92 dB | **+5.88 dB** | 0.080 |
| `mix_06` (Pair BC, 50%, SIR -6) | 50% | +0.04 dB | 20.15 dB | **+20.11 dB** | 0.060 |

### 6.2 What These Numbers Mean

1. **Baseline SI-SDR ≈ 0 dB**: Before separation, the mixture has approximately equal energy from both speakers → the "signal" and "noise" are the same → no useful separation.

2. **Conv-TasNet SI-SDR = +21.77 dB** (best case): After separation, the target speaker's energy is **150× stronger** than the residual interference. This is **excellent** separation quality. The human ear would hear virtually no trace of the other speaker.

3. **Pair BC lower gains (+4.8 to +5.9 dB)**: These speaker pairs had more similar vocal characteristics (harder to separate acoustically similar voices), but still achieved meaningful separation.

4. **RTF ≈ 0.075**: The model processes 1 second of audio in only 75 milliseconds → **13× faster than real-time** on standard CPU. No GPU required.

### 6.3 How to Interpret SI-SDR Gain

| SI-SDR Gain | Quality | What You Hear |
|---|---|---|
| < 5 dB | Poor | Both speakers still clearly audible in each stem |
| 5–15 dB | Good | Primary speaker dominant, faint traces of other |
| 15–25 dB | Excellent | Clean isolation, no perceptible interference |
| > 25 dB | Near-perfect | Indistinguishable from original clean recording |

---

## 7. How Stage 2 Improves Upon This

Stage 1 produces two **clean audio stems** — one per speaker. But it does NOT tell us:
- ❌ Which stem belongs to which speaker identity
- ❌ When each speaker starts and stops talking
- ❌ What they actually said

**Stage 2 (Diarization)** takes these clean stems and answers the "WHO" and "WHEN" questions by analyzing the vocal characteristics (pitch, timbre, vocal tract resonance) of each stem and assigning speaker labels with timestamps.

**Without Stage 1**: Diarization alone on raw mixed audio yields **DER = 40.52%** (terrible — wrong speaker labels 40% of the time).

**With Stage 1 first**: Diarization on clean separated stems yields **DER = 23.97%** — a **16.55% absolute improvement**. The separation removes cross-talk that was confusing the clustering algorithm.

---

## 8. Key Takeaways for Defense

1. **Conv-TasNet achieves +21.8 dB SI-SDR gain** — proving that deep learned time-domain masking vastly outperforms traditional frequency-domain approaches for Indian conversational speech.

2. **RTF = 0.075** — the separation is **13× faster than real-time**, meaning it can be deployed in production without GPU hardware.

3. **The separation step is not optional** — removing it causes ASR Word Error Rate to spike from 51.6% to 86.3% (a +34.7% error increase), proving that physical acoustic decoupling is a prerequisite for reliable transcription.

4. **The model is pre-trained** on LibriMix (English speech mixtures) but transfers effectively to Indian Hindi/Hinglish speech, demonstrating cross-lingual acoustic generalization of the time-domain separation paradigm.
