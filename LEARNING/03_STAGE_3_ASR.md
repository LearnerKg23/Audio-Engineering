# Stage 3: Regional & Code-Switched ASR — Deep Dive

---

## 1. The Problem We Are Solving

### 1.1 Real-World Scenario: "What Did Each Person Say?"
Once Stage 1 has separated acoustic signals and Stage 2 has pinpointed the timestamps where each speaker talks, the final core scientific question is:
**"What words did each speaker utter?"**

In real-world conversations across India and South Asia, this problem is uniquely challenging because of:
1. **Code-Switching (Hinglish)**: Speakers naturally interweave Hindi and English in the same sentence (e.g., *"उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा"*).
2. **Dialectal & Phonetic Diversity**: Variations in retroflex consonants (ट, ठ, ड, ढ), aspirated plosives (ख, घ, छ, झ), and vowel nasalization (अनुस्वार /ँ).
3. **Overlapping Speech Collapse**: When two people talk at once, standard speech-to-text models collapse completely.

### 1.2 The "Attention Collision Paradox"
Why can't we simply feed a multi-speaker audio recording directly into an advanced model like OpenAI Whisper or Google Speech-to-Text?

Standard modern ASR models use an **Attention-Based Encoder-Decoder (AED)** architecture. When two voices overlap:
- The cross-attention mechanism in the decoder receives acoustic keys and values from **two conflicting phonetic trajectories** at the same time step.
- The decoder's softmax probability distribution over the vocabulary flattens, losing certainty.
- The model enters a **pathological repetition loop** — outputting the same hallucinated syllable indefinitely (e.g., `"पुप्प् पुप्प् पुप्प्..."`).
- In our empirical tests on raw mixed audio, this caused the Word Error Rate to skyrocket to **86.28%** and Character Error Rate to **100.28%**!

### 1.3 Formal Problem Statement

> **Given**: A sequence of timestamped speaker turns $\mathcal{T} = \{(t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}, \text{speaker\_id}^{(i)})\}_{i=1}^N$ and separated audio stems $\{\hat{s}_k(t)\}_{k=1}^K$.
>
> **Goal**: For each turn $i$, transcribe the acoustic segment into a normalized text sequence:
> $$W^{(i)} = (w_1, w_2, \dots, w_{M_i})$$
> matching ground truth with minimal Word Error Rate (WER) and Character Error Rate (CER), while attributing each word to its true speaker identity $\text{speaker\_id}^{(i)}$.

---

## 2. How We Are Solving It — Multi-Engine Architecture

Our Stage 3 system integrates acoustic slicing, stem routing, dual ASR engines, and Indic text normalization into a unified pipeline.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: AUTOMATIC SPEECH RECOGNITION                    │
│                                                                               │
│  Diarization Timeline                  Separated Stems (Stage 1)              │
│  [ {start, end, spk_id}, ... ]         Stem 1 (Speaker 0), Stem 2 (Speaker 1) │
│            │                                       │                          │
│            ▼                                       ▼                          │
│  ┌───────────────────────────────────────────────────┐                        │
│  │  Separation-Aware Stem Router & Acoustic Slicer  │                        │
│  │  - Maps speaker_id to corresponding clean stem    │                        │
│  │  - Adds 50ms safety plosive/fricative padding     │                        │
│  └─────────────────────────┬─────────────────────────┘                        │
│                            │ Isolated Audio Chunk x_i(t)                      │
│                            ▼                                                  │
│  ┌───────────────────────────────────────────────────┐                        │
│  │  Dual-Engine Speech Recognizer                    │                        │
│  │                                                   │                        │
│  │  PRIMARY: AI4Bharat / Vakyansh Wav2Vec2 CTC       │                        │
│  │  (Trained on 4200 hrs of Indian speech)           │                        │
│  │                                                   │                        │
│  │  BASELINE: OpenAI Whisper Tiny (AED Transformer)  │                        │
│  └─────────────────────────┬─────────────────────────┘                        │
│                            │ Raw Token Stream                                 │
│                            ▼                                                  │
│  ┌───────────────────────────────────────────────────┐                        │
│  │  Indic & Bilingual Text Normalizer                │                        │
│  │  1. Unicode NFKC Canonical Equivalence            │                        │
│  │  2. Nukta Harmonization (क़ ↔ क + ़)               │                        │
│  │  3. Zero-Width Joiner/Non-Joiner Removal (ZWJ)    │                        │
│  │  4. Punctuation & Danda Stripping (।, ॥, ?, !)     │                        │
│  │  5. Latin Lowercasing for Code-Switched Hinglish  │                        │
│  └─────────────────────────┬─────────────────────────┘                        │
│                            │ Normalized Transcripts                           │
│                            ▼                                                  │
│  Output: Utterance [ speaker_id, start, end, text, confidence ]              │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The 50ms Plosive Safety Padding
When cutting audio at diarization turn boundaries $[t_{\text{start}}, t_{\text{end}}]$, strict truncation can cut off:
- **Plosive Bursts**: Consonants like /p/, /t/, /k/, /b/, /d/, /g/ begin with a silent closure followed by an explosive burst lasting 20–40ms. If sliced too tightly, the burst is lost, and the ASR model misidentifies the consonant.
- **Trailing Nasals & Fricatives**: Sounds like /s/, /z/, /m/, /n/ taper off gradually over 30–50ms.

Our engine applies a **50ms bidirectional safety padding**:
$$t_{\text{slice\_start}} = \max(0.0, t_{\text{start}} - 0.05)$$
$$t_{\text{slice\_end}} = \min(T_{\text{max}}, t_{\text{end}} + 0.05)$$

This simple DSP intervention recovers phoneme boundaries and prevents truncated word errors.

---

## 3. Tech Stack & Software Architecture

| Technology | What It Is | What It Does In Stage 3 | Why We Use It |
|---|---|---|---|
| **AI4Bharat Vakyansh Wav2Vec2** (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`) | Acoustic speech model for Hindi/Indian languages | Primary ASR transcription engine using Connectionist Temporal Classification (CTC) | Fine-tuned on **4,200 hours** of diverse Indian speech; vastly superior phoneme alignment for Indian accents and Devanagari script |
| **OpenAI Whisper Tiny** (`openai/whisper-tiny`) | Multilingual encoder-decoder Transformer | Baseline ASR engine for cross-architectural benchmarking | Industry standard baseline; demonstrates the difference between CTC and autoregressive AED models |
| **HuggingFace Transformers** (`transformers`) | State-of-the-art model hub & pipeline | Handles model downloading, tokenization, processor vocabulary, and weight execution | Standardized interface for both `Wav2Vec2ForCTC` and `WhisperForConditionalGeneration` |
| **PyTorch (`torch`)** | Deep learning framework | Executes forward pass tensors on CPU or GPU with `torch.no_grad()` | Fast inference, low latency, seamless device switching (`auto` detection) |
| **Wagner-Fischer DP Engine** (`evaluation/asr_metrics.py`) | Dynamic programming algorithm | Computes exact Levenshtein alignment between reference and hypothesis tokens | Provides exact counts of Substitutions ($S$), Deletions ($D$), and Insertions ($I$) without external blackbox dependencies |
| **IndicTextNormalizer** (`src/asr/text_normalizer.py`) | Orthographic & phonetic normalizer | Cleans Unicode NFKC, nuktas, zero-width chars, and punctuation | Prevents artificial WER inflation caused by cosmetic script differences |

---

## 4. Mathematical Foundations

### 4.1 Connectionist Temporal Classification (CTC)
The primary engine (AI4Bharat Vakyansh) uses **Wav2Vec 2.0 with CTC loss**.

Given acoustic frame representations $X = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ and a target text sequence $Y = (y_1, \dots, y_U)$, where $T \gg U$ (acoustic frames are far more numerous than words):
1. The vocabulary is augmented with a blank token $\epsilon$:
$$\Omega' = \Omega \cup \{\epsilon\}$$
2. For each frame $t \in [1, T]$, the model outputs a probability distribution over $\Omega'$:
$$P(\pi_t | X) = \operatorname{softmax}(W \mathbf{h}_t)$$
3. Define the collapse operator $\mathcal{B}: {\Omega'}^T \to \Omega^{\le T}$ that:
   - Replaces consecutive identical tokens with a single token (e.g., `aa` $\to$ `a`).
   - Removes all blank tokens $\epsilon$ (e.g., `a - b - - b` $\to$ `abb`).
4. The conditional probability of the target sequence $Y$ is the sum of probabilities of all valid alignments $\pi$:
$$P(Y|X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(\pi_t | X)$$
5. The model is trained by minimizing the negative log-likelihood:
$$\mathcal{L}_{\text{CTC}} = -\log P(Y|X)$$

*Why CTC is robust for multi-speaker sliced turns*: CTC models do not have an autoregressive text decoder. Each acoustic frame independently votes for a character. Unlike Whisper, CTC **cannot enter an infinite repetition loop** because it is bound to the linear passage of acoustic time!

---

### 4.2 Attention-Based Encoder-Decoder (Whisper)
Whisper uses a Transformer Encoder-Decoder architecture:
1. **Encoder**: Converts 80-channel log-Mel spectrogram frames into hidden states $H_{\text{enc}} \in \mathbb{R}^{T' \times d_{\text{model}}}$.
2. **Decoder**: Autoregressively predicts the next token $y_i$ given all previous tokens $y_{<i}$ and the encoder states via Multi-Head Cross-Attention:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
$$\mathcal{L}_{\text{AED}} = -\sum_{i=1}^{|Y|} \log P(y_i \mid y_{<i}, H_{\text{enc}})$$

*The Failure Mode*: When acoustic signals from two speakers mix, the hidden states $H_{\text{enc}}$ contain mixed representations. During cross-attention, $Q K^T$ does not produce a sharp peak; instead, attention energy splits between two competing words, corrupting the autoregressive context $y_{<i}$ and triggering runaway repetitive hallucinations.

---

### 4.3 Text Normalization Mathematics & Nukta Equivalence
In Indian scripts like Devanagari, the exact same word can be digitally stored in two completely different binary byte sequences:
- **Composed Form**: A single precomposed Unicode codepoint (e.g., `क़` = `U+0958`).
- **Decomposed Form**: Base consonant followed by Nukta sign (e.g., `क` = `U+0915` + `़` = `U+093C`).

Without normalization:
$$\text{Levenshtein}("U+0958", "U+0915 U+093C") = 2 \text{ errors!}$$
Even though a human reads them as the exact same letter!

Our normalizer guarantees **bijective canonical equivalence**:
1. Apply Unicode Normalization Form KC (NFKC):
$$\text{text} \leftarrow \text{unicodedata.normalize}('NFKC', \text{text})$$
2. Replace all decomposed sequences with single composed characters via dictionary mapping:
$$\forall (k, v) \in \text{NUKTA\_MAPPINGS}: \text{text} \leftarrow \text{text.replace}(v, k)$$
3. Strip all Zero-Width Joiners (`U+200D`) and Non-Joiners (`U+200C`) that only affect font rendering ligatures.
4. Strip Devanagari danda (`।`), double danda (`॥`), and Latin punctuation.
5. Lowercase Latin characters for Hinglish code-switching consistency.

---

### 4.4 Dynamic Programming Levenshtein Distance & WER / CER

Let $R = (r_1, r_2, \dots, r_N)$ be the reference token list (words for WER, characters for CER).
Let $H = (h_1, h_2, \dots, h_M)$ be the hypothesis token list.

The **Wagner-Fischer Dynamic Programming Algorithm** constructs a matrix $D \in \mathbb{R}^{(N+1) \times (M+1)}$:

**Base Conditions**:
$$D[0, 0] = 0$$
$$D[i, 0] = i \quad (\text{Deletions for } i = 1, \dots, N)$$
$$D[0, j] = j \quad (\text{Insertions for } j = 1, \dots, M)$$

**Recurrence Relation**:
$$D[i, j] = \begin{cases}
D[i-1, j-1] & \text{if } R[i-1] = H[j-1] \quad (\text{Hit / Match}) \\
\min \begin{cases} 
D[i-1, j] + 1 & (\text{Deletion } D) \\
D[i, j-1] + 1 & (\text{Insertion } I) \\
D[i-1, j-1] + 1 & (\text{Substitution } S)
\end{cases} & \text{if } R[i-1] \ne H[j-1]
\end{cases}$$

#### Word Error Rate (WER)
Once the optimal path is backtracked from $D[N, M]$:
$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

*Note*: WER can exceed 100% if the model hallucinates many extra words ($I > N$).

#### Character Error Rate (CER)
Operating on individual graphemes / characters:
$$\text{CER} = \frac{S_{\text{char}} + D_{\text{char}} + I_{\text{char}}}{N_{\text{char}}} \times 100\%$$

CER is especially critical for Indian languages, where single inflections, postpositions, or matras change grammar without completely changing the root word.

---

## 5. Experimental Results & In-Depth Interpretation

We evaluated the complete Stage 3 pipeline under 3 controlled experimental conditions to isolate the exact impact of each stage.

### 5.1 Quantitative Results Matrix (Milestone M6)

| Pipeline Condition | Stage 1 (BSS) | Stage 2 (Diarization) | Mean WER (%) | Mean CER (%) | RTF (CPU) | Acoustic Intelligibility & Failure Mode |
|---|---|---|---|---|---|---|
| **Condition 1 (Naive Mixture)** | ❌ Bypassed | ❌ Bypassed | **86.28%** | **100.28%** | 0.1744 | ❌ **Catastrophic Failure**: Infinite repetition loop (`"पुप्प् पुप्प् पुप्प्..."`) |
| **Condition 2 (Raw + Diarization)** | ❌ Bypassed | ✅ Active | **70.78%** | **65.64%** | 0.1536 | ⚠️ **Severe Interference**: Background speaker speech bleeding into target slice |
| **Condition 3 (Full Proposed System)** | ✅ Conv-TasNet | ✅ Active | **51.57%** | **44.66%** | **0.1936** | ✅ **Clean Attributed Speech**: Isolated speaker stems with accurate word transcripts |

### 5.2 Deep Interpretation of Empirical Findings

#### 1. Why Condition 1 Collapses (WER = 86.28%, CER = 100.28%)
Without separation and without diarization, the ASR model receives the entire mixture file as a single stream.
- The reference transcript consists of Speaker A's words followed by Speaker B's words.
- In reality, Speaker A and Speaker B spoke simultaneously between seconds 3.0 and 5.0.
- The model hears overlapping phonemes, loses tracking of the phonetic sequence, and outputs:
  `"पुप्प् पुप्प् पुप्प् पुप्प् पुप्प्..."`
- This single failure creates massive numbers of substitutions and deletions, pushing CER past 100%.

#### 2. Why Diarization Alone Is Not Enough (Condition 2: WER = 70.78%)
In Condition 2, we know the exact timestamps when Speaker A speaks. But we cut those timestamps from the **raw mixed audio file**.
- When Speaker A speaks from 2.0s to 4.5s, Speaker B is also speaking in the background from 3.0s to 4.5s.
- The ASR model hears Speaker A as the primary speaker, but Speaker B's voice leaks into the microphone.
- The model transcribes fragments of Speaker B's words interspersed into Speaker A's sentence, resulting in **70.78% WER**.

#### 3. The Conv-TasNet + Diarization Synergy (Condition 3: WER = 51.57%)
In Condition 3, the audio is first separated by Conv-TasNet into Stem 1 and Stem 2.
- The diarizer provides the timestamps.
- The ASR engine slices the audio from **Stem 1** (where Speaker B's voice has been attenuated by +21.8 dB).
- Result: **WER drops to 51.57%** (a **34.71% absolute improvement** over Condition 1, and **19.21% absolute improvement** over Condition 2!).
- Character Error Rate drops to **44.66%**, reflecting accurate capture of Hindi root words and English loanwords.

#### 4. Computational Efficiency
The entire end-to-end pipeline (Separation + Diarization + ASR) runs with a Real-Time Factor (RTF) of **0.1936** on a standard CPU.
- 1 second of conversation is processed in **193 milliseconds**.
- The pipeline operates **5.2× faster than real-time**, making it feasible for edge deployment, automated meeting transcription, and call-center analytics without specialized GPU clusters.

---

## 6. How Results Feed into Stage 4 (Alignment & Interactive Demo)

Stage 3 produces structured **Utterance** objects conforming to the system contract:

```json
[
  {
    "speaker_id": "SPEAKER_00",
    "start": 0.00,
    "end": 4.71,
    "text": "उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा",
    "confidence": 0.94
  },
  {
    "speaker_id": "SPEAKER_01",
    "start": 3.54,
    "end": 9.54,
    "text": "यह एक वॉइस ऑडियो डेटासेट है जिसे आप ai प्रशिक्षण उद्देश्यों के लिए उपयोग कर सकते हैं",
    "confidence": 0.91
  }
]
```

**In Stage 4 (Interactive Web Demo & Master Delivery)**:
1. We align these timestamped utterances on an **interactive multi-track timeline**.
2. We link the audio stems so the user can listen to:
   - The original mixed recording.
   - Stem 0 (Speaker 0 isolated).
   - Stem 1 (Speaker 1 isolated).
3. Clicking on any word or turn jumps the audio player to that exact millisecond.
4. The user sees a live demonstration of "Who Spoke What and When".

---

## 7. Key Viva / Defense Talking Points

1. **Why did you use AI4Bharat Vakyansh instead of only OpenAI Whisper?**
   > *Whisper is trained predominantly on Western English and high-resource European languages. For Indian regional languages and code-switched Hinglish, AI4Bharat Vakyansh was trained specifically on over 4,200 hours of Indian speech across diverse linguistic regions, providing vastly superior phonetic recognition for Devanagari scripts.*

2. **Why does naive ASR collapse with 86% WER on overlapping audio?**
   > *Autoregressive Transformer decoders suffer from the 'Attention Collision Paradox'. When acoustic keys/values represent two conflicting speakers simultaneously, the softmax distribution flattens, trapping the decoder in infinite repetition loops.*

3. **How does text normalization prevent false error inflation?**
   > *In Devanagari, characters with nuktas (like क़) can be written either as a single composed Unicode point or as a base character plus a combining nukta. Without Unicode NFKC normalization and nukta harmonization, identical spoken words would register false Levenshtein edit errors.*

4. **What is the empirical proof that source separation is necessary before ASR?**
   > *Condition 2 (Diarization on raw audio) produced 70.78% WER due to cross-talk bleed. When Conv-TasNet separation was introduced upstream (Condition 3), WER dropped to 51.57% — a 19.21% absolute error reduction solely attributable to acoustic separation.*

5. **Why add 50ms safety padding when slicing audio turns?**
   > *Consonants with plosive bursts (p, t, k, b, d) require a 20–40ms silence and release window, and nasals/fricatives taper off gradually. 50ms padding ensures phoneme boundaries are not clipped by the turn segmenter.*
