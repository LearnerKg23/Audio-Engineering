# 06 — Stage 3: Regional & Code-Switched Automatic Speech Recognition (ASR)

## 1. Core Objective & Indian Linguistic Context

The core objective of Stage 3 is to translate speaker-diarized acoustic audio segments into high-fidelity textual transcripts while faithfully preserving:
1. **Regional Indian Dialects**: Native vocabulary, grammatical markers, and phonetic inflections (e.g., Hindi, Bengali, Marathi, etc.).
2. **Dynamic Code-Switching**: Intra-sentential alternating between an Indian vernacular language and English (e.g., Hinglish: *"आज market में गेहूं का price बहुत low है"*).
3. **Phonetic Output Fidelity**: Accurately mapping acoustic tokens into native scripts (Devanagari, Bengali, etc.) or standardized Latin transliterations without silent word omissions.

### Why Standard Vanilla ASR Fails on Indian Speech
Standard Western-centric ASR architectures (e.g., vanilla OpenAI Whisper) fail on conversational Indian speech for specific structural reasons:
* **Monolingual Decoding Bias**: Vanilla models often force a single language token at the beginning of an audio segment (e.g., `<|hi|>` or `<|en|>`), forcing the decoder into an unnatural monolingual language model prior.
* **Code-Switching Hallucinations**: When encountering rapid language swaps mid-sentence, models frequently drop code-switched English words or hallucinate phonetically similar Hindi words (or vice versa).
* **Acoustic Noise Brittleness**: Standard models trained on clean studio or podcast audio degrade sharply when acoustic SNR drops below $+15\text{ dB}$.

---

## 2. Model Selection: Localized Indian Models vs. Baselines

The proposal explicitly mandates deploying localized, fine-tuned speech models: **IndicASR** or **IndicWav2Vec** (developed by AI4Bharat).

| Dimension | AI4Bharat IndicWav2Vec / IndicASR | Vanilla Whisper Baseline (`whisper-base` / `whisper-small`) | Engineering Strategy |
|---|---|---|---|
| **Architecture** | Pre-trained self-supervised Wav2Vec 2.0 / Conformer backbone fine-tuned on 22 Indian languages with CTC/transducer decoding. | Encoder-Decoder Transformer with autoregressive text generation. | **Primary Engine: IndicWav2Vec / IndicASR**.<br>Leverages specialized Indian acoustic representations. |
| **Code-Switching Handling** | Trained explicitly on spontaneous bilingual Indian datasets (including IndicVoices and Nirantar). | Prone to severe language flipping, repetition loops, or English phonetic transliteration errors. | We establish vanilla Whisper as the comparative baseline to prove the quantitative necessity of localized models. |
| **Script Handling** | Directly outputs native Indic scripts (Devanagari, Bengali, etc.) with support for Latin code-switched tokens. | Mixed script output; inconsistent casing and tokenization on Indic orthography. | Implement an Indic Text Normalization layer (Unicode NFKC, Devanagari character collation). |
| **Compute / Footprint** | Lightweight CTC decoder; fast inference on CPU and Apple Silicon / CUDA GPUs. | Autoregressive beam search creates higher compute latency and memory overhead. | IndicWav2Vec provides lower Real-Time Factor (RTF). |

---

## 3. Evaluation Metrics & Indic Text Normalization

### 1. Word Error Rate (WER) & Character Error Rate (CER)
$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Reference Words}}$$
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
In Indian scripts with complex conjunct consonants (matras / halants), CER is often a more reliable indicator of phonetic accuracy than WER, because a minor matra difference can flag an entire word as an error in WER.

### 2. Indic-Specific Text Normalization
To prevent false WER inflation from orthographic variations:
* **Unicode Normalization (NFKC)**: Resolves ambiguous Devanagari codepoint representations (e.g., nukta representation variants like क़ vs क + ़).
* **Punctuation and Whitespace Stripping**: Normalizes punctuation before computing Levenshtein edit distance.
* **Code-Switched Transliteration Alignment**: For bilingual tokens, compares phonetic matches using Soundex or standardized transliteration where ground truth includes Latin equivalents.

---

## 4. Pipeline Integration: Diarization Slicing to Transcript

```
Diarization Timeline JSON:
  [Segment 0: 00:00 - 00:06, SPEAKER_00]
  [Segment 1: 00:07 - 00:14, SPEAKER_01]
             ↓
Audio Slicer (with 50ms safety margin to avoid cutting plosives)
             ↓
Batch Ingestion to Indic ASR Engine
             ↓
Raw Speaker-Attributed Transcript JSON:
  [00:00 - 00:06] SPEAKER_00: "नमस्कार किसान भाइयों आज हम गेहूं के mandi rate की चर्चा करेंगे"
  [00:07 - 00:14] SPEAKER_01: "हां भाई साहब इस बार MSP से ऊपर price मिल रहा है..."
```

---

## 5. Experimental Ablation: Impact of Separation on ASR Accuracy

We evaluate ASR under three explicit pipeline conditions:
1. **Raw Audio directly to ASR** (without Separation or Diarization): ASR decodes mixed talkers simultaneously, creating chaotic word interleaving.
2. **Raw Audio + Diarization to ASR** (no Source Separation): ASR receives overlapping segments where secondary speaker babble causes severe insertions ($I$) and substitutions ($S$).
3. **Full Pipeline: Separation + Diarization + ASR**: ASR processes isolated speaker stems, significantly reducing acoustic interference.

---

## 6. Definition of Done (DoD) for Stage 3

- [x] IndicWav2Vec / IndicASR model loads and runs inference on standardized 16 kHz audio chunks.
- [x] Baseline comparator (vanilla Whisper) is implemented behind the same `BaseASR` interface.
- [x] Text normalizer resolves Indic Unicode codepoint variations and Devanagari matras.
- [x] WER and CER calculation pipeline is verified on ground-truth transcripts from IndicVoices / Nirantar.
- [x] ASR output strictly conforms to Contract 4 (`raw_transcript.json`), mapping speaker IDs and timestamps to text.
- [x] Comparative benchmark between IndicASR and Whisper baseline is executed on code-switched audio.
- [x] Processing time, RTF, RAM, and GPU memory metrics are documented.
