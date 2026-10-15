# 📚 Project Overview — "Who Spoke What and When"

## End-to-End Pipeline for Multi-Speaker Indian Conversational Audio

---

## 1. What Is This Project?

This project builds a **complete speech engineering pipeline** that takes a **raw audio recording** of multiple Indian speakers talking (often overlapping each other) and produces a **structured, speaker-attributed transcript** — telling you exactly **who** said **what** and **when**.

### The Three Fundamental Questions

| Question | Stage That Answers It | Output |
|---|---|---|
| **WHO** is speaking? | Stage 2 — Speaker Diarization | `SPEAKER_00`, `SPEAKER_01` |
| **WHEN** are they speaking? | Stage 2 — Speaker Diarization | `[00:00.00 → 04.71s]` |
| **WHAT** did they say? | Stage 3 — Automatic Speech Recognition | `"उन्हें दिन तक रीहाब करना होगा"` |

But there is a critical prerequisite: **when speakers talk at the same time (overlap)**, neither diarization nor ASR can work properly on the raw mixed audio. This is why we need:

| Problem | Stage That Solves It |
|---|---|
| Two voices mixed in one audio stream | Stage 1 — Blind Source Separation |
| Combined transcript needs insights | Stage 4 — LLM Intelligence |

---

## 2. The 4-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAW MIXED AUDIO INPUT                           │
│     (Two or more Indian speakers talking, possibly overlapping)        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: BLIND SOURCE SEPARATION (BSS)                                 │
│  Tech: Conv-TasNet (PyTorch / Torchaudio)                               │
│  Input:  Mixed audio waveform                                           │
│  Output: Isolated single-speaker audio stems (Contract 2)               │
│  Purpose: Physically decouple overlapping voices into clean tracks      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SPEAKER DIARIZATION                                           │
│  Tech: Mel Filterbank + Spectral Clustering + Eigengap Heuristic        │
│  Input:  Separated audio stems from Stage 1                             │
│  Output: DiarizationTimeline with speaker IDs + timestamps (Contract 3) │
│  Purpose: Answer "WHO spoke WHEN" — assign speaker labels & timestamps  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: AUTOMATIC SPEECH RECOGNITION (ASR)                            │
│  Tech: AI4Bharat IndicWav2Vec (Wav2Vec2 CTC) via HuggingFace           │
│  Input:  Diarized speaker segments + separated stems                    │
│  Output: Speaker-attributed Hindi transcript (Contract 4)               │
│  Purpose: Answer "WHAT was said" — convert speech to Devanagari text    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: LLM INTELLIGENCE (Downstream)                                 │
│  Tech: Groq / OpenAI LLM APIs                                          │
│  Input:  Structured speaker-attributed transcript                       │
│  Output: Meeting summary, action items, sentiment, topic tags           │
│  Purpose: Extract actionable intelligence from conversation             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Contracts (Inter-Stage Communication)

Each stage produces a strictly typed output (using **Pydantic** data models) that the next stage consumes:

| Contract | Pydantic Model | Stage Producer | Stage Consumer |
|---|---|---|---|
| Contract 1 | `AudioMetadata` | Preprocessing | Stage 1 |
| Contract 2 | `SeparationResult` | Stage 1 (BSS) | Stage 2 |
| Contract 3 | `DiarizationTimeline` | Stage 2 (Diarization) | Stage 3 |
| Contract 4 | `RawTranscript` | Stage 3 (ASR) | Stage 4 |
| Contract 5 | `PostProcessedReport` | Stage 4 (LLM) | End User |

---

## 4. Tech Stack Summary

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.12+ | Primary language |
| **PyTorch** | ≥ 2.0 | Deep learning framework |
| **Torchaudio** | ≥ 2.0 | Audio processing & pretrained models |
| **SciPy** | ≥ 1.10 | Signal processing, resampling, STFT |
| **NumPy** | ≥ 1.24 | Numerical computation |
| **Demucs** | ≥ 4.0 | Meta's music/speech separation (baseline) |
| **HuggingFace Transformers** | ≥ 4.30 | AI4Bharat IndicWav2Vec & Whisper models |
| **scikit-learn** | ≥ 1.3 | KMeans, GMM clustering |
| **Pydantic** | ≥ 2.0 | Data validation & typed contracts |
| **Pytest** | ≥ 7.0 | Test framework (36 tests) |

---

## 5. Learning Path

Read the documents in this folder in order:

| Document | Stage / Scope | What You Will Learn |
|---|---|---|
| [`01_STAGE_1_SEPARATION.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/01_STAGE_1_SEPARATION.md) | Stage 1 | Blind Source Separation (BSS), Cocktail Party Problem, Conv-TasNet vs Demucs, SI-SDR, PIT |
| [`02_STAGE_2_DIARIZATION.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/02_STAGE_2_DIARIZATION.md) | Stage 2 | Speaker Diarization ("Who Spoke When"), 192-dim Embeddings, Spectral Clustering, Eigengap Heuristic, NIST DER & JER |
| [`03_STAGE_3_ASR.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/03_STAGE_3_ASR.md) | Stage 3 | Regional & Hinglish ASR ("What Was Said"), CTC vs AED, AI4Bharat Vakyansh vs Whisper, Indic Normalization, WER/CER |
| [`04_STAGE_4_SEMANTIC_POST_PROCESSING.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/04_STAGE_4_SEMANTIC_POST_PROCESSING.md) | Stage 4 | Semantic Intelligence & LLM Guardrails, Disfluency Cleaning, Contract 5 Validation, ROUGE, BERTScore |
| [`05_MASTER_EVALUATION_AND_METRICS.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/05_MASTER_EVALUATION_AND_METRICS.md) | Full Pipeline | Unified Mathematical Formulation for all Metrics, Cross-Stage Matrix, Master Viva Cheat Sheet |
| [`06_DEMO_AND_DEPLOYMENT_GUIDE.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/06_DEMO_AND_DEPLOYMENT_GUIDE.md) | Web Demo & Deployment | `demo.html` Web Audio API Architecture, GitHub Pages Deployment, Git Push Workflow, 2-Min Reviewer Walkthrough |
| [`07_LIVE_DEMO_SPEAKING_SCRIPT.md`](file:///c:/Users/Kushal%20Gupta/Desktop/AudioEngineering/Audio-Engineering/LEARNING/07_LIVE_DEMO_SPEAKING_SCRIPT.md) | Presentation Script | Turn-by-Turn Word-for-Word Speaking Guide, Stage 1 & 2 Live Audition Steps, Evaluator Q&A Cheat Sheet |

---

## 6. The Final Output

After all 4 stages, the system produces a structured JSON transcript like this:

```json
[
  {
    "speaker": "SPEAKER_00",
    "start_time": "00:00.00",
    "end_time": "00:04.71",
    "text": "उन्हें दिन तक रीहाब करना होगा और उसके बाद उनका फिटनेस टेस लिया जाएगा"
  },
  {
    "speaker": "SPEAKER_01",
    "start_time": "00:03.54",
    "end_time": "00:09.54",
    "text": "ये एक वॉइस ऑडियो डेटासेट है जिसका उपयोग आप आर्टिफिशियल इंटेलिजेंस एआई के प्रशिक्षण के"
  }
]
```

Even though both speakers were talking between **3.54s – 4.71s** (overlap), the pipeline correctly:
1. ✅ Separated their voices (Stage 1)
2. ✅ Identified who spoke when (Stage 2)
3. ✅ Transcribed each speaker's Hindi speech individually (Stage 3)

---

**Team**: Kushal Gupta • Vansh • Dorothy • Pranav
