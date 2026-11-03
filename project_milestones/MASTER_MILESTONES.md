# Master Milestones Specification: Two-Phase Execution Architecture

```
================================================================================
FIRST EVALUATION COMPLETION POINT: MILESTONE 3
================================================================================
At the completion of Milestone 3, the team has satisfied the first evaluation
requirements: data strategy, model selection, evaluation methodology, and a
concrete working demonstration/result.
================================================================================
```

---

## Strategic Phase Split

To align with the professor's formal evaluation requirements while building a world-class AI engineering system, the project execution is explicitly divided into two phases:

* **PHASE A — FIRST EVALUATION (Milestones M0 through M3)**:
  - Focused strictly on satisfying the first academic checkpoint (3–4 minute flash talk, 1–2 slides, concrete working demonstration).
  - Delivers: (1) Data strategy, (2) Model architecture selection, (3) Evaluation methodology with SI-SDR, (4) Concrete working Stage 1 Demucs demonstration, and (5) Presentation slide package.
* **PHASE B — POST-EVALUATION / FULL PROJECT EXECUTION (Milestones M4 through M12)**:
  - Downstream pipeline stages, deep acoustic modeling, cross-stage error analysis, and full system optimization.
  - Delivers: Full multi-speaker diarization, regional Hinglish ASR, guardrailed LLM structuring, end-to-end integration, error cascade study, system benchmark notebook, real-world radio validation, and resume/portfolio packaging.

---

## Roadmap Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PHASE A: FIRST EVALUATION                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  M0: Project Scope, Data & Evaluation Design                                 │
│   ↓                                                                          │
│  M1: Controlled Data Pipeline (Controlled Mixtures + Ground Truth)          │
│   ↓                                                                          │
│  M2: Stage 1 Source Separation Prototype (Demucs + PIT SI-SDR Result)       │
│   ↓                                                                          │
│  M3: First Evaluation Package & Flash Talk (Slides, Script & Live Demo)     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
    ★ PROFESSOR'S FIRST EVALUATION CHECKPOINT (3-4 min presentation) ★
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE B: POST-EVALUATION / FULL PROJECT EXECUTION               │
├──────────────────────────────────────────────────────────────────────────────┤
│  M4: Full Data Pipeline (Corpus Expansion, Dialects, Robustness Slices)      │
│   ↓                                                                          │
│  M5: Speaker Diarization (pyannote + Spectral/GMM + DER Evaluation)          │
│   ↓                                                                          │
│  M6: Regional & Code-Switched ASR (IndicWav2Vec vs. Whisper + WER)           │
│   ↓                                                                          │
│  M7: LLM Semantic Post-Processing (API-Based, Guardrails & Factual Audit)    │
│   ↓                                                                          │
│  M8: End-to-End System Integration & CLI Orchestrator                       │
│   ↓                                                                          │
│  M9: Cross-Stage Error Propagation & Scientific Ablation Study               │
│   ↓                                                                          │
│  M10: System Benchmarking & Computational Profiling Notebook (RTF, VRAM)     │
│   ↓                                                                          │
│  M11: Real-World Indian Audio Validation (AIR Radio & Vaani Field Slices)    │
│   ↓                                                                          │
│  M12: Final Deliverables, Academic Report & Technical Interview Portfolio    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# ==============================================================================
# PHASE A — FIRST EVALUATION (M0 to M3)
# ==============================================================================

## Milestone M0: Project Scope, Data & Evaluation Design

### 1. Objective
Establish exactly what problem we are solving, what data we will use, what models we will test, and how success will be measured, setting up the foundation for the first evaluation presentation.

### 2. Why it Exists
To provide clear, defensible answers to the first three core questions asked by the professor:
1. *What data are we using?*
2. *What models/approaches have we selected?*
3. *How will we evaluate them?*

### 3. Core Problem Definition
The system targets chaotic Indian conversational speech characterized by:
* Multiple concurrent speakers with overlapping speech turns.
* Severe non-stationary ambient noise (market babble, agricultural machinery, street traffic) recorded on uncalibrated, low-cost microphones.
* Dynamic intra-sentential code-switching between regional languages and English.
* **Primary Controlled Linguistic Focus**: **Hindi-English (Hinglish)**. Broader Indian languages (Bengali, Marathi, Tamil) remain architectural extension targets for Phase B.

### 4. Data Strategy Definition
* **Corpus Selection & Utility**:
  - `IndicVoices` (AI4Bharat): Source for clean single-speaker Hinglish/Hindi reference utterances.
  - `Nirantar` (AI4Bharat): Source for natural conversational turns and dialogues.
  - `AIR-RS-DB` (All India Radio): Source for authentic broadcast debate snippets.
  - `Project Vaani` (IISc/ARTPARK): Deferred to Phase B for dialectal stress-testing.
* **Audio Standard**: Uniform 16,000 Hz, 16-bit mono PCM `.wav`.
* **Controlled Mixture Concept**:
  $$\mathbf{x}(t) = \mathbf{s}_A(t) + \alpha \mathbf{s}_B(t - \tau) + \mathbf{n}(t)$$
  Two clean reference stems mixed at calibrated overlap percentages ($\Omega \in \{0\%, 25\%, 50\%, 75\%\}$) and Signal-to-Interference Ratios ($\text{SIR} \in \{0\text{ dB}, \pm 6\text{ dB}\}$).

### 5. Model Architecture Selection
* **Stage 1 (Separation)**: **Meta's Demucs (`htdemucs`)** as primary generative model (hybrid time-frequency U-Net preserving vocal formants); **Conv-TasNet** documented as backup comparison.
* **Stage 2 (Diarization)**: `pyannote.audio` neural embeddings + Spectral Clustering / GMM (Planned for Phase B).
* **Stage 3 (Regional ASR)**: AI4Bharat `IndicWav2Vec` / `IndicASR` (Planned for Phase B).
* **Stage 4 (LLM)**: Modular provider interface (`BaseLLMProvider`) using fast API endpoints (Groq / OpenAI) during development, with local quantized Airavata 7B as a future experiment (Planned for Phase B).

### 6. Evaluation Framework Definition
* **Stage 1 (Immediate Focus)**: **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**:
  $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2} \right)$$
  Evaluated with **Permutation Invariant Training (PIT)** matching to ensure output channel permutations do not register false failures. Higher SI-SDR indicates superior separation.
* **Future Metrics (Phase B)**: Diarization Error Rate (DER) for Stage 2, Word Error Rate (WER) with Indic normalization for Stage 3, Hallucination Rate and ROUGE-L for Stage 4, and Real-Time Factor (RTF) for system compute.

### 7. Definition of Done (DoD)
- [x] Project scope, problem definition, and primary Hinglish focus are formally documented.
- [x] Selected data sources (`IndicVoices`, `Nirantar`, `AIR-RS-DB`) are documented with acoustic rationale.
- [x] Model selection for all 4 stages is finalized and documented.
- [x] Evaluation methodology (SI-SDR, PIT matching, future DER/WER) is defined mathematically.
- [x] Foundation repository structure and environment are operational (M0 verification report completed).

---

## Milestone M1: Controlled Data Pipeline

### 1. Objective
Build a small, clean, and completely reproducible controlled dataset containing single-speaker reference recordings, parameterized multi-speaker mixtures, and verified ground-truth metadata.

### 2. Why it Exists
Real-world radio recordings lack clean isolated speaker stems, making mathematical calculation of SI-SDR impossible on raw broadcasts alone. M1 provides the ground-truth foundation required for the Stage 1 demonstration.

### 3. Inputs & Outputs
* **Inputs**: Clean single-speaker audio slices from `IndicVoices` and `Nirantar` (Hindi/Hinglish).
* **Outputs**:
  - `data/controlled_eval_suite/clean_sources/`: Clean single-speaker reference tracks (`speaker_A.wav`, `speaker_B.wav`, etc.).
  - `data/controlled_eval_suite/synthetic_mixtures/`: Controlled mixture tracks (`mix_01_overlap25.wav`, `mix_02_overlap50.wav`, etc.).
  - `data/manifests/controlled_ground_truth.json`: JSON manifest mapping each mixture to its exact constituent stems, overlap duration, SIR, and speaker IDs.

### 4. Implementation Tasks
* [ ] Curate 4–6 clean single-speaker slices (male and female speakers, Hindi/Hinglish speech).
* [ ] Implement `src/preprocessing/mixture_generator.py` synthesizing overlapping mixtures:
  - Configurable overlap levels: 25% (natural turn boundary interjection) and 50% (conversational overlap).
  - Configurable relative volumes: $\text{SIR} = 0\text{ dB}$ (equal volume) and $+6\text{ dB}$ (dominant talker).
* [ ] Implement peak normalization to prevent digital clipping ($[-1.0, +1.0]$ bounds).
* [ ] Generate and serialize `data/manifests/controlled_ground_truth.json`.
* [ ] Verify that sample mixtures can be played back and listened to cleanly.

### 5. Definition of Done (DoD)
- [ ] At least 3 distinct controlled mixtures exist with known clean ground-truth stems.
- [ ] Ground-truth audio files are saved in uniform 16 kHz 16-bit mono `.wav` format.
- [ ] Metadata manifest (`controlled_ground_truth.json`) links mixtures to exact constituent source files.
- [ ] Controlled mixture generator script is fully reproducible via a single CLI command with fixed random seed.
- [ ] Audio playback verified: clean acoustic separation between speakers is audibly discernible in the ground-truth files.

---

## Milestone M2: Stage 1 Source Separation Prototype

### 1. Objective
Implement the first genuinely working deep learning component of the pipeline: Meta's pre-trained Demucs (`htdemucs`), run inference on controlled mixtures, isolate overlapping talkers into separate audio channels, and compute quantitative SI-SDR metrics with Permutation Invariant Training (PIT) matching.

### 2. Why it Exists
Fulfills the core technical demonstration required for the first evaluation: proving that deep generative models can isolate overlapping conversational speech, and providing the first concrete quantitative result.

### 3. Inputs & Outputs
* **Inputs**: Controlled mixture `.wav` files from M1.
* **Outputs**:
  - Separated single-speaker audio files (`outputs/first_eval_demo/separated_speaker_1.wav`, `separated_speaker_2.wav`).
  - Separation manifest (`separation_manifest.json`) logging energy ratios.
  - Quantitative evaluation report logging measured **SI-SDR (dB)** for each source stem.
  - Computational metrics: inference runtime, audio duration, and Real-Time Factor (RTF).

### 4. Implementation Tasks
* [ ] Implement `DemucsSeparator` in `src/separation/demucs_wrapper.py` wrapping pre-trained `htdemucs`.
* [ ] Implement audio pre-processing and post-processing (resampling to model rate, writing 16 kHz output stems).
* [ ] Implement `evaluation/bss_metrics.py` calculating:
  - Scale-Invariant Signal-to-Distortion Ratio (**SI-SDR**).
  - **Permutation Invariant Training (PIT)** optimal assignment ($\pi^*$).
* [ ] Run separation inference on the controlled mixtures generated in M1.
* [ ] Log inference runtime, processing time, RTF, and output peak amplitude.
* [ ] Verify output audio quality by listening: ensure vocal tracks are isolated without destructive cancellation.

### 5. Required Quantitative & Qualitative Results
* **Quantitative Result**: Calculated SI-SDR score (dB) demonstrating positive separation improvement over the raw mixture.
* **Qualitative Result**: A pair of separated audio files clearly demonstrating vocal separation of the two speakers.

### 6. Definition of Done (DoD)
- [ ] A real mixture `.wav` is ingested and successfully processed by Demucs.
- [ ] Separated single-speaker `.wav` files are generated, saved, and audibly verified.
- [ ] SI-SDR calculation with permutation-invariant matching executes against ground-truth references.
- [ ] At least one concrete quantitative SI-SDR result is logged and documented.
- [ ] Inference wall-clock time, audio duration, and basic memory footprint are recorded.
- [ ] The entire separation and evaluation workflow is 100% reproducible via script.

---

## Milestone M3: First Evaluation Package & Flash Talk

```
================================================================================
★ END OF PHASE A — FIRST EVALUATION COMPLETION POINT ★
================================================================================
```

### 1. Objective
Package the data strategy, model architecture, evaluation methodology, and Stage 1 source separation demo into a cohesive **First Evaluation Deliverable Package** and a **3–4 minute Flash Talk (1–2 slides)**.

### 2. Why it Exists
This milestone directly satisfies the professor's explicit evaluation requirements:
> *“We expect that by that time you will have figured out data, models and how to evaluate them, maybe with a demonstration. These will be taken as a flash talk (1-2 slides or a very short demo). Each team will get 3-4 minutes to give a talk, and we will evaluate your progress.”*

### 3. Deliverables Breakdown

#### Deliverable 1 — DATA (Slide & Talking Points)
* Explain the open Indian corpora selected (`IndicVoices`, `Nirantar`, `AIR-RS-DB`).
* Explain why Hinglish was established as the primary controlled-experiment language.
* Explain the controlled mixture methodology (synthesizing calibrated overlap and SIR).
* Highlight the ground-truth reference stems and metadata manifest.

#### Deliverable 2 — MODELS (Architecture Slide & Roadmap)
* Present the full multi-stage pipeline diagram:
  $$\text{Raw Mixed Audio} \rightarrow \text{Demucs (BSS)} \rightarrow \text{pyannote (Diarization)} \rightarrow \text{IndicWav2Vec (ASR)} \rightarrow \text{LLM Guardrails} \rightarrow \text{Report}$$
* Clearly demarcate:
  - **Currently Implemented**: Stage 1 Generative Source Separation (Demucs).
  - **Planned for Later Milestones**: Stages 2, 3, and 4.

#### Deliverable 3 — EVALUATION (Metrics Table & SI-SDR Explanation)
* Present the SI-SDR metric formulation and explain Permutation Invariant Training (PIT) matching.
* Present the first measured quantitative result:
  - Input Mixture SI-SDR vs. Separated Output SI-SDR (+X dB improvement).
  - Processing runtime and Real-Time Factor (RTF).
* Present the roadmap for future metrics: Diarization DER, ASR WER, and LLM factual hallucination rate.

#### Deliverable 4 — DEMONSTRATION (Audio Snippets / Visual Waveforms)
* A concise 1-minute audio demonstration:
  1. *Mixture*: Play 5–8 seconds of overlapping two-speaker conversational audio.
  2. *Output 1*: Play the isolated Speaker A track.
  3. *Output 2*: Play the isolated Speaker B track.
  4. *Metric*: Display the measured SI-SDR score.

#### Deliverable 5 — FLASH TALK (1–2 Slides & Script)
* Create `docs/flash_talk_slides.md` (and optional PowerPoint/PDF template):
  - **Slide 1: Problem & System Architecture**: The challenge of spontaneous Indian speech, the "Who Spoke What and When" objective, the pipeline diagram, and Hinglish focus.
  - **Slide 2: Current Progress, Evaluation & Demo**: Selected data corpora, Stage 1 Demucs separation, measured SI-SDR result, audio demo link/spectrogram, and Phase B roadmap.
* Create `docs/flash_talk_script.md`: A 3–4 minute timed presenter script with clear section transitions.

### 4. Definition of Done (DoD)
- [ ] Deliverable 1 (Data Strategy) is documented and slide-ready.
- [ ] Deliverable 2 (System Models Architecture) is documented with clear Implemented vs. Planned distinction.
- [ ] Deliverable 3 (Evaluation Methodology + Concrete SI-SDR result) is documented with quantitative figures.
- [ ] Deliverable 4 (Audio demonstration files + spectrogram comparison) is generated and stored in `outputs/first_eval_demo/`.
- [ ] Deliverable 5 (1–2 slide presentation + 3–4 minute timed script) is finalized.
- [ ] All items in [`project_milestones/FIRST_EVALUATION_CHECKLIST.md`](file:///Users/vanshsharma/Documents/AI%20Project/project_milestones/FIRST_EVALUATION_CHECKLIST.md) are checked and verified.

---

# ==============================================================================
# PHASE B — POST-EVALUATION / FULL PROJECT EXECUTION (M4 to M12)
# ==============================================================================

> **NOTE**: The following milestones represent the full engineering execution of the remaining project stages. They are scheduled for implementation immediately following the first evaluation.

---

## Milestone M4: Full Data Pipeline & Corpus Expansion
* **Objective**: Expand data engineering beyond the initial prototype to support the entire pipeline: incorporate `Project Vaani` for dialectal stress-testing, curate multi-speaker radio debate vignettes from `AIR-RS-DB`, and build long-duration test suites.
* **Deliverables**: Extended dataset manifests, multi-dialect audio samples, and automated preprocessing pipeline.

---

## Milestone M5: Stage 2 — Acoustic Speaker Diarization
* **Objective**: Implement unsupervised speaker tracking using `pyannote.audio` neural embeddings, evaluate Spectral Clustering versus Gaussian Mixture Models (GMM), implement automated speaker count ($K$) estimation via the Eigengap heuristic, and calculate Diarization Error Rate (DER) with NIST collar tolerance.
* **Deliverables**: `src/diarization/` module, `diarization_timeline.json`, DER/JER evaluation script, and ablation comparing diarization on raw audio vs. separated stems.

---

## Milestone M6: Stage 3 — Regional & Code-Switched ASR
* **Objective**: Implement acoustic-to-text transcription using AI4Bharat's `IndicWav2Vec` / `IndicASR`, establish vanilla OpenAI Whisper as a baseline comparator, implement Indic Unicode NFKC normalization, and evaluate Word Error Rate (WER) and Character Error Rate (CER) on spontaneous Hinglish audio.
* **Deliverables**: `src/asr/` module, speaker-attributed `raw_transcript.json`, and comparative WER/CER report.

---

## Milestone M7: Stage 4 — Semantic Post-Processing & LLM Guardrails
* **Objective**: Implement the modular LLM provider layer (`BaseLLMProvider` supporting Groq, OpenAI, and local backends), engineer strict prompt guardrails to eliminate hallucinations, clean conversational disfluencies, and output structured executive reports (JSON and Markdown).
* **Deliverables**: `src/llm/` module, Pydantic schema validator, automated factual hallucination auditor, and `final_report.json`.

---

## Milestone M8: End-to-End System Integration & CLI Orchestrator
* **Status**: **COMPLETED & VERIFIED** (See [`outputs/reports/m8_pipeline_integration_report.md`](file:///outputs/reports/m8_pipeline_integration_report.md)).
* **Accomplished**:
  - Unified Python Orchestrator (`src/pipeline.py`) integrating Stages 0, 1, 2, 3, and 4 into a cohesive execution flow.
  - Multi-tier intermediate artifact caching avoiding redundant model inference with sub-second cache hit retrieval.
  - Graceful single-speaker bypass handling (`--bypass-separation`) for clean monologues.
  - Production CLI Interface (`src/cli.py`, `src/__main__.py`) supporting model overrides, hardware options, and telemetry.
  - Strict compliance with inter-stage Pydantic Data Contracts 1 through 5, with automated JSON and Markdown serialization.
  - 9 automated end-to-end integration tests (`tests/test_pipeline_integration.py`) passing with 100% success (55/55 total tests passing).
* **Deliverables**:
  - `src/pipeline.py`: Master `SpeechEngineeringPipeline`.
  - `src/cli.py`: Production Command-Line Interface.
  - `src/__main__.py`: Package entrypoint (`python -m src`).
  - `tests/test_pipeline_integration.py`: End-to-end test suite.
  - `scripts/run_pipeline_demo.py`: Benchmark demonstrator.
  - `outputs/reports/m8_pipeline_integration_report.md`: Formal integration report.

---

## Milestone M9: Cross-Stage Error Propagation & Scientific Ablation Study
* **Objective**: Execute the complete scientific experiment suite: benchmark progressive pipeline configurations (Pipelines A, B, C, D) and conduct the formal Error Cascading Study tracing how separation SI-SDR loss cascades into diarization DER, ASR WER, and LLM hallucination rate.
* **Deliverables**: Progressive ablation comparison table, error cascade correlation curves, failure case taxonomy, and `outputs/reports/error_propagation_study.md`.

---

## Milestone M10: System Benchmarking & Computational Profiling Notebook
* **Objective**: Author the official System Benchmark Notebook (`notebooks/01_system_benchmark.ipynb`) mandated by the proposal, measuring computational latency, processing time, Real-Time Factor (RTF), Peak RAM, and Peak GPU VRAM across all module stages and audio durations (15s, 30s, 60s, 120s).
* **Deliverables**: Fully executable Jupyter notebook with interactive profiling plots and exported benchmark summaries.

---

## Milestone M11: Real-World Indian Audio Validation
* **Objective**: Stress-test the full pipeline on uncurated, in-the-wild Indian recordings (AIR agricultural phone-in programs, rural panel discussions from Vaani) containing severe real-world noise, clipping, and rapid code-switching.
* **Deliverables**: Domain robustness report, real-world case studies, and boundary analysis.

---

## Milestone M12: Final Deliverables, Academic Report & Technical Interview Portfolio
* **Objective**: Finalize all core software deliverables mandated by the proposal, polish the public GitHub repository (`README.md`, diagrams, audio samples), author the comprehensive academic project report, and synthesize interview defense documentation.
* **Deliverables**: Complete production-ready repository, academic submission report, and technical interview talking points.
