# 11 — Sequential Implementation Roadmap: Two-Phase Execution

## 1. Phased Execution Architecture

The implementation plan is structured around two distinct operational phases:
1. **PHASE A — FIRST EVALUATION (M0 $\rightarrow$ M3)**: The immediate priority. Implements the controlled data pipeline, pre-trained Demucs source separation, PIT SI-SDR evaluation, audio demo, and flash talk presentation package.
2. **PHASE B — POST-EVALUATION / FULL EXECUTION (M4 $\rightarrow$ M12)**: Downstream diarization, regional ASR, LLM guardrails, end-to-end integration, error propagation analysis, benchmarking notebook, and final portfolio defense.

```mermaid
gantt
    title Two-Phase Project Roadmap
    dateFormat  X
    axisFormat  W%W
    
    section PHASE A: FIRST EVALUATION
    M0: Environment & Core Scaffolding            :done, m0, 0, 1
    M1: Controlled Data Pipeline                  :active, m1, 1, 2
    M2: Stage 1 Demucs Separation & SI-SDR        :m2, 2, 3
    M3: First Evaluation Package & Flash Talk     :m3, 3, 4
    
    section EVALUATION GATE
    ★ PROFESSOR'S FIRST EVALUATION (3-4 min) ★   :milestone, gate, 4, 4
    
    section PHASE B: FULL PROJECT EXECUTION
    M4: Full Data Pipeline & Corpus Expansion     :m4, 4, 5
    M5: Stage 2 Speaker Diarization (pyannote)    :m5, 5, 6
    M6: Stage 3 Regional Hinglish ASR             :m6, 6, 7
    M7: Stage 4 LLM Semantic Post-Processing      :m7, 7, 8
    M8: End-to-End System Integration & CLI       :m8, 8, 9
    M9: Error Propagation & Scientific Ablations  :m9, 9, 10
    M10: System Benchmark Notebook (RTF, VRAM)    :m10, 10, 11
    M11: Real-World Indian Audio Validation       :m11, 11, 12
    M12: Final Deliverables & Portfolio Packaging :m12, 12, 13
```

---

## 2. Phase A: First Evaluation Tasks & Deliverables

### Milestone M0: Project Scope, Data & Evaluation Design
* **Status**: **COMPLETED & VERIFIED** (See [`outputs/reports/m0_environment_report.md`](file:///Users/vanshsharma/Documents/AI%20Project/outputs/reports/m0_environment_report.md)).
* **Accomplished**:
  - Python 3.12.7 runtime with PyTorch 2.9.0 and Apple Silicon Metal (`mps:0`) acceleration verified.
  - SoundFile + CFFI `libsndfile` audio engine operational.
  - BaseModule abstract class, strongly-typed Pydantic state models (`src/core/state.py`), and audio I/O utilities (`src/core/audio_io.py`) implemented.
  - 9/9 unit tests passing with zero warnings.

---

### Milestone M1: Controlled Data Pipeline
* **Objective**: Build the controlled dataset and mixture synthesizer required for the Stage 1 demonstration.
* **Tasks**:
  1. Curate 4–6 clean single-speaker slices from `IndicVoices` and `Nirantar` (Hindi/Hinglish).
  2. Implement `src/preprocessing/mixture_generator.py` synthesizing overlapping mixtures:
     - Configurable overlap levels: 25% and 50%.
     - Configurable relative volumes: $\text{SIR} = 0\text{ dB}$ (equal volume) and $+6\text{ dB}$ (dominant talker).
  3. Generate `data/manifests/controlled_ground_truth.json` mapping mixtures to clean reference stems.
  4. Verify audible acoustic separation in ground-truth source files.
* **Checkpoint**: Controlled mixture WAV files and ground-truth manifests exist and are reproducible.

---

### Milestone M2: Stage 1 Source Separation Prototype
* **Objective**: Implement pre-trained Demucs (`htdemucs`), run inference on controlled mixtures, and calculate quantitative SI-SDR with PIT matching.
* **Tasks**:
  1. Implement `DemucsSeparator` in `src/separation/demucs_wrapper.py`.
  2. Implement `evaluation/bss_metrics.py` calculating Scale-Invariant Signal-to-Distortion Ratio (**SI-SDR**) and **Permutation Invariant Training (PIT)** optimal assignment.
  3. Execute separation inference on the controlled mixtures from M1.
  4. Save isolated single-speaker tracks in `outputs/first_eval_demo/`.
  5. Measure wall-clock latency, processing time, and Real-Time Factor (RTF).
* **Checkpoint**: At least one quantitative SI-SDR result (+X dB improvement) and before/after audio files are generated.

---

### Milestone M3: First Evaluation Package & Flash Talk
* **Objective**: Package deliverables into a 3–4 minute flash talk and 1–2 slide presentation for the professor.
* **Tasks**:
  1. Author **Slide 1** (Problem Statement, Multi-speaker Indian speech challenges, Hinglish focus, System Pipeline diagram).
  2. Author **Slide 2** (Selected Corpora, Selected Models, SI-SDR evaluation methodology, Stage 1 Demucs result, audio demo link, Phase B roadmap).
  3. Author `docs/flash_talk_script.md` containing a 3–4 minute timed presenter script.
  4. Prepare self-contained 1-minute audio demo: Raw Mixture $\rightarrow$ Separated Speaker A $\rightarrow$ Separated Speaker B.
  5. Audit and verify all items in [`project_milestones/FIRST_EVALUATION_CHECKLIST.md`](file:///Users/vanshsharma/Documents/AI%20Project/project_milestones/FIRST_EVALUATION_CHECKLIST.md).
* **Checkpoint**: Team is 100% prepared to deliver the First Evaluation talk and demonstration.

---

## 3. Phase B: Post-Evaluation / Full Project Execution Tasks

*(To be implemented sequentially following the completion of the First Evaluation checkpoint)*

* **Milestone M4: Full Data Pipeline & Corpus Expansion**: Expand to `Project Vaani` and `AIR-RS-DB` long-duration broadcast vignettes.
* **Milestone M5: Stage 2 — Speaker Diarization**: Implement `pyannote.audio` embeddings, Spectral vs. GMM clustering, Eigengap $K$ estimation, and Diarization Error Rate (DER) calculation.
* **Milestone M6: Stage 3 — Regional & Code-Switched ASR**: Implement AI4Bharat `IndicWav2Vec`, Indic Unicode NFKC normalization, Whisper baseline comparator, and WER/CER evaluation.
* **Milestone M7: Stage 4 — Semantic Post-Processing & LLM Guardrails**: Implement modular LLM provider layer (Groq/OpenAI APIs during dev, Airavata local fallback), negative prompt guardrails, and automated factual hallucination auditor.
* **Milestone M8: End-to-End System Integration & CLI Orchestrator**: **COMPLETED & VERIFIED**. Built unified `src/pipeline.py` CLI orchestrator with multi-tier intermediate caching, graceful single-speaker bypass, Contract 1-5 serialization, and 9/9 integration tests passing (55/55 total suite passing).
* **Milestone M9: Cross-Stage Error Propagation & Scientific Ablation Study**: **COMPLETED & VERIFIED**. Executed progressive ablations (Pipelines A, B, C, D), modeled formal error cascading correlations (SI-SDR vs DER r=-0.977, WER vs Hallucination r=+0.978), quantified Critical Breakdown Threshold (+4.2 dB SI-SDR), and validated with 7/7 tests passing (62/62 total test suite passing).
* **Milestone M10: System Benchmarking & Computational Profiling Notebook**: **COMPLETED & VERIFIED**. Implemented `src/benchmarking/profiler.py`, authored official interactive `notebooks/01_system_benchmark.ipynb`, measured isolated latency & RTF across all stages, performed duration scaling sweeps (15s, 30s, 60s, 120s), generated publication plots and `outputs/reports/system_benchmark_report.md` with 6/6 benchmarking tests passing (68/68 total test suite passing).
* **Milestone M11: Real-World Indian Audio Validation**: Evaluate robustness on uncurated, in-the-wild Indian radio debate snippets.
* **Milestone M12: Final Deliverables, Academic Report & Technical Interview Portfolio**: Finalize codebase packaging, academic report, polished `README.md`, and technical interview defense cheat-sheet.
