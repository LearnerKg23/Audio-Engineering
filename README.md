# Advanced Computational Speech Engineering: "Who Spoke What and When"

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Pydantic](https://img.shields.io/badge/Contracts-Pydantic%20v2-e92063.svg)](https://docs.pydantic.dev/)
[![Status: Complete](https://img.shields.io/badge/Project%20Status-Complete%20%26%20Verified%20(M1--M12)-brightgreen.svg)]()
[![Tests: 127 Passed](https://img.shields.io/badge/Tests-127%2F127%20Passing-success.svg)]()

> **A Modular Multi-Stage Speech Processing Pipeline for Controlled Overlapping Hinglish Speech and Simulated Domain-Stress Evaluation.**

---

## 1. Project Overview & Architectural Vision

In conversational Indian speech—across panel discussions, helplines, and multi-party recordings—off-the-shelf ASR engines struggle under **acoustic speaker overlap**, **rapid Hindi-English (Hinglish) code-switching**, **narrowband telephony filtering (300–3400 Hz)**, and **dynamic turn-taking**.

This repository implements a modular five-stage computational speech engineering pipeline that takes single-channel 16 kHz overlapping audio and outputs structured, time-stamped, speaker-attributed transcripts, Pydantic-validated summary reports with rule-based entity grounding checks, and extractive BM25 conversational search answering:
**"Who Spoke What and When?"**

```
                     ┌────────────────────────────────────────────────────────┐
                     │          RAW MIXED / OVERLAPPING AUDIO WAV            │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 0: PREPROCESSING (Peak Limiting, RMS Normalization, DC Filter, VAD)      │
     └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        │ Single-Speaker Detected?                       │
                       YES                                              NO
                        │                                                │
                        ▼                                                ▼
     ┌──────────────────────────────────────┐     ┌─────────────────────────────────────┐
     │  GRACEFUL BYPASS MODE                │     │  STAGE 1: BLIND SOURCE SEPARATION   │
     │  (Zero Artifact Generation)          │     │  (Conv-TasNet TCN / Hybrid Demucs)  │
     └──────────────────┬───────────────────┘     └──────────────────┬──────────────────┘
                        │                                            │ Isolated Stems
                        └───────────────────────┬────────────────────┘
                                                │
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 2: SPEAKER DIARIZATION (192-d Embeddings + Eigengap Spectral Clustering)  │
     │  Outputs: [Contract 3] DiarizationTimeline JSON with millisecond bounds          │
     └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                │ Sliced Acoustic Segments
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 3: REGIONAL & CODE-SWITCHED ASR (Vakyansh Hindi Wav2Vec2 / Whisper Alt.)  │
     │  Outputs: [Contract 4] RawTranscript JSON with Devanagari & Latin tokens         │
     └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                │ Speaker-Attributed Dialogue
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 4: SEMANTIC POST-PROCESSING & GUARDRAILS (Jinja2 + Swappable LLM Providers)│
     │  Outputs: [Contract 5] PostProcessedReport (JSON Schema + Formal Markdown)       │
     └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Empirical Findings & Project Highlights

Across 12 engineering milestones, the pipeline was benchmarked across **Track A (Controlled Synthetic Mixtures)** and **Track B (Simulated Broadcast & Telephony Domain-Stress Clips)**:

* **Blind Source Separation (Stage 1, Track A)**: Achieved **$+15.45\text{ dB}$ mean $\Delta\text{SI-SDR}$** ($+15.451\text{ dB}$ exact; peak $+21.76\text{ dB}$, min $+4.81\text{ dB}$) over raw overlapping speech (mean baseline **$-0.003\text{ dB}$**, i.e., $-0.00\text{ dB}$ rounded; mean output **$15.448\text{ dB}$**, $15.45\text{ dB}$ rounded) using **Conv-TasNet** (`CONVTASNET_BASE_LIBRI2MIX` with $16\leftrightarrow 8\text{ kHz}$ polyphase resampling; mean RTF **$0.090$** in committed JSON `stage1_separation_results.json`; earlier M2 report on `mps:0` logged $0.075$). Hybrid Demucs (`htdemucs`) served as comparator ($-8.598\text{ dB}$ output, $-8.595\text{ dB}$ $\Delta\text{SI-SDR}$, RTF $0.561$ due to music-stem grouping).
* **Speaker Diarization (Stage 2, Track A)**: Reduced mean NIST-collar DER across nine controlled mixtures from **$40.52\%$ to $23.97\%$** ($16.55\text{ pp}$ reduction; GMM baseline: $55.51\%$) via native 192-d acoustic-statistics hyperspherical embeddings (`pyannote.audio` weights absent at runtime) and maximal eigengap spectral clustering (mean RTF $0.006$).
* **Bilingual ASR Recovery (Stage 3, Track A)**: Reduced `controlled_mix_01` WER from **$86.28\%$ (raw unseparated)** and **$70.78\%$ (raw + diarization)** down to **$51.57\%$** (CER $44.66\%$, CPU RTF $0.1936$) using greedy **Vakyansh Hindi Wav2Vec2** (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`) decoding with Unicode NFKC normalization; comparator `openai/whisper-tiny` collapsed to $100.0\%$ WER on raw overlap and $103.34\%$ on separated stems.
* **Exploratory Synthetic Scenario Sweep (M9)**: Across 7 predefined synthetic conditions (`EXP-2.1`–`EXP-2.7` in `scripts/run_ablation_study.py`), SI-SDR correlated inversely with DER ($r = -0.977$) and WER ($r = -0.969$), with configured scenario `EXP-2.5` at **$\text{SI-SDR} = +4.2\text{ dB}$** ($28.9\%$ DER, $42.7\%$ WER, $5.8\%$ hallucination) marking an exploratory stress boundary rather than an end-to-end empirical threshold. Note that actual end-to-end execution inside `run_ablation_study.py` ran default `htdemucs` + `mock_mode=True` ASR (`-8.52 dB` SI-SDR, `42.53%` DER, `100.0%` mock WER).
* **Schema & Rule-Based Grounding (Stage 4)**: Evaluated offline via deterministic `MockLLMProvider` templates, achieving **$0.0\%$** unsupported-entity rate (`LLMMetricsEvaluator` checks entities/numbers against despaced CTC tokens and an Indic-English phonetic map, returning $0.0\%$ when zero candidate entities exist). This validates deterministic schema/template fidelity rather than open-ended LLM safety.
* **Domain Stress & Hardware Profiling (M10–M11)**: On WSL2 8-core x86_64 CPU (`benchmark_summary.json`), isolated 30 s end-to-end execution took **$20.90\text{ s}$ (RTF $0.697$)**, Stage 1 separation alone took **$19.40\text{ s}$ (RTF $0.647$)**, and continuous 30 s scaling achieved **RTF $0.490$**. Track B cold-start execution ran at RTF $19.98$, exposing cluster collapse ($k^*=1$, $58.7\%$ DER) on 3-speaker overlap and cellular bandpass WER ($63.0\%$).

### Empirical Benchmark Summary

| Evaluation Dimension | Suite / Scope | Baseline Comparator | Proposed / Primary Result | Measured Impact |
|---|---|---|---|---|
| **Separation Fidelity ($\Delta\text{SI-SDR}$)** | Track A: 6 two-spk mixes | Raw mixture: $-0.003\text{ dB}$ ($-0.00\text{ dB}$) | **Conv-TasNet: $+15.45\text{ dB}$** (out: $15.45\text{ dB}$; Demucs: $-8.60\text{ dB}$) | $+15.45\text{ dB}$ clean stem gain |
| **Diarization Error Rate (DER)** | Track A: 9 mixtures | Raw spectral: $40.52\%$ (GMM: $55.51\%$) | **Separated spectral: $23.97\%$** (`mix_01`: $9.20\%$) | $-16.55\text{ pp}$ mean DER reduction |
| **Bilingual Word Error Rate (WER)** | Track A: `mix_01` (9.54 s) | Raw Vakyansh: $86.28\%$ (Whisper: $100.0\%$) | **Sep+Diar Vakyansh: $51.57\%$** (`44.66%` CER) | $-34.71\text{ pp}$ WER reduction |
| **Stage 4 Unsupported Entities** | `MockLLMProvider` (2 clips) | — | **$0.00\%$** (`LLMMetricsEvaluator` rule check) | Deterministic Pydantic Contract 5 fidelity |
| **Execution Latency (RTF)** | M2 Stage 1 / M10 WSL2 CPU | Stage 1 CPU 30 s: $0.647$ ($19.40\text{ s}$) | **Stage 1 JSON: $0.090$** (M2 `mps:0`: $0.075$); **CPU E2E: $0.697$ / $0.490$** | $11.1\times$ Stage 1 JSON; $1.44\times\text{–}2.04\times$ CPU |

---

## 3. Repository Directory Layout

```
Audio-Engineering/
├── Project_report/                   # Official 4-Page Academic Report & LaTeX Source
│   ├── Final_project_submission.pdf  # Compiled 4-page camera-ready academic report
│   └── Final_project_submission.tex  # Verified LaTeX template source
├── configs/                          # Production YAML configurations
│   ├── default_config.yaml           # Master system default configuration
│   ├── pipeline.yaml                 # Master orchestrator config (alias)
│   └── llm_guardrails.yaml           # Jinja2 prompt constraints & negative rules
├── data/                             # Standardized benchmark datasets
│   ├── controlled_eval_suite/        # Track A: Synthetic controlled mixtures (M1, M9)
│   ├── real_world_eval_suite/        # Track B: AIR Radio & Vaani field slices (M11)
│   ├── benchmark_fixtures/           # Standard duration sweeps (15s, 30s, 60s, 120s)
│   └── manifests/                    # Ground-truth JSON manifests
├── docs/                             # Interactive Demonstration & Web Showcase (GitHub Pages)
│   ├── assets/                       # Scientific benchmark plots and waveform diagrams
│   ├── .nojekyll                     # GitHub Pages deployment configuration
│   └── index.html                    # Interactive audio showcase & demonstration
├── evaluation/                       # Evaluation engines & metrics
│   ├── bss_metrics.py                # SI-SDR, SDR, SIR, PIT Hungarian assignment
│   ├── diarization_metrics.py        # NIST-collar DER, JER, Eigengap heuristic
│   ├── asr_metrics.py                # Levenshtein WER, CER, Indic Unicode NFKC normalizer
│   ├── llm_metrics.py                # Hallucination auditor, ROUGE-L, attribution integrity
│   ├── ablation_runner.py            # Progressive ablation matrix (Pipelines A, B, C, D)
│   ├── error_propagation.py          # Mathematical error cascading & correlation engine
│   ├── failure_taxonomy.py           # 4-Tier canonical failure taxonomy & diagnostics
│   └── real_world_evaluator.py       # Track B domain robustness & cellular stress engine
├── notebooks/                        # Interactive analysis notebooks
│   └── 01_system_benchmark.ipynb     # Official M10 latency, RTF, RAM & VRAM profiling
├── outputs/                          # Generated artifacts, plots, and reports
│   ├── benchmarks/                   # Latency, RTF, and memory waterfall plots
│   ├── real_world_validation/        # Domain WER/CER, radar charts, and JSON dataset
│   └── reports/                      # Formal Markdown reports for M0 through M11
├── scripts/                          # Standalone CLI benchmark & evaluation runners
│   ├── run_ablation_study.py         # M9 Scientific ablation & error propagation CLI
│   ├── run_system_benchmark.py       # M10 System benchmarking & profiling CLI
│   └── run_real_world_validation.py  # M11 Real-world domain robustness CLI
├── src/                              # Core production package (BaseModule architecture)
│   ├── __main__.py                   # CLI entrypoint (python -m src)
│   ├── cli.py                        # Rich Command-Line Interface
│   ├── pipeline.py                   # Master SpeechEngineeringPipeline orchestrator
│   ├── core/                         # Audio I/O, device resolution, and Pydantic contracts
│   ├── preprocessing/                # Audio normalization, dynamic range, mixture synthesis
│   ├── separation/                   # Meta's Hybrid Demucs & Conv-TasNet wrappers
│   ├── diarization/                  # Neural speaker embeddings & spectral clustering
│   ├── asr/                          # Vakyansh Hindi Wav2Vec2, Whisper comparator, Indic normalizer
│   └── llm/                          # Swappable LLM providers, Jinja2 prompts, guardrails, RAG engine
└── tests/                            # 127 automated unit and integration tests (100% passing)
```

---

## 4. Getting Started & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.3 in Ubuntu WSL2 / Linux / macOS).
- PyTorch with CPU, CUDA, or Apple Silicon MPS support.
- C++ build tools and `libsndfile` (`sudo apt-get install -y libsndfile1 ffmpeg` or `brew install libsndfile ffmpeg`).

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/LearnerKg23/Audio-Engineering.git
cd Audio-Engineering

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Full Test Suite (Verification)
Execute all automated unit and integration tests (`127 collected, 127 passed, 0 failed, 0 skipped`):
```bash
pytest tests/ -v
```
*(Verified result: `127 passed in 116.63s` with 0 failures and 0 skipped tests).*

---

## 5. Usage & Quickstart

### Command-Line Interface (CLI)

The pipeline can be executed directly on any audio file (.wav, .mp3, .flac) via `python -m src`:

```bash
# 1. Full Multi-Speaker Pipeline (Separation -> Diarization -> ASR -> LLM Guardrails)
python -m src --input data/real_world_eval_suite/air_spontaneous_panel_01.wav \
              --output-dir outputs/runs/demo_panel

# 2. Single-Speaker Clean Audio (Graceful Separation Bypass)
python -m src --input data/real_world_eval_suite/air_formal_news_01.wav \
              --bypass-separation \
              --output-dir outputs/runs/demo_news

# 3. Model Overrides & Device Selection
python -m src --input path/to/audio.wav \
              --separation-model demucs \
              --asr-engine indic_wav2vec \
              --llm-provider mock \
              --device cpu

# 4. Interactive Conversational Intelligence (RAG Query with Negative Guardrails)
python -m src --input data/real_world_eval_suite/air_spontaneous_panel_01.wav \
              --ask "What announcement was made regarding financial assistance for women?"
```

### Benchmark & Experiment Reproduction Guide

All empirical results, ablation matrices, and benchmark tables reported in the four-page academic paper and documentation can be reproduced using the standalone evaluation scripts below:

#### 1. Track A: Blind Source Separation Benchmark (Milestone M2)
Evaluates Conv-TasNet (`CONVTASNET_BASE_LIBRI2MIX`, with $16\leftrightarrow 8\text{ kHz}$ polyphase resampling) and Hybrid Demucs (`htdemucs`) across the 6 controlled two-speaker mixtures under permutation-invariant training (PIT Hungarian matching):
```bash
python scripts/evaluate_stage1_separation.py
python scripts/plot_sisdr_comparison.py
```
- **Configuration**: `configs/default_config.yaml` (`separation` section)
- **Primary Inputs**: `data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01` through `06`
- **Expected Output Artifacts**:
  - `outputs/first_eval_demo/stage1_separation_results.json` (Mean baseline SI-SDR: $-0.003\text{ dB}$; Conv-TasNet output: $15.448\text{ dB}$, mean $\Delta$SI-SDR: $+15.451\text{ dB}$ [$+15.45\text{ dB}$], peak $+21.76\text{ dB}$, mean RTF $0.090$ [earlier M2 report `e712501` on `mps:0` logged $0.075$]; Demucs output: $-8.598\text{ dB}$, $\Delta$SI-SDR: $-8.595\text{ dB}$, mean RTF $0.561$)
  - `outputs/first_eval_demo/sisdr_model_comparison_barchart.png`
  - `outputs/first_eval_demo/final_report.json`

#### 2. Track A: Speaker Diarization Benchmark (Milestone M5)
Evaluates spectral clustering on 192-d native acoustic embeddings against unseparated spectral and GMM baselines across 9 controlled mixtures:
```bash
python scripts/evaluate_stage2_diarization.py
```
- **Configuration**: `configs/default_config.yaml` (`diarization` section: window 1.5s, step 0.5s, collar 0.25s)
- **Primary Inputs**: `data/controlled_eval_suite/synthetic_mixtures/` (`controlled_mix_01` through `09`)
- **Expected Output Artifacts**:
  - `outputs/reports/stage2_diarization_evaluation.json` (Raw spectral DER: $40.52\%$, GMM DER: $55.51\%$, Separated DER: $23.97\%$, mean RTF $0.006$)

#### 3. Track A: Downstream Indic & Code-Switched ASR Evaluation (Milestone M6)
Evaluates downstream Vakyansh Hindi Wav2Vec2 (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`, greedy CTC argmax decoding without LM rescoring) across naive mixture, raw+diarized, and separated+diarized pipelines on `controlled_mix_01`:
```bash
python scripts/evaluate_stage3_asr.py
# Optional Whisper-tiny comparator (100.0% raw / 103.34% separated):
python scripts/evaluate_downstream_asr.py
```
- **Configuration**: `configs/default_config.yaml` (`asr` section: `primary_engine: "indic_wav2vec"`, NFKC + nukta harmonization)
- **Primary Input**: `data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav`
- **Expected Output Artifacts**:
  - `outputs/reports/stage3_asr_evaluation.json` & `stage3_asr_report.md` (Raw WER: $86.28\%$, Diarized: $70.78\%$, Separated+Diarized: $51.57\%$, CER $44.66\%$, CPU RTF $0.1936$)
  - `outputs/first_eval_demo/downstream_asr_evaluation.json` (Whisper-tiny comparator: $100.0\%$ raw repetition loop; $103.34\%$ separated stems)

#### 4. Cross-Stage Error Propagation & Progressive Ablation Study (Milestone M9)
Executes progressive end-to-end ablation runs (`htdemucs` + `mock_mode=True` ASR/LLM: `Pipeline D` yields $-8.52\text{ dB}$ SI-SDR, $42.53\%$ DER, $100.0\%$ mock WER, $0.0\%$ mock hallucination, $0.5581$ RTF) and evaluates correlations across 7 predefined synthetic stress scenarios (`EXP-2.1` to `EXP-2.7`):
```bash
python scripts/run_ablation_study.py
```
- **Configuration**: `configs/llm_guardrails.yaml`, `src/llm/prompts/post_processing_prompt.jinja`
- **Primary Inputs**: `controlled_mix_01` + 7 configured synthetic scenario grid points (`EXP-2.1`–`EXP-2.7`)
- **Expected Output Artifacts**:
  - `outputs/reports/ablation_study_results.json` & `error_propagation_study.md` (Synthetic sweep correlations $r = -0.977$ SI-SDR vs DER, $r = -0.969$ SI-SDR vs WER; configured stress boundary at `EXP-2.5`: $+4.2\text{ dB}$)

#### 5. System Computational Benchmarking & Profiling (Milestone M10)
Profiles isolated stage latencies, memory footprint, and continuous duration scaling (15s, 30s, 60s, 120s) on an 8-core x86_64 WSL2 CPU host:
```bash
python scripts/run_system_benchmark.py
# Or launch interactive Jupyter notebook:
jupyter notebook notebooks/01_system_benchmark.ipynb
```
- **Configuration**: `configs/default_config.yaml` (`system` section)
- **Primary Inputs**: `data/benchmark_fixtures/benchmark_audio_{15s,30s,60s,120s}.wav`
- **Expected Output Artifacts**:
  - `outputs/benchmarks/benchmark_summary.json` (30s isolated E2E: $20.90\text{ s}$ / RTF $0.697$; Stage 1 separation alone: $19.40\text{ s}$ / RTF $0.647$; continuous 30s scaling: $14.71\text{ s}$ / RTF $0.490$)
  - `outputs/benchmarks/stage_latency_breakdown.png`, `rtf_scaling_curves.png`, `memory_profile_waterfall.png`

#### 6. Track B: Simulated Broadcast & Telephony Domain-Stress Suite (Milestone M11)
Stress-tests default pipeline execution (`htdemucs`, native 192-d spectral diarization, Vakyansh CTC, `MockLLMProvider`) on simulated domain-stress recordings synthesized from Track A stems via `scripts/expand_data_pipeline.py`:
```bash
python scripts/run_real_world_validation.py
```
- **Configuration**: `configs/default_config.yaml` (Track B defaults; does not invoke Conv-TasNet)
- **Manifest**: `data/manifests/real_world_manifest.json`
- **Expected Output Artifacts**:
  - `outputs/real_world_validation/real_world_validation_results.json` (News DER $0.0\%$, nominal 41-word WER $92.68\%$; Panel DER $58.7\%$, WER $57.78\%$; Cellular call WER $62.96\%$)
  - `outputs/real_world_validation/air_formal_news_01_diagnostic.json` (Clip-aligned 14-word spoken diagnostic WER: $57.14\%$ [$8/14$], CER $0.00\%$ [$69/69$ chars with Devanagari `"एआई"`])
  - `outputs/reports/real_world_domain_robustness_report.md` and domain charts in `outputs/real_world_validation/`

### Python API Integration

```python
from pathlib import Path
from src.pipeline import SpeechEngineeringPipeline

# Initialize pipeline
pipeline = SpeechEngineeringPipeline(config_path="configs/pipeline.yaml")

# Run end-to-end inference
state = pipeline.run(
    input_audio_path="data/real_world_eval_suite/vaani_rural_call_01.wav",
    save_artifacts=True,
)

# Inspect inter-stage Pydantic data contracts
print(f"Session ID: {state.session_id}")
print(f"Total Speakers: {state.diarization_timeline.total_speakers_detected}")
for utt in state.raw_transcript.utterances:
    print(f"[{utt.start_time:.1f}s - {utt.end_time:.1f}s] {utt.speaker_id}: {utt.raw_text}")

# Query conversation with Stage 4 RAG Engine
rag_answer = pipeline.answer_question(
    query="What baggage restrictions were mentioned?",
    state=state,
)
print(f"RAG Grounded Answer: {rag_answer.answer}")
print(f"Cited Speakers: {rag_answer.cited_speakers}")
print(f"Guardrail Status: {rag_answer.guardrail_status}")
```

---

## 6. Comprehensive Milestones Summary

| Milestone | Phase | Description | Status | Verification Reference |
|---|---|---|---|---|
| **M0** | Setup | Environment, PyTorch hardware resolution, Audio I/O contracts | **VERIFIED** | `tests/test_audio_io.py`, `tests/test_core.py` |
| **M1** | Data | Controlled multi-speaker mixture generator with calibrated overlap | **VERIFIED** | `tests/test_mixture_generator.py` |
| **M2** | Stage 1 | Demucs & Conv-TasNet BSS with PIT SI-SDR evaluation | **VERIFIED** | `tests/test_separation.py`, `tests/test_bss_metrics.py` |
| **M3** | Demo | Interactive evaluation web showcase & audio players | **VERIFIED** | [`docs/index.html`](docs/index.html) |
| **M4** | Corpus | Extended Track A (9 mixtures) & Track B simulated domain-stress suite | **VERIFIED** | `data/manifests/controlled_ground_truth.json`, `real_world_manifest.json` |
| **M5** | Stage 2 | Speaker Diarization with Eigengap spectral clustering & DER/JER | **VERIFIED** | `tests/test_diarization.py` |
| **M6** | Stage 3 | Vakyansh Hindi Wav2Vec2 CTC ASR with Unicode NFKC normalization | **VERIFIED** | `tests/test_asr.py` |
| **M7** | Stage 4 | Semantic LLM post-processing engine & structural guardrails | **VERIFIED** | `tests/test_llm.py` |
| **M8** | Integration| Master `SpeechEngineeringPipeline` CLI orchestrator & caching | **VERIFIED** | `tests/test_pipeline_integration.py` |
| **M9** | Science | Cross-stage error propagation & progressive ablation study | **VERIFIED** | `outputs/reports/error_propagation_study.md` |
| **M10**| Benchmarking| Official computational profiling notebook (Latency, RTF, RAM, VRAM) | **VERIFIED** | `notebooks/01_system_benchmark.ipynb` |
| **M11**| Validation | Track B simulated broadcast & cellular-bandpass domain stress suite | **VERIFIED** | `outputs/reports/real_world_domain_robustness_report.md` |
| **M12**| Portfolio | Final 4-page Academic Report, Test Validation, and Packaging | **VERIFIED** | [`Project_report/Final_project_submission.pdf`](Project_report/Final_project_submission.pdf) |

---

## 7. Interactive Showcase & Web Audio Demonstrator

- **Live Application**: [https://learnerkg23.github.io/Audio-Engineering/](https://learnerkg23.github.io/Audio-Engineering/) (or local [`docs/index.html`](docs/index.html))

Explore:
- Interactive A/B waveform and audio players comparing Raw Overlapping Mixtures against Clean Stems.
- Interactive Speaker Diarization timelines with multi-track DAW lanes.
- Comparative WER/CER tables for Vakyansh Hindi Wav2Vec2 vs. Whisper.
- Stage 4 Structured Reports (`MockLLMProvider` Pydantic Contract 5 template outputs & rule-based entity checks).
- Exploratory synthetic stress diagnostics (`EXP-2.1`–`EXP-2.7`, $+4.2\text{ dB}$ boundary) and Track B domain-stress telemetry.

---

## 8. License & Academic Attribution

### 8.1 Software License
This project is released under the **MIT License**.

### 8.2 Dataset Provenance & Audio Construction
The evaluation suites in this repository are organized into two distinct tracks (verified from `scripts/expand_data_pipeline.py` and `data/manifests/`):

- **Track A: Controlled Synthetic Benchmark**
  - Synthesized from 7 clean single-speaker Hindi/Hinglish stems (`speaker_A` through `speaker_G` in `data/controlled_eval_suite/clean_sources/`). Repository documentation identifies Hugging Face `nameissakthi/hindi-english-bilingual` for `speaker_A` and `speaker_B`, and `Speech-data/Hindi-Speech-Dataset` for `speaker_D`; upstream source metadata and license terms for `speaker_C`, `speaker_E`, `speaker_F`, and `speaker_G` are not specified in repository manifests or scripts.
  - Mixed at calibrated overlap ratios ($\Omega \in \{25\%, 50\%, 75\%\}$) and Signal-to-Interference Ratios ($\text{SIR} \in \{-6, 0, +6\}\text{ dB}$), plus additive background noise ($10\text{ dB}$ SNR in `mix_09`), strictly for non-commercial academic evaluation.

- **Track B: Simulated Broadcast & Telephony Domain-Stress Suite**
  - All three Track B WAV files in `data/real_world_eval_suite/` were synthetically generated from Track A stems inside `scripts/expand_data_pipeline.py` to model target domain acoustics (rather than ingested from uncurated third-party broadcasts):
  - `air_formal_news_01.wav` (6.0 s, SNR 47.1 dB): RMS-normalized (`-18.0 dBFS`) `speaker_D_male_hindi.wav`. **Manifest Alignment Note**: `real_world_manifest.json` stores a 41-word two-sentence reference whose Sentence 1 phrasing diverges (`"जिसे आप AI..."`) plus an unuttered 24-word Sentence 2, yielding `92.68%` nominal WER (`8` substitutions, `30` Levenshtein deletions). Evaluated against the actual 14 spoken words in the 6.0 s clip (`outputs/real_world_validation/air_formal_news_01_diagnostic.json`), diagnostic spoken WER is `57.14%` (`8/14`: 5 substitutions, 3 deletions) with `0.00%` CER (`69/69` chars exact with Devanagari `"एआई"`), showing errors come from CTC single-delimiter word merging.
  - `air_spontaneous_panel_01.wav` (13.0 s, SNR 35.6 dB): Synthetic 3-speaker overlay of `speaker_A`, `speaker_F`, and `speaker_G` with ambient Gaussian noise (DER `58.70%` due to $k^*=1$ spectral collapse under dense overlap; WER `57.78%`).
  - `vaani_rural_call_01.wav` (8.3 s, SNR 21.1 dB): `speaker_E_male_hinglish.wav` filtered through a 4th-order Butterworth telephony bandpass (`300–3400 Hz`) plus 100 Hz line hum and noise (DER `0.65%`, WER `62.96%`).

### 8.3 Benchmark Artifact Versioning & Verification Environment
- **Baseline Benchmark Commit**: `26a5319` (`Stage 1` JSON `134da09`, `Stage 2–5` JSONs `2026-10-12` to `2026-11-03`)
- **Evaluation Environment**: macOS / WSL2 Ubuntu x86_64 CPU & Apple Silicon MPS, Python 3.12, PyTorch 2.2+, Torchaudio 2.2+, **127/127 passing pytest suite** (`127 collected, 127 passed, 0 failed, 0 skipped`).

### 8.3 Academic Citation
If you use this work or its error propagation methodologies in your academic research, please cite:
```bibtex
@misc{indian_speech_engineering_2026,
  author = {Audio Engineering AI Research Group},
  title = {Advanced Computational Speech Engineering: A Multi-Stage "Who Spoke What and When" Audio Processing Pipeline},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/LearnerKg23/Audio-Engineering}}
}
```

