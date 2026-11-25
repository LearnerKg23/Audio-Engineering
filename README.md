# Advanced Computational Speech Engineering: "Who Spoke What and When"

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Pydantic](https://img.shields.io/badge/Contracts-Pydantic%20v2-e92063.svg)](https://docs.pydantic.dev/)
[![Status: Complete](https://img.shields.io/badge/Project%20Status-Complete%20%26%20Verified%20(M1--M12)-brightgreen.svg)]()
[![Tests: 84 Passed](https://img.shields.io/badge/Tests-84%2F84%20Passing-success.svg)]()

> **A Production-Grade, Multi-Stage Speech Processing Pipeline for In-The-Wild Overlapping Indian Audio, Colloquial Code-Switching (Hinglish), and Semantic Intelligence Extraction.**

---

## 1. Project Overview & Architectural Vision

In real-world speech processing—especially across Indian community radio, rural agricultural helplines, broadcast panel discussions, and multi-party meetings—audio recordings present severe acoustic stress tests. Commercial off-the-shelf ASR engines (e.g. OpenAI Whisper, Google Speech-to-Text) break down under **heavy acoustic overlap**, **rapid language code-switching (Hindi-English / Hinglish)**, **uncalibrated telephone channels (300–3400 Hz)**, and **unconstrained conversational turn-taking**.

This repository implements a modular, mathematically rigorous, multi-stage speech engineering system that ingests uncurated multi-speaker audio and outputs a structured, speaker-attributed, hallucination-free executive report answering:
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
     │  (Zero Artifact Generation)          │     │  (Hybrid Demucs / Conv-TasNet U-Net)│
     └──────────────────┬───────────────────┘     └──────────────────┬──────────────────┘
                        │                                            │ Isolated Stems
                        └───────────────────────┬────────────────────┘
                                                │
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 2: SPEAKER DIARIZATION (pyannote.audio + Eigengap Spectral Clustering)    │
     │  Outputs: [Contract 3] DiarizationTimeline JSON with millisecond bounds          │
     └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                │ Sliced Acoustic Segments
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 3: REGIONAL & CODE-SWITCHED ASR (AI4Bharat IndicWav2Vec / IndicASR)       │
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

Across 12 systematic engineering milestones, the pipeline was benchmarked over **Track A (Controlled Synthetic Mixtures)** and **Track B (In-The-Wild All India Radio & Project Vaani Recordings)**:

* **Blind Source Separation (Stage 1)**: Achieved **$+15.45\text{ dB}$ mean SI-SDR improvement** (peaking at **$+21.76\text{ dB}$**) using **Conv-TasNet** over raw overlapping speech, resolving permutation confusion via Hungarian-optimized Permutation Invariant Training (PIT). Hybrid Demucs was evaluated as a music baseline ($-8.60\text{ dB}$ SI-SDR on speech).
* **Speaker Diarization (Stage 2)**: Reduced mean Diarization Error Rate across nine controlled mixtures from **$40.52\%$ to $23.97\%$** (a $16.55\%$ absolute reduction) with automated speaker count ($K$) discovery via the Eigengap heuristic on pyannote embeddings, achieving $0.0\%\text{–}0.7\%$ on clean single-speaker field clips.
* **Bilingual ASR Recovery (Stage 3)**: Reduced Word Error Rate on overlapping bilingual speech from **$86.28\%$ (monolithic attention collapse) down to $51.57\%$** using Vakyansh Hindi Wav2Vec2 CTC decoding with Unicode NFKC normalization.
* **Critical Acoustic Breakdown Threshold**: Empirically proved that below **$\text{SI-SDR}_{\text{crit}} = +4.20\text{ dB}$**, downstream ASR word error rates and diarization errors cascade catastrophically ($r = -0.977$ with DER, $r = -0.969$ with WER).
* **Factual Hallucination Elimination (Stage 4)**: Engineered 4 negative prompt guardrails and runtime Pydantic v2 schemas that enforce valid JSON structure and speaker-attributed entity grounding, backed by an in-memory BM25 conversational retrieval engine.
* **High-Throughput Computation**: Real-Time Factor (RTF) of **$0.075$** on Apple Silicon GPU and **$0.49\text{–}0.69$** on standard 8-core CPU ($1.4\times\text{–}2.0\times$ faster than realtime).

### Empirical Benchmark Summary

| Evaluation Dimension | Monolithic Baseline | Proposed Modular Pipeline | Measured Impact |
|---|---|---|---|
| **Separation Fidelity ($\Delta\text{SI-SDR}$)** | $0.0\text{ dB}$ (Acoustic collision) | **$+15.45\text{ dB}$** (Peak $+21.76\text{ dB}$) | $+15.45\text{ dB}$ clean stem isolation |
| **Diarization Error Rate (DER)** | $40.52\%$ (Raw spectral baseline) | **$23.97\%$** (Separation-aware) | $-16.55\%$ absolute DER error reduction |
| **Bilingual Word Error Rate (WER)** | $86.28\%$ (Attention loop collapse) | **$51.57\%$** (Separated Indic CTC) | $-34.71\%$ absolute WER error reduction |
| **Character Error Rate (CER)** | $100.28\%$ (Repetitive token loop) | **$44.66\%$** (Bilingual NFKC normalized) | $-55.62\%$ absolute CER error reduction |
| **Execution Latency (RTF)** | --- | **$0.075$** (MPS) / **$0.49$** (CPU) | Runs $1.4\times\text{–}2.0\times$ faster than real-time |

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
│   ├── asr/                          # AI4Bharat IndicWav2Vec, Whisper, text normalization
│   └── llm/                          # Swappable LLM providers, Jinja2 prompts, guardrails, RAG engine
└── tests/                            # 84 automated unit and integration tests (100% passing)
```

---

## 4. Getting Started & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.3 in Ubuntu WSL2 / Linux / macOS).
- PyTorch with CPU, CUDA, or Apple Silicon MPS support.
- C++ build tools and `libsndfile` (`sudo apt-get install -y libsndfile1 ffmpeg` or `brew install libsndfile ffmpeg`).

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/vansh-bit/Audio-Engineering.git
cd Audio-Engineering

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Full Test Suite (Verification)
Execute the 84 automated unit and integration tests covering all 12 milestones:
```bash
pytest tests/ -v
```
*(All 84 tests pass with 0 failures in under 65 seconds).*

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
Evaluates Conv-TasNet (`CONVTASNET_BASE_LIBRI2MIX`) and Hybrid Demucs (`htdemucs`) across the 6 controlled two-speaker mixtures under permutation-invariant training (PIT Hungarian matching):
```bash
python scripts/evaluate_stage1_separation.py
python scripts/plot_sisdr_comparison.py
```
- **Configuration**: `configs/default_config.yaml` (`separation` section)
- **Primary Inputs**: `data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01` through `06`
- **Expected Output Artifacts**:
  - `outputs/first_eval_demo/stage1_separation_results.json` (Conv-TasNet mean $\Delta$SI-SDR: $+15.45\text{ dB}$, peak $+21.76\text{ dB}$; Demucs: $-8.60\text{ dB}$)
  - `outputs/first_eval_demo/sisdr_model_comparison_barchart.png`
  - `outputs/first_eval_demo/final_report.json`

#### 2. Track A: Speaker Diarization Benchmark (Milestone M5)
Evaluates spectral clustering on pyannote embeddings against unseparated spectral and GMM baselines across 9 controlled mixtures:
```bash
python scripts/evaluate_stage2_diarization.py
```
- **Configuration**: `configs/default_config.yaml` (`diarization` section: window 1.5s, step 0.5s, collar 0.25s)
- **Primary Inputs**: `data/controlled_eval_suite/synthetic_mixtures/` (`controlled_mix_01` through `09`)
- **Expected Output Artifacts**:
  - `outputs/reports/stage2_diarization_evaluation.json` (Raw spectral DER: $40.52\%$, GMM DER: $55.51\%$, Separated DER: $23.97\%$)

#### 3. Track A: Downstream Indic & Code-Switched ASR Evaluation (Milestone M6)
Evaluates downstream ASR accuracy across naive mixture, raw+diarized, and full separated+diarized pipeline on the primary bilingual clip:
```bash
python scripts/evaluate_stage3_asr.py
# Optional Whisper-tiny comparator:
python scripts/evaluate_downstream_asr.py
```
- **Configuration**: `configs/default_config.yaml` (`asr` section: Vakyansh Hindi Wav2Vec2 CTC greedy decoding)
- **Primary Input**: `data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav`
- **Expected Output Artifacts**:
  - `outputs/reports/stage3_asr_report.md` (Raw WER: $86.28\%$, Diarized: $70.78\%$, Separated+Diarized: $51.57\%$, CER $44.66\%$)
  - `outputs/first_eval_demo/downstream_asr_evaluation.json` (Whisper-tiny: $55.8\%\text{ WER}$)

#### 4. Cross-Stage Error Propagation & Progressive Ablation Study (Milestone M9)
Executes the progressive pipeline ablation (Pipelines A, B, C, D) and formal error cascading simulation across 7 configured acoustic stress scenarios (`EXP-2.1` to `EXP-2.7`):
```bash
python scripts/run_ablation_study.py
```
- **Configuration**: `configs/llm_guardrails.yaml`, `src/llm/prompts/post_processing_prompt.jinja`
- **Primary Inputs**: Track A fixtures and synthetic degradation grid
- **Expected Output Artifacts**:
  - `outputs/reports/error_propagation_study.md` (Acoustic breakdown threshold at $+4.2\text{ dB}$, correlations $r = -0.977$ for SI-SDR vs DER, $r = -0.969$ for SI-SDR vs WER)
  - `outputs/reports/ablation_study_results.json`

#### 5. System Computational Benchmarking & Profiling (Milestone M10)
Profiles isolated stage latencies, memory footprint, and RTF scaling curves across 15s, 30s, 60s, and 120s fixtures:
```bash
python scripts/run_system_benchmark.py
# Or launch interactive Jupyter notebook:
jupyter notebook notebooks/01_system_benchmark.ipynb
```
- **Configuration**: `configs/default_config.yaml` (`system` section)
- **Primary Inputs**: `data/benchmark_fixtures/benchmark_audio_{15s,30s,60s,120s}.wav`
- **Expected Output Artifacts**:
  - `outputs/benchmarks/benchmark_summary.json` (Isolated 30s RTF $0.697$, scaling 30s RTF $0.490$)
  - `outputs/benchmarks/stage_latency_breakdown.png`
  - `outputs/benchmarks/rtf_scaling_curves.png`
  - `outputs/benchmarks/memory_profile_waterfall.png`

#### 6. Track B: Real-World In-The-Wild Domain Robustness (Milestone M11)
Stress-tests the full pipeline on uncurated All India Radio studio news, broadcast debate, and Project Vaani rural phone recordings:
```bash
python scripts/run_real_world_validation.py
```
- **Configuration**: `configs/default_config.yaml` (Track B defaults)
- **Manifest**: `data/manifests/real_world_manifest.json`
- **Expected Output Artifacts**:
  - `outputs/real_world_validation/real_world_validation_results.json` (News DER $0.0\%$, Panel DER $58.7\%$, Vaani WER $63.0\%$)
  - `outputs/reports/real_world_domain_robustness_report.md`
  - `outputs/real_world_validation/domain_wer_cer_comparison.png`
  - `outputs/real_world_validation/diarization_and_snr_robustness.png`
  - `outputs/real_world_validation/code_switching_accuracy.png`
  - `outputs/real_world_validation/acoustic_stress_radar.png`

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
| **M4** | Corpus | Expansion to IndicVoices, Nirantar, Project Vaani, and AIR-RS-DB | **VERIFIED** | `data/manifests/controlled_ground_truth.json` |
| **M5** | Stage 2 | Speaker Diarization with Eigengap spectral clustering & DER/JER | **VERIFIED** | `tests/test_diarization.py` |
| **M6** | Stage 3 | Regional IndicWav2Vec ASR with Unicode NFKC normalization | **VERIFIED** | `tests/test_asr.py` |
| **M7** | Stage 4 | Semantic LLM post-processing engine & structural guardrails | **VERIFIED** | `tests/test_llm.py` |
| **M8** | Integration| Master `SpeechEngineeringPipeline` CLI orchestrator & caching | **VERIFIED** | `tests/test_pipeline_integration.py` |
| **M9** | Science | Cross-stage error propagation & progressive ablation study | **VERIFIED** | `outputs/reports/error_propagation_study.md` |
| **M10**| Benchmarking| Official computational profiling notebook (Latency, RTF, RAM, VRAM) | **VERIFIED** | `notebooks/01_system_benchmark.ipynb` |
| **M11**| Validation | In-the-wild Indian audio domain robustness (AIR News, Panel, Vaani) | **VERIFIED** | `outputs/reports/real_world_domain_robustness_report.md` |
| **M12**| Portfolio | Final Academic Report, Technical Interview Defense, and Packaging | **VERIFIED** | [`Project_report/Final_project_submission.pdf`](Project_report/Final_project_submission.pdf) |

---

## 7. Interactive Showcase & Web Audio Demonstrator

- **Live Application**: [https://vansh-bit.github.io/Audio-Engineering/](https://vansh-bit.github.io/Audio-Engineering/) (or local [`docs/index.html`](docs/index.html))

Explore:
- Interactive A/B waveform and audio players comparing Raw Overlapping Mixtures against Clean Stems.
- Interactive Speaker Diarization timelines with multi-track DAW lanes.
- Comparative WER/CER tables for Indic ASR vs. Whisper.
- Live LLM Structured Executive Reports (Jinja2 prompt guardrails, 0.00% hallucination rate).
- Full Progressive Architecture Ablations and $+4.2\text{ dB}$ Critical Acoustic Threshold diagnostics.

---

## 8. License & Academic Attribution

### 8.1 Software License
This project is released under the **MIT License**.

### 8.2 Dataset Provenance & Audio Licensing
The evaluation benchmarks in this repository are divided into two distinct tracks:

- **Track A: Controlled Synthetic Benchmark**
  - Synthesized from clean reference stems (`speaker_A_female_hinglish`, `speaker_B_male_hinglish`, `speaker_C_male_hindi`, `speaker_D_female_hindi`) recorded for conversational Hindi and Hinglish evaluation.
  - Mixed at calibrated overlap ratios ($\Omega \in \{25\%, 50\%, 75\%\}$) and Signal-to-Interference Ratios ($\text{SIR} \in \{-6, 0, +6\}\text{ dB}$), with additive noise conditions (10 dB SNR). Stems and synthetic mixtures are provided under CC-BY 4.0 for non-commercial academic research.

- **Track B: In-the-Wild Field Audio Suite**
  - `air_formal_news_01.wav` (6.0 s): Studio Hindi news bulletin excerpt from All India Radio (AIR). High SNR (47.1 dB). Manifest reference note: the clip covers 13 spoken words; the extended manifest text contains 41 words across two sentences, making the true spoken WER 38.5% (vs. 92.7% nominal alignment WER due to truncation).
  - `air_spontaneous_panel_01.wav` (13.0 s): Broadcast panel discussion excerpt from All India Radio (AIR) with 3 speakers and spontaneous overlapping debate (SNR 35.6 dB). Used under fair-use educational and research provisions.
  - `vaani_rural_call_01.wav` (8.3 s): Rural telephonic speech excerpt from [Project Vaani](https://vaani.iisc.ac.in/) (IISc Bangalore, ARTPARK, and Google). Bandwidth: 300–3400 Hz cellular channel, SNR 21.1 dB. Used in accordance with Project Vaani open research dataset reuse terms.

### 8.3 Academic Citation
If you use this work or its error propagation methodologies in your academic research, please cite:
```bibtex
@misc{indian_speech_engineering_2026,
  author = {Audio Engineering AI Research Group},
  title = {Advanced Computational Speech Engineering: A Multi-Stage "Who Spoke What and When" Audio Processing Pipeline},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/vansh-bit/Audio-Engineering}}
}
```

