# 09 — System Benchmarking & Computational Profiling

## 1. Benchmarking Objective & Authoritative Mandate

The project proposal explicitly mandates:
> *"System Benchmark Notebook: A validation notebook measuring the computational latency and memory consumption of each module stage."*

In senior AI engineering, building a pipeline that produces correct outputs is only half the battle. The system must also be profiled for computational feasibility, memory footprints, and bottleneck identification across diverse hardware profiles.

---

## 2. Standardized Profiling Metrics & Mathematical Definitions

Every stage of the pipeline is evaluated across five standardized computational dimensions:

### 1. Wall-Clock Latency & Processing Time ($T_{\text{proc}}$)
Measured using high-resolution monotonic timers (`time.perf_counter()`):
$$T_{\text{proc}} = t_{\text{end}} - t_{\text{start}} \quad (\text{seconds})$$
Measured after an initial warmup pass to eliminate one-off PyTorch JIT compilation or model initialization overhead.

### 2. Real-Time Factor (RTF)
RTF quantifies how quickly an audio processing system operates relative to the physical playback duration of the input signal:
$$\text{RTF} = \frac{T_{\text{proc}}}{T_{\text{audio}}}$$
* **$\text{RTF} < 1.0$**: System operates *faster than real-time* (e.g., $\text{RTF} = 0.25$ means a 60-second audio clip is processed in 15 seconds).
* **$\text{RTF} = 1.0$**: System operates at parity with physical playback.
* **$\text{RTF} > 1.0$**: System is slower than real-time (unsuitable for streaming; requires chunking or GPU acceleration).

### 3. Peak System Memory (RAM)
Tracked via `tracemalloc` and `psutil.Process().memory_info().rss`:
$$\Delta \text{RAM} = \text{Peak RSS} - \text{Baseline RSS} \quad (\text{MB})$$

### 4. Peak GPU VRAM Allocation
Tracked using PyTorch device monitoring APIs:
* **CUDA**: `torch.cuda.max_memory_allocated()`, `torch.cuda.memory_reserved()`
* **Apple Silicon (MPS)**: `torch.mps.current_allocated_memory()` (where supported) or unified system memory profiling.

### 5. Audio Throughput
$$\text{Throughput} = \frac{T_{\text{audio}}}{T_{\text{proc}}} = \frac{1}{\text{RTF}} \quad (\text{audio seconds processed per second})$$

---

## 3. Standardized Benchmarking Schema

The benchmark results are logged into a structured dataframe and exported as JSON/CSV:

| Pipeline Stage | Model / Algorithm | Input Audio Duration ($T_{\text{audio}}$) | Processing Time ($T_{\text{proc}}$) | Real-Time Factor (RTF) | Peak RAM (MB) | Peak GPU VRAM (MB) | Throughput (sec/sec) |
|---|---|---|---|---|---|---|---|
| **0. Preprocessing** | Resampling + VAD | 60.0 s | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **1. Source Separation** | Meta Demucs (`htdemucs`) | 60.0 s | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **2. Diarization** | `pyannote` + Spectral | 60.0 s | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **3. ASR** | IndicWav2Vec / IndicASR | 60.0 s | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **4. LLM Post-Processing** | Local LLM (Airavata 7B) | ~150 Tokens | *Measured* | N/A (Tokens/s) | *Measured* | *Measured* | *Measured* |
| **Total End-to-End** | Unified Pipeline | 60.0 s | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |

---

## 4. Scaling & Stress-Testing Experiments

To analyze algorithmic complexity, the benchmark harness sweeps input durations across four orders of magnitude:
* **$T_1 = 15\text{ seconds}$** (Short audio snippet)
* **$T_2 = 30\text{ seconds}$** (Standard conversational turn)
* **$T_3 = 60\text{ seconds}$** (One-minute panel snippet)
* **$T_4 = 180\text{ seconds}$** (Three-minute complete radio discussion)

### Expected Scaling Observations:
* **Demucs (BSS)**: Time complexity scales linearly $\mathcal{O}(T)$ with duration, but memory scales with chunk size (Demucs uses internal sliding-window chunking, bounding peak VRAM).
* **Spectral Diarization**: Affinity matrix calculation scales quadratically $\mathcal{O}(M^2)$ where $M$ is the number of sliding-window embedding segments. Eigendecomposition scales $\mathcal{O}(M^3)$. Benchmarking will highlight where long audio files must be chunked or sub-sampled to prevent quadratic memory blowup.
* **ASR**: CTC decoding scales linearly $\mathcal{O}(T)$ with segment duration.
* **LLM**: Token generation scales with transcript length, but prompt processing scales with context size.

---

## 5. Benchmark Suite Architecture (`notebooks/01_system_benchmark.ipynb`)

The benchmark notebook is structured into executable cells that execute without manual interventions:
1. **Cell 1: Environment & Hardware Inspector**
   - Logs CPU architecture, core count, RAM capacity, GPU device name, PyTorch version, and CUDA/MPS availability.
2. **Cell 2: Benchmark Harness Utilities**
   - Defines a standardized `@benchmark_stage` context manager measuring precise timing, garbage collection, and peak VRAM.
3. **Cell 3: Isolated Stage Benchmarks**
   - Runs isolated 3-pass warm-up and timed inference across Stages 1, 2, 3, and 4 on standardized 30s and 60s clips.
4. **Cell 4: End-to-End Pipeline Latency Profiler**
   - Processes complete audio files and generates a stacked bar chart visualizing time-consumption share per stage.
5. **Cell 5: Memory Profile Waterfall**
   - Plots peak RAM and VRAM allocation over the lifecycle of an end-to-end pipeline run.
6. **Cell 6: Metric Export**
   - Writes `outputs/benchmarks/benchmark_summary.json` and renders markdown tables.

---

## 6. Definition of Done (DoD) for Benchmarking

- [x] Automated profiling harness (`BenchmarkProfiler`) measures wall-clock time, RAM, and GPU VRAM with zero synthetic data fabrication.
- [x] Profiling runs successfully across test audio lengths (15s, 30s, 60s, 120s).
- [x] Real-Time Factor (RTF) is computed and analyzed for every pipeline stage.
- [x] Bottleneck identification is clearly documented (identifying which stage dominates runtime).
- [x] `notebooks/01_system_benchmark.ipynb` executes cleanly from top to bottom, generating interactive visualizations and exported summary tables.
