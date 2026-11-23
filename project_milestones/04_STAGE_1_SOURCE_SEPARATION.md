# 04 — Stage 1: Blind Source Separation (BSS)

## 1. Mathematical Formulation & Core Objective

The fundamental acoustic question addressed in Stage 1 is:
> *"Given an unconstrained single-channel or multi-channel audio mixture containing concurrent talkers and acoustic noise, can we mathematically isolate individual source waveforms without prior speaker profiles?"*

### Acoustic Mixture Formulation
Mathematically, the observed continuous-time acoustic signal $x(t)$ is modeled as a linear superposition:
$$x(t) = \sum_{i=1}^{K} s_i(t) + n(t)$$
where:
* $K$ is the number of active acoustic sources (concurrent speakers),
* $s_i(t)$ is the true time-domain signal emitted by speaker $i$,
* $n(t)$ is the additive ambient noise (environmental reverberation, microphone thermal noise, sensor hum).

In the discrete time domain with sampling rate $f_s = 16\text{ kHz}$:
$$\mathbf{x} = \sum_{i=1}^{K} \mathbf{s}_i + \mathbf{n}, \quad \mathbf{x}, \mathbf{s}_i, \mathbf{n} \in \mathbb{R}^N$$

The objective of the Blind Source Separation (BSS) module is to compute an estimator function $f_\theta(\mathbf{x})$ yielding estimated source vectors:
$$\{\hat{\mathbf{s}}_1, \hat{\mathbf{s}}_2, \dots, \hat{\mathbf{s}}_K\} = f_\theta(\mathbf{x})$$
such that each $\hat{\mathbf{s}}_i$ approximates a true source $\mathbf{s}_{\pi(i)}$ up to an arbitrary permutation $\pi \in \mathcal{P}_K$ and scale factor $\alpha \in \mathbb{R}$.

---

## 2. Model Architecture Evaluation & Selection

The proposal explicitly identifies two candidate deep learning architectures: **Meta's Demucs** and **Conv-TasNet**.

| Dimension | Meta's Demucs (Hybrid Demucs / HTDemucs) | Conv-TasNet (Convolutional Time-domain Audio Separation Network) | Architectural Decision & Rationale |
|---|---|---|---|
| **Domain Representation** | Dual: Hybrid Time-Frequency (Complex STFT + Raw Waveform U-Net with dilated convolutions and cross-attention). | Pure Time-Domain: 1D convolutional encoder/decoder with temporal convolutional network (TCN) blocks. | **Primary Choice: Demucs (`htdemucs` / `demucs_extra`)**.<br>Demucs demonstrates superior acoustic realism on complex non-stationary audio, handles broader frequency spectra without harsh phase artifacts, and has mature PyTorch packaging. |
| **Acoustic Artifacts** | Low musical noise; preserves natural vocal timbres and formant transitions. | Prone to metallic ringing and phase distortion at higher frequencies. | Demucs provides cleaner signals for downstream speaker embedding extraction. |
| **Separation Quality** | High separation fidelity on polyphonic and choral mixtures. | High SDR on clean benchmark datasets (e.g., WSJ0-2mix); degrades on real-world noisy backgrounds. | Demucs is more resilient to ambient room reflections and background babble. |
| **Inference Overhead** | Moderate to high memory footprint (requires approx. 1.2–2.5 GB VRAM). | Very lightweight; ultra-fast inference suitable for real-time mobile DSP. | We evaluate Demucs as our primary engine, and implement a benchmark stub for Conv-TasNet for comparative computational profiling. |

---

## 3. Evaluation Metrics: Mathematical Rigor

### Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
Traditional Signal-to-Noise Ratio (SNR) or Mean Squared Error (MSE) is sensitive to arbitrary gain changes. An estimated signal that is acoustically identical to the reference but shifted in volume by $3\text{ dB}$ would be heavily penalized by MSE. 

**SI-SDR** resolves this by projecting the estimated signal onto the target signal to find the optimal scale factor $\alpha$:
$$\mathbf{s}_{\text{target}} = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2} \mathbf{s}$$
$$\mathbf{e}_{\text{noise}} = \hat{\mathbf{s}} - \mathbf{s}_{\text{target}}$$
$$\text{SI-SDR}(\mathbf{s}, \hat{\mathbf{s}}) = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2} \right)$$
*Interpretation*: Higher values (in decibels, dB) indicate superior separation. A separation score of $\ge +10\text{ dB}$ represents high separation fidelity; $5\text{ to }10\text{ dB}$ represents moderate separation; $\le 0\text{ dB}$ indicates severe cross-talk or destructive interference.

### The Permutation Problem & Permutation Invariant Training (PIT)
Neural separation models output channels in arbitrary order:
$$\text{Model Output: } [\hat{\mathbf{s}}_1, \hat{\mathbf{s}}_2] \quad \text{vs. Ground Truth: } [\mathbf{s}_A, \mathbf{s}_B]$$
If $\hat{\mathbf{s}}_1$ corresponds to $\mathbf{s}_B$ and $\hat{\mathbf{s}}_2$ corresponds to $\mathbf{s}_A$, a naive pairwise comparison ($\hat{\mathbf{s}}_1 \leftrightarrow \mathbf{s}_A$) will register a catastrophic error.

To solve this, our evaluation engine implements **Permutation Invariant Matching**:
$$\text{Optimal Permutation } \pi^* = \arg\max_{\pi \in \mathcal{P}_K} \sum_{i=1}^K \text{SI-SDR}(\mathbf{s}_i, \hat{\mathbf{s}}_{\pi(i)})$$
where $\mathcal{P}_K$ is the set of all $K!$ permutations (for $K=2$, $\mathcal{P}_2 = \{(1,2), (2,1)\}$). All reported metrics use $\pi^*$.

---

## 4. Controlled Mixture Generation Pipeline

Because real-world broadcasts do not provide isolated clean stems, we construct a calibrated synthetic evaluation harness:

```
Pristine Speaker A (IndicVoices) ──┐
                                  ├──> Synthesizer (Gain $\alpha$, Overlap %, Additive Noise $\mathbf{n}$) ──> Controlled Mixture $\mathbf{x}$
Pristine Speaker B (Nirantar)   ──┘
```

### Parameterized Variables for Difficulty Experiments:
1. **Overlap Ratio ($\Omega$)**:
   - $0\%$ (Strict turn-taking / sequential dialogue)
   - $25\%$ (Natural interjection at speaker boundaries)
   - $50\%$ (Heated conversational argument)
   - $75\%$ (Concurrent shouting / simultaneous speech)
2. **Signal-to-Interference Ratio (SIR)**:
   - $0\text{ dB}$ (Equal loudness between talkers)
   - $+6\text{ dB}$ (Speaker A dominant)
   - $-6\text{ dB}$ (Speaker B dominant, Speaker A quiet)
3. **Additive Background Noise (Stationary vs. Non-Stationary)**:
   - Clean ($\text{SNR} = \infty$)
   - Moderate environmental noise ($\text{SNR} = +15\text{ dB}$)
   - Severe field noise ($\text{SNR} = +5\text{ dB}$, market babble / rural machinery)

---

## 5. Implementation Architecture (`src/separation/`)

```python
class BaseSeparator(ABC):
    """Abstract interface for all speech separation modules."""
    
    @abstractmethod
    def separate(self, audio_path: Path) -> SeparationResult:
        """Ingests mixed audio file and returns paths to isolated tracks."""
        pass

class DemucsSeparator(BaseSeparator):
    """Production wrapper for Meta's Demucs separation model."""
    def __init__(self, model_name: str = "htdemucs", device: str = "cpu"):
        self.model = load_pretrained_demucs(model_name, device=device)
        self.device = device
        
    def separate(self, audio_path: Path) -> SeparationResult:
        # 1. Load audio, resample to model rate (44.1k or 16k)
        # 2. Forward pass through Hybrid U-Net
        # 3. Post-process separated stems, resample to 16kHz PCM
        # 4. Energy thresholding to prune silent/phantom channels
        # 5. Return structured SeparationResult
        pass
```

---

## 6. Definition of Done (DoD) for Stage 1

Stage 1 is complete if and only if all following conditions are verified:
- [x] Pre-trained Demucs model runs deterministically without throwing shape mismatch or CUDA/MPS memory faults.
- [x] Synthetic mixture generator creates reproducible test pairs with verifiable ground truth.
- [x] SI-SDR metric calculation incorporates permutation-invariant matching ($\pi^*$).
- [x] Controlled experiment results table is generated across 3 overlap conditions ($25\%, 50\%, 75\%$) and 2 noise levels.
- [x] Qualitative test run executes on at least 1 real-world All India Radio (AIR-RS-DB) debate snippet.
- [x] Latency, processing time, RTF, RAM, and GPU VRAM benchmarks are recorded.
- [x] Output audio files are saved in strict 16 kHz 16-bit mono `.wav` format.
