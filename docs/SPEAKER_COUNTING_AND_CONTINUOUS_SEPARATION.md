# Speaker Counting & Continuous Speech Separation (CSS)
**Document**: Architectural Deep-Dive on Dynamic Speaker Counting and Real-World Long-Form Audio  
**System Target**: Multi-Speaker Indian Conversational Audio (AIR Radio Debates, Agricultural Phone-Ins, Panel Discussions)  
**Location**: `docs/SPEAKER_COUNTING_AND_CONTINUOUS_SEPARATION.md`  

---

## 1. Executive Summary

A fundamental challenge in real-world computational speech engineering is the **"Chicken-and-Egg" dilemma** between Source Separation and Speaker Diarization:

* To **separate** voices cleanly, traditional neural separation models assume a fixed number of speakers (e.g., exactly 2 talkers).
* To **count** how many unique people spoke, you need to analyze their individual voice embeddings without the audio being contaminated by overlapping cross-talk.

In real-world Indian broadcasts (like All India Radio debate shows or rural phone-ins), you do not know the speaker count in advance. You might have 1 speaker (the news anchor), 2 speakers (a natural interview), or 3+ speakers (a heated debate with frequent interruptions and filler words).

This document explains:
1. **The Problem**: Why deep separation networks (like Conv-TasNet) cannot dynamically count speakers.
2. **The Solution**: How **Stage 2 (Diarization)** uses the **Eigengap Heuristic** to discover the true speaker count $K$.
3. **The Framework**: How **Continuous Speech Separation (CSS)** uses sliding windows to separate and stitch real-world multi-party audio of arbitrary length.

---

## 2. The Core Problem: The Fixed-Channel Bottleneck

### Why Separation Models Cannot Count Speakers
Deep generative separation modelsâ€”such as **Conv-TasNet**, **Demucs**, and **SepFormer**â€”are designed with fixed neural network output heads:

```
[Input: 1 Mixed Audio Channel] â”€â”€â–º [Encoder + Separation Masks] â”€â”€â–º [Fixed Output Heads: Channel 0, Channel 1]
```

When a model is compiled and trained (for example, on Libri2Mix), its final 1D convolutional decoder layer has a hard-coded output tensor shape:
`[Batch Size, Num Channels = 2, Audio Samples]`

Because the architecture has exactly 2 output channels, it faces severe degradation when deployed in unconstrained real-world environments:

| Real-World Scenario | Model Design | What Actually Happens (The Failure Mode) |
|---|---|---|
| **Single Speaker Talking**<br>*(e.g., Solo Newsreader)* | Model expects 2 speakers | **Over-Separation / Phantom Voice**: The model is forced to fill 2 output channels. It often splits one person's voice into unnatural high-pitch and low-pitch fragments, creating synthetic acoustic artifacts. |
| **Two Speakers Overlapping**<br>*(e.g., Host + 1 Guest)* | Model expects 2 speakers | **Optimal Case**: Model cleanly separates Speaker 0 and Speaker 1 (+15.45 dB SI-SDR gain). |
| **Three or More Speakers**<br>*(e.g., 3-Person Debate Panel)* | Model expects 2 speakers | **Forced Channel Merging**: The model has only 2 slots for 3 voices. It cleanly isolates one speaker into Channel 0, but forces the other two speakers into Channel 1, resulting in unseparated residual cross-talk. |

**Key Takeaway**: A pure separation model does not possess "identity awareness." It only learns local harmonic masking filters. It cannot answer: *"How many unique people are in this recording?"*

---

## 3. The Solution Part 1: Automated Speaker Count Discovery via Diarization

To solve the speaker count problem, we pass the speech through **Stage 2: Acoustic Speaker Diarization** using `pyannote.audio` and **Spectral Clustering**.

```
[Raw Audio Stream] 
        â”‚
        â–¼
[Voice Activity Detection (VAD)] â”€â”€â–º Strips background silence and room rumble
        â”‚
        â–¼
[Deep Embedding Extractor]       â”€â”€â–º Slices speech into 1.5s frames; maps each to a 
                                     512-dimensional voice identity vector (e)
        â”‚
        â–¼
[Cosine Affinity Matrix (A)]     â”€â”€â–º Measures similarity between all voice vectors
        â”‚
        â–¼
[Normalized Graph Laplacian (L)] â”€â”€â–º Computes eigenvalue spectrum
        â”‚
        â–¼
[Eigengap Heuristic]             â”€â”€â–º Mathematically discovers optimal speaker count K*
```

---

### The Mathematical Formula: The Eigengap Heuristic

How does the math automatically decide whether there are 2, 3, or 4 speakers?

#### Step 1: Construct the Affinity Matrix $A$
For every pair of extracted voice embeddings $\mathbf{e}_i$ and $\mathbf{e}_j$, we compute the normalized cosine similarity:

$$A_{ij} = \max\left(0, \frac{\mathbf{e}_i^\top \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}\right)$$

If frame $i$ and frame $j$ are spoken by the same human vocal tract, their angle is small and $A_{ij} \approx 1.0$. If spoken by different people, $A_{ij} \approx 0.0$.

#### Step 2: Compute the Normalized Graph Laplacian $L$
From the affinity matrix $A$ and degree matrix $D$ (where $D_{ii} = \sum_j A_{ij}$):

$$L = D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

#### Step 3: Spectral Decomposition & The Eigengap
We calculate the sorted eigenvalues of the Laplacian matrix:

$$\lambda_1 \le \lambda_2 \le \lambda_3 \le \dots \le \lambda_N$$

According to spectral graph theory, if the recording contains $K$ distinct, well-separated human speakers, the first $K$ eigenvalues will be very close to $0$, and there will be a **sudden, dramatic jump** (an "eigengap") between eigenvalue $\lambda_K$ and eigenvalue $\lambda_{K+1}$.

The optimal speaker count $K^*$ is determined by finding the maximum gap:

$$K^* = \arg\max_{k} \left( \lambda_{k+1} - \lambda_k \right)$$

```
Eigenvalue Magnitude
     â”‚
     â”‚                                     â—  (Î»4 = 0.88)
     â”‚                               â—        (Î»3 = 0.82)
     â”‚                     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
     â”‚                     â”‚  â˜… MAXIMUM EIGENGAP (Gap = 0.74)
     â”‚                     â”‚  Indicates exactly K = 2 true clusters!
     â”‚                     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
     â”‚           â—  (Î»2 = 0.08)
     â”‚     â—        (Î»1 = 0.01)
     â””â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Eigenvalue Index
           1     2     3     4     5
```

By computing this gap, the pipeline discovers the true number of speakers **completely unsupervised**, without any human setting the number in advance.

---

## 4. The Solution Part 2: Continuous Speech Separation (CSS)

Once we understand that real recordings have an arbitrary number of speakers and irregular pauses, how do we run separation on a 10-minute radio debate?

### The Acoustic Reality of Multi-Party Speech
In a real 3-person or 4-person radio debate:
* Rarely do all 3 or 4 people shout at the exact same split-second.
* Overlapping speech is almost always **pairwise** (Speaker A is talking, and Speaker B chimes in for 1â€“2 seconds to interrupt).
* Overlap typically accounts for **10% to 25%** of total conversational time.

### The 4-Step CSS Workflow

```
[Long-Duration Continuous Audio: 10 Minutes]
                         â”‚
                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ STEP 1: SLIDING WINDOW CHUNKING                                  â”‚
â”‚ Slice audio into 2.5-second windows with 1.25-second step (50%  â”‚
â”‚ overlap) using AudioPreprocessor.sliding_window_segmentation.   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â”‚
                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ STEP 2: LOCAL 2-CHANNEL SEPARATION                               â”‚
â”‚ Run Conv-TasNet on each 2.5s window. Within any tiny 2.5s chunk, â”‚
â”‚ at most 2 people are active simultaneously.                     â”‚
â”‚ Output: Channel 0 and Channel 1 for this local window.          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â”‚
                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ STEP 3: VOICE EMBEDDING STITCHING                                â”‚
â”‚ Extract pyannote voice embedding on Channel 0 and Channel 1.    â”‚
â”‚ Match embeddings against the global speaker clusters (from      â”‚
â”‚ Diarization) using cosine similarity.                           â”‚
â”‚ â€¢ Local Channel 0 matches Speaker A cluster â”€â”€â–º Assign to Spk A â”‚
â”‚ â€¢ Local Channel 1 matches Speaker C cluster â”€â”€â–º Assign to Spk C â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â”‚
                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ STEP 4: OVERLAP-ADD WAVEFORM RECONSTRUCTION                      â”‚
â”‚ Smooth window boundaries using a Hann window taper. Stitch      â”‚
â”‚ matched audio chunks into K separate, continuous audio tracks!  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 5. Real-World Stress Factors in Indian Conversational Audio

When stress-testing on uncurated corpora like **`AIR-RS-DB`**, **`Nirantar`**, or **`Project Vaani`**, three acoustic anomalies occur that clean lab datasets never test:

### 1. Extreme Loudness Disparities (Wide SIR Range)
* **The Scenario**: In a phone-in program, the in-studio host speaks into a calibrated Shure SM7B broadcast microphone at $-14\text{ dBFS}$, while a rural farmer calls in over a mobile cellular connection at $-32\text{ dBFS}$ (an extreme Signal-to-Interference Ratio of $+18\text{ dB}$).
* **The Failure**: Standard separation algorithms focus 99% of their masking energy on the loud host, treating the faint caller as background room noise and erasing them.
* **Our Pipeline Defense**: 
  Our `AudioPreprocessor.normalize_loudness` applies dynamic range leveling and peak limiting before separation, boosting the weak caller's formants so the separation mask can isolate both speakers.

### 2. Conversational Disfluencies & Filler Words
* **The Scenario**: In spontaneous Hinglish dialogue, speakers frequently use short hesitation particles:
  *"...toh humein **matlab**... **uh**... **arre yaar**... kal jaana chahiye..."*
* **The Failure**: Crude Voice Activity Detectors (VAD) interpret a 150 ms pause as the end of a sentence, chopping one continuous speaker turn into 5 fragmented audio splinters.
* **Our Pipeline Defense**:
  Our energy VAD in `AudioPreprocessor.energy_vad` implements **temporal hangover hysteresis**:
  * Minimum speech duration ($T_{\text{min\_speech}} = 200\text{ ms}$) ignores brief microphone pops.
  * Minimum silence bridge ($T_{\text{hangover}} = 250\text{ ms}$) bridges short pauses between words, preserving whole sentences.

### 3. Irregular Turn Cadence & Background Chatter
* **The Scenario**: Guests talk in bursts, interrupt each other for 0.8 seconds, and then both fall silent while a news jingle plays.
* **Our Pipeline Defense**:
  By using **VAD-guided gating**, we only trigger the heavy Conv-TasNet separation module when overlapping energy is actively detected. If only one person is speaking, the audio bypasses separation and passes directly to ASR, cutting computational overhead by **60% to 70%**.

---

## 6. How to Explain This to Your Professor

If your evaluator asks:
> *"What happens if there are 3 or 4 speakers in a real radio debate? How does your 2-channel separator handle that?"*

### Your Model Answer:
> *"That is a fundamental limitation of fixed-head separation models like Conv-TasNet or Demucsâ€”they cannot count speakers on their own.*
>
> *In our 4-stage architecture, we solve this by coupling Stage 1 with **Stage 2 Diarization**:*
> 1. *We use **`pyannote.audio` neural embeddings** and the **Eigengap Heuristic** on the graph Laplacian to automatically discover the true number of speakers ($K$).*
> 2. *For real-world continuous audio, we employ **Continuous Speech Separation (CSS)**: we process speech in 2.5-second sliding windows. Because speech overlap in natural debates is overwhelmingly pairwise (two people interrupting at a time), Conv-TasNet cleanly isolates the active pair locally, and Diarization embeddings stitch the isolated chunks into continuous, speaker-consistent tracks across the entire recording.*
>
> *This allows our system to scale from a clean 2-person interview up to an unconstrained multi-party radio debate."*

---

## 7. Document Reference Summary

* **Master Architecture & Benchmarks**: [`docs/FIRST_EVALUATION_MASTER_GUIDE.md`](docs/FIRST_EVALUATION_MASTER_GUIDE.md)
* **2-Slide Presentation Deck**: [`docs/flash_talk_slides.md`](docs/flash_talk_slides.md)
* **Timed Presenter Script**: [`docs/flash_talk_script.md`](docs/flash_talk_script.md)
* **Audio Preprocessor & Windowing Code**: [`src/preprocessing/audio_preprocessor.py`](src/preprocessing/audio_preprocessor.py)
* **Separation Models Code**: [`src/models/separation.py`](src/models/separation.py)

