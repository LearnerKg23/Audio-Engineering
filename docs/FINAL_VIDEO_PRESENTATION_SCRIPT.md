# 🎬 3-Minute Academic Video Presentation Script
## "Who Spoke What and When: Deep Multistage Speech Disentanglement and Conversational Intelligence for Overlapping Code-Switched Audio"

---

### 📋 Production & Compliance Overview
* **Course**: CS F407 Artificial Intelligence, AY 2026–27 Semester I
* **Institution**: BITS Pilani, K.K. Birla Goa Campus
* **Team**: Team 14
* **Authors**: 
  - Kushal Gupta (2022A7PS0023G) — *Interactive Web Hub, RAG Chatbot, Indic ASR, Benchmarking*
  - Vansh Sharma (2022A7PS0042G) — *Mathematical Formulation, BSS, Diarization, Backend Engine*
  - Dorothy (2022A7PS0110G), Pranav (2022A7PS0188G)
* **Strict Time Constraint**: **3 Minutes (180 Seconds) Maximum**  
  > *Evaluator Note: "If your video is longer, we will watch and score only the first 3 minutes."*
* **Target Recorded Runtime**: **2 Minutes 48 Seconds (168 Seconds)** *(Leaves a 12-second safety buffer)*
* **Total Spoken Word Count**: **377 words** (~138 words per minute + live audio playback breaks)
* **Live Demo URL**: [https://learnerkg23.github.io/Audio-Engineering/](https://learnerkg23.github.io/Audio-Engineering/)
* **Codebase & Tests**: [https://github.com/LearnerKg23/Audio-Engineering](https://github.com/LearnerKg23/Audio-Engineering) *(84/84 Unit & Integration Tests Passing)*

---

## ⏱️ Video Timeline & Scene Breakdown

| Segment | Timestamp | Duration | Section | Media Type | Primary Speaker |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Scene 1** | `0:00 – 0:25` | 25s | Problem Statement & Real-World Indian Audio | Title Slide + Acoustic Waveforms | **Kushal Gupta** |
| **Scene 2** | `0:25 – 0:50` | 25s | Modular 5-Stage Architecture & Formulation | Architecture Flowchart Slide | **Vansh Sharma** |
| **Scene 3** | `0:50 – 1:55` | 65s | **Live Interactive Showcase & RAG Demonstration** | Screen Capture of Web Application | **Kushal Gupta** |
| **Scene 4** | `1:55 – 2:35` | 40s | Quantitative Results & The Breakdown Threshold | Benchmark Tables & Scatter Plots | **Vansh Sharma** |
| **Scene 5** | `2:35 – 2:48` | 13s | Summary, GitHub Repository & Conclusion | Summary Slide + QR Code | **Kushal & Vansh** |
| **Buffer** | `2:48 – 3:00` | 12s | Hard Stop Buffer Before 3:00 Cutoff | — | — |

---

## 🎥 Word-for-Word Production Script

```
====================================================================================================
SCENE 1: THE REAL-WORLD PROBLEM & MOTIVATION
Duration: 0:00 – 0:25 (25 Seconds) | Word Count: 55 Words | Speaker: Kushal Gupta
====================================================================================================
```

#### 🖥️ Visual Track:
* **0:00 – 0:08**: Display **Slide 1: Title Slide**. Clean typography showing:
  - Project Title: *"Who Spoke What and When: Deep Multistage Speech Disentanglement and Conversational Intelligence for Overlapping Code-Switched Audio"*
  - Course: *CS F407 Artificial Intelligence | BITS Pilani, K.K. Birla Goa Campus*
  - Team 14: *Kushal Gupta, Vansh Sharma, Dorothy, Pranav*
* **0:08 – 0:25**: Transition to split-screen graphic: On the left, an overlapping acoustic waveform showing cross-talk; on the right, bulleted acoustic realities:
  - *Severe Overlapping Speech ($\Omega = 25\%\text{--}75\%$)*
  - *Dynamic Hindi-English (Hinglish) Code-Switching*
  - *Band-limited Telephony & Ambient Noise ($300\text{--}3400\text{ Hz}$)*

#### 🎙️ Voiceover (Kushal Gupta):
> *"Hello everyone. Monolithic speech recognition systems like Whisper and Google STT degrade catastrophically on spontaneous Indian speech.*
>
> *In community radio broadcasts, panel debates, and agricultural helplines, speakers constantly talk over each other, switch dynamically between Hindi and English, and record over noisy telephony channels.*
>
> *Our project resolves this by answering: **'Who Spoke What and When.'**"*

---

```
====================================================================================================
SCENE 2: SYSTEM ARCHITECTURE & MATHEMATICAL DECOUPLING
Duration: 0:25 – 0:50 (25 Seconds) | Word Count: 58 Words | Speaker: Vansh Sharma
====================================================================================================
```

#### 🖥️ Visual Track:
* **0:25 – 0:50**: Display **Slide 2: 5-Stage Pipeline Architecture Diagram**.
  - Animated highlight moving progressively across 5 blocks:
    1. *Stage 0: Audio Conditioning & RMS Normalization*
    2. *Stage 1: Conv-TasNet / Hybrid Demucs BSS with Hungarian PIT*
    3. *Stage 2: PyAnnote Neural Embeddings & Eigengap Spectral Clustering*
    4. *Stage 3: IndicWav2Vec Bilingual ASR with Unicode Normalization*
    5. *Stage 4 & 5: Guardrailed Meeting Intelligence & Extractive RAG*

#### 🎙️ Voiceover (Vansh Sharma):
> *"To solve this, we decouple acoustic disentanglement from linguistic decoding across five stages.*
>
> *Raw mixtures are conditioned and separated in the time domain using Conv-TasNet with Hungarian Permutation Invariant Training.*
>
> *The isolated stems undergo hyperspherical neural embedding and automated Eigengap spectral clustering for diarization, followed by AI4Bharat IndicWav2Vec for bilingual transcription, and Pydantic-guardrailed LLM meeting intelligence."*

---

```
====================================================================================================
SCENE 3: LIVE WORKING DEMONSTRATION & RAG CHATBOT
Duration: 0:50 – 1:55 (65 Seconds) | Word Count: 140 Words | Speaker: Kushal Gupta
Screen: Live Web Showcase (learnerkg23.github.io/Audio-Engineering/)
====================================================================================================
```

#### 🖥️ Visual Track & Screen Actions:
* **0:50 – 1:05**: Browser in full view showing the live interactive web hub.
  - Mouse hovers over **Track A (Controlled 2-Speaker Hinglish Mix)**.
  - Mouse clicks **"▶ Play Raw Mixture"** button.
  - **[AUDIO CUE 1 (0:54 – 0:57)]**: Let raw audio play for **3 seconds**. Two speakers talk over each other in severe cross-talk.
  - Mouse clicks **Pause**.
* **1:05 – 1:20**: Mouse moves down to the **Multi-Track Separation DAW**.
  - Mouse clicks **"▶ Solo Stem 1"** (Female Hinglish).
  - **[AUDIO CUE 2 (1:07 – 1:10)]**: Let Stem 1 play for **3 seconds** *(Clean female voice talking about fitness test)*.
  - Mouse clicks **"▶ Solo Stem 2"** (Male Hindi).
  - **[AUDIO CUE 3 (1:13 – 1:16)]**: Let Stem 2 play for **3 seconds** *(Clean male voice talking about the AI dataset)*.
  - Point mouse cursor to show zero cross-talk bleed.
* **1:20 – 1:35**: Mouse scrolls smoothly past **Stage 2 Diarization Timeline** and **Stage 3 Indic Transcripts**.
  - Highlight the synchronized dual-channel speaker pills: `[0.00s–5.72s: SPEAKER_00]` and `[3.80s–8.40s: SPEAKER_01]`.
* **1:35 – 1:55**: Mouse scrolls to **Stage 4: Structured LLM Intelligence**.
  - Mouse clicks the 4th tab: **`💬 Ask LLM (Interactive RAG)`**.
  - Mouse clicks the suggested query chip: **`"What was said about rehab?"`**.
  - The answer immediately renders: *"SPEAKER_00 stated that 10 days of rehabilitation are required before taking the fitness test."*
  - **CRITICAL DEMO ACTION**: Mouse clicks the interactive audio deep-linking citation pill: **`[SPEAKER_00 • 0.0s–5.7s ▶]`**.
  - **[AUDIO CUE 4 (1:44 – 1:47)]**: Audio player instantly seeks and plays the exact citation segment!
  - Mouse types in input box: *"What is the capital of France?"* and hits enter.
  - Camera zooms in slightly as the negative guardrail appears:
    `[🛡️ Guardrail Triggered: This entity or topic is not mentioned in the source audio transcript. Speculation prevented.]`

#### 🎙️ Voiceover (Kushal Gupta):
> *(0:50)* *"Here is our live demonstration hub deployed on GitHub Pages.*
>
> *(0:53)* *First, listen to this raw overlapping mixture on Track A:*  
> *(Audio plays: 3 seconds of entangled voices)*  
>
> *(0:58)* *As you heard, both voices overlap concurrently.*
>
> *(1:02)* *Now, we audition our isolated stems:*  
> *(Audio plays: 3 seconds of clean Female Stem 1)*  
> *(Audio plays: 3 seconds of clean Male Stem 2)*  
>
> *(1:16)* *Cross-talk is completely suppressed.*
>
> *(1:19)* *Next, Stage 2 and 3 produce precise millisecond timelines and bilingual Devanagari transcripts.*
>
> *(1:27)* *Finally, in Stage 4, our Conversational RAG engine allows users to interrogate the meeting.*
>
> *(1:35)* *Clicking 'What was said about rehab?' retrieves the grounded answer. Clicking the citation pill instantly seeks and plays that exact audio stem.*
>
> *(1:48)* *And if we ask an out-of-domain question, our strict negative guardrails prevent speculation with zero percent hallucination."*

---

```
====================================================================================================
SCENE 4: QUANTITATIVE EVALUATION & BREAKDOWN THRESHOLD
Duration: 1:55 – 2:35 (40 Seconds) | Word Count: 92 Words | Speaker: Vansh Sharma
====================================================================================================
```

#### 🖥️ Visual Track:
* **1:55 – 2:15**: Display **Slide 3: Quantitative Performance Benchmarks**.
  - Top Table: Metric comparison table across synthetic mixtures and authentic field recordings:
    - *SI-SDR Improvement: $\mathbf{+21.80\text{ dB}}$*
    - *Diarization Error Rate (DER): $40.52\% \rightarrow \mathbf{23.97\%}$ (Hungarian matched)*
    - *Word Error Rate (WER): $86.28\% \rightarrow \mathbf{51.57\%}$ (IndicWav2Vec)*
    - *Hallucination Rate: $\mathbf{0.00\%}$ (Pydantic Contract 5)*
    - *Real-Time Factor (RTF): $\mathbf{0.49\times\text{--}0.67\times}$*
* **2:15 – 2:35**: Slide switches to the **Empirical Breakdown Threshold Plot** (SI-SDR vs Downstream WER scatter plot with regression curve $r = -0.977$).
  - Red dashed line highlighting the critical cliff at **$\text{SI-SDR}_{\text{crit}} = +4.20\text{ dB}$**.

#### 🎙️ Voiceover (Vansh Sharma):
> *"Quantitatively, across controlled mixtures and All India Radio field data, our pipeline achieves an average **+21.80 dB** SI-SDR separation gain.*
>
> *Decoupling separation reduces Diarization Error Rate from 40.5% down to **23.97%**, and cuts downstream Word Error Rate from 86.3% to **51.57%**.*
>
> *Crucially, our ablation study uncovered an empirical acoustic breakdown threshold at **+4.20 dB SI-SDR**. Below this boundary, downstream phoneme decoding collapses catastrophically.*
>
> *Finally, our end-to-end pipeline executes at **0.49 to 0.67 times real-time**, enabling high-throughput speech intelligence."*

---

```
====================================================================================================
SCENE 5: SUMMARY, REPOSITORY & CONCLUSION
Duration: 2:35 – 2:48 (13 Seconds) | Word Count: 32 Words | Speaker: Kushal Gupta (or Both)
====================================================================================================
```

#### 🖥️ Visual Track:
* **2:35 – 2:48**: Display **Slide 4: Conclusion & Project Deliverables**.
  - Clean card showing:
    - *GitHub Repository*: `github.com/LearnerKg23/Audio-Engineering`
    - *Live Web Demo*: `learnerkg23.github.io/Audio-Engineering`
    - *Verification Badge*: `84/84 Pytest Suites Passing`
    - *QR Code linking directly to the live demo*
  - Team 14 photo or student IDs: *Kushal Gupta, Vansh Sharma, Dorothy, Pranav*

#### 🎙️ Voiceover (Kushal Gupta):
> *"Our complete open-source codebase, passing all 84 test suites, and our live interactive demonstration are publicly available on GitHub.*
>
> *Thank you for your time, and we look forward to your questions."*

```
====================================================================================================
[2:48 – 3:00] VIDEO COMPLETE (12-Second Buffer before the strict 3:00 evaluation cutoff)
====================================================================================================
```

---

## 📊 Word Count & Timing Budget Audit

| Scene | Target Timestamp | Target Seconds | Spoken Words | Audio / Interaction Stems | Pacing (WPM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scene 1 (Problem)** | `0:00 – 0:25` | 25s | 55 words | 0s | 132 wpm |
| **Scene 2 (Architecture)**| `0:25 – 0:50` | 25s | 58 words | 0s | 139 wpm |
| **Scene 3 (Live Demo)** | `0:50 – 1:55` | 65s | 140 words | 13s (4 audio cues) | 136 wpm (excluding audio) |
| **Scene 4 (Metrics)** | `1:55 – 2:35` | 40s | 92 words | 0s | 138 wpm |
| **Scene 5 (Conclusion)** | `2:35 – 2:48` | 13s | 32 words | 0s | 147 wpm |
| **Total** | `0:00 – 2:48` | **168s (2m 48s)** | **377 words** | **13s audio** | **~138 wpm (Optimal)** |

---

## 🛠️ Recording & Production Checklist

### 1. Screen & Recording Settings
* **Resolution**: 1920 × 1080 (Full HD) at 60 FPS.
* **Recording Software**: OBS Studio or macOS QuickTime Screen Recorder.
* **Audio Routing**:
  - Voice Microphone: Crisp, close-proximity USB mic (or headset) with gentle noise gate.
  - System Audio: Loopback / Desktop Audio captured directly in OBS so that the browser audio stems are loud and crystal clear (mix audio at -6 dBFS; duck voice when audio stems play).

### 2. Browser Preparation (for Scene 3)
* Open Chrome / Edge at `https://learnerkg23.github.io/Audio-Engineering/`.
* Set browser zoom to **100%** (or **110%** if high-DPI font scaling is needed).
* Ensure browser extensions and bookmark bar are hidden (`F11` or full-screen view).
* Pre-test audio playback to ensure no network caching delays.

### 3. Slide Deck Preparation (4 Slides)
* **Slide 1**: Title, Team Members, Acoustic Challenge wave diagram.
* **Slide 2**: 5-Stage Architecture Flowchart (Conv-TasNet $\rightarrow$ PyAnnote $\rightarrow$ IndicWav2Vec $\rightarrow$ Pydantic $\rightarrow$ RAG).
* **Slide 3**: Quantitative Results Table (+21.80 dB SI-SDR, 23.97% DER, 51.57% WER, 0.0% Hallucination) and the Scatter Plot with $+4.20\text{ dB}$ critical threshold.
* **Slide 4**: Conclusion, GitHub URL, Live Demo QR Code, 84/84 Passing Tests Badge.

---

## 🎙️ Alternative Single-Presenter Adaptation
*If only one team member is recording the voiceover, read all sections continuously without pause. The pacing and word counts remain 100% identical.*
