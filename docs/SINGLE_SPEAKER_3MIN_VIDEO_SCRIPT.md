# "Who Spoke What and When" — 3-Minute Video Presentation Master Script
## Single-Speaker Walkthrough: Topic, 5-Stage Architecture, Live Demonstration & Final Results

**Target Duration**: Exactly **2 Minutes 50 Seconds** (leaving a 10-second safety margin before the strict 3:00 cutoff)  
**Presenter**: Single Speaker (Solo Presenter speaking on behalf of Team 14: Kushal, Vansh, Dorothy, and Pranav)  
**Course**: CS F407 Artificial Intelligence — Academic Year 2026–27 Semester I, BITS Pilani, K.K. Birla Goa Campus  
**Live Showcase URL**: [https://learnerkg23.github.io/Audio-Engineering/](https://learnerkg23.github.io/Audio-Engineering/)  
**Primary Repository**: [github.com/LearnerKg23/Audio-Engineering](https://github.com/LearnerKg23/Audio-Engineering)  
**Main Academic Base**: [github.com/vansh-bit/Audio-Engineering](https://github.com/vansh-bit/Audio-Engineering)

---

## ⏱️ Video Runtime & Timing Architecture

```
0:00 ──────────────── 0:25 ─────────────── 1:00 ──────────────────────── 2:15 ────────────── 2:45 ── 2:50
  │  Part 1: Topic   │   Part 2: Problem  │    Part 3: Complete Live     │  Part 4: Results   │ Outro│
  │  & What We Built │   & 5-Stage Arch   │    Audio & RAG Demonstration │  & Diagnostics     │ Wrap │
  │    (25 sec)      │      (35 sec)      │            (75 sec)          │     (30 sec)       │ (5s) │
```

| Segment | Timestamp | Duration | Section / Screen | Core Objective |
| :--- | :---: | :---: | :--- | :--- |
| **Part 1** | `0:00 – 0:25` | 25s | **Hero Landing Page** | Project topic, team introduction, and plain-English explanation of what we built. |
| **Part 2** | `0:25 – 1:00` | 35s | **Problem Statement & 5-Stage Architecture** | Single-channel mono constraint, cocktail party overlap, 5 decoupled stages, and downstream ASR proof. |
| **Part 3** | `1:00 – 2:15` | 75s | **Live Interactive Demonstration** | Stage 1 audio auditioning (mix vs stems), Stage 2 timeline, Stage 3 Hinglish transcript, Stage 4 RAG citation playback. |
| **Part 4** | `2:15 – 2:45` | 30s | **Results, Ablations & Breakdown Threshold** | Final metrics (+21.77 dB SI-SDR, 9.20% DER, 21.88% WER), +4.20 dB threshold, and real-world field robustness. |
| **Outro** | `2:45 – 2:50` | 5s | **Academic Footer & GitHub Link** | Professional sign-off and link reference. |

---

## 🎬 Master Teleprompter Script (Word-for-Word with Visual Cues)

---

### PART 1: Project Topic & What We Have Built
**Timestamp**: `0:00 – 0:25` (25 seconds)  
**Screen**: Hero Landing Page (`https://learnerkg23.github.io/Audio-Engineering/`)  
**Visual / Mouse Actions**:
- Start screen centered on the hero section.
- Hover mouse smoothly across the central soundwave graphic.
- Point briefly to the team attribution badge (`Project Team: Kushal • Vansh • Dorothy • Pranav`).
- At `0:23`, move cursor to the navigation tab: `[ Problem Statement & 5-Stage Architecture ]`.

> 🗣️ **SPOKEN SCRIPT** *(Calm, confident, engaging tone)*:
> 
> *"Hello everyone. Welcome to our presentation of **'Who Spoke What and When'**—an advanced computational speech engineering pipeline for overlapping, code-switched Indian conversational audio, built by Team 14: Kushal, Vansh, Dorothy, and Pranav.
> 
> In everyday Indian environments—like lively television panel debates, courtroom discussions, and rural helpline calls—people frequently interrupt each other while fluidly mixing Hindi and English. 
> 
> We have built a complete, end-to-end artificial intelligence system that takes this chaotic, single-channel audio, disentangles the overlapping voices into clean individual speaker stems, tracks who spoke at every millisecond, accurately transcribes their bilingual words, and extracts structured, hallucination-free conversational intelligence."*

---

### PART 2: The Problem Statement & The 5-Stage Architecture
**Timestamp**: `0:25 – 1:00` (35 seconds)  
**Screen**: Overview Tab (`#overview-mode`)  
**Visual / Mouse Actions**:
- **`0:25`**: Click tab `[ Problem Statement & 5-Stage Architecture ]`. Screen switches smoothly to the 4-quadrant overview.
- **`0:28 – 0:38`**: Point cursor to **Quadrant 1 (The Real-World Acoustic Problem)**, highlighting `Single-Channel Mono Constraint` and `Dynamic Code-Switching`.
- **`0:38 – 0:48`**: Glide cursor to **Quadrant 2 (End-to-End Pipeline Architecture)**, tracing down the 5 numbered badges from Stage 1 to Stage 5.
- **`0:48 – 0:56`**: Hover cursor over **Quadrant 4 (Downstream ASR Proof)**, contrasting the red `FAILED BASELINE` badge with the green `CLEAN RECOVERY` badge.
- **`0:57`**: **CLICK** the glowing orange button: `[ 🚀 Proceed to Live Working Audio & RAG Demonstration ↓ ]`.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Let's examine why this problem is notoriously difficult. In real-world audio, we have only a **single microphone channel**—meaning spatial beamforming is physically impossible. Furthermore, overlapping voices cause acoustic formants to collide, while dynamic Hinglish code-switching breaks standard monolingual speech models.
> 
> Rather than relying on a fragile black-box model, we engineered a mathematically decoupled **5-stage pipeline**:
> 1. **Time-domain Blind Speech Separation** using Conv-TasNet to isolate voices,
> 2. **Speaker Diarization** via pyannote ResNet embeddings and spectral clustering,
> 3. **Bilingual ASR** combining AI4Bharat IndicWav2Vec and Whisper with Unicode normalization,
> 4. **LLM Conversational Intelligence** powered by deterministic Pydantic schema contracts, and
> 5. **Diagnostic Field Validation** across All India Radio and Project Vaani phone calls.
> 
> As shown in Quadrant 4, feeding raw overlapping speech into Whisper causes catastrophic failure—an infinite repetition loop and 100% Word Error Rate. But with our front-end separation, word-for-word accuracy is fully restored. Let’s see this live in action!"*

---

### PART 3: Complete Live Working Audio & RAG Demonstration
**Timestamp**: `1:00 – 2:15` (75 seconds)  
**Screen**: Live Interactive DAW Workbench (`#workbench-mode`)  

#### 3A. Stage 1: Audio Separation DAW (`1:00 – 1:25`, 25s)
**Visual / Mouse Actions**:
- Screen automatically lands on **Stage 1: Speech Separation (BSS)**.
- **`1:02`**: Click **Play** on `Mix Track (Raw Input Audio)`. Let the overlapping audio play for 3–4 seconds.
- **`1:09`**: Click **Solo** on `Stem 1 (Male Hindi)`. The player seamlessly solos the isolated male voice.
- **`1:16`**: Click **Solo** on `Stem 2 (Female Hinglish)`. The player solos the clean female voice.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Here on our live interactive workbench, Stage 1 acts as an audio prism. 
> 
> Listen to this raw input recording where a male and female speaker talk simultaneously:  
> 🔊 `[Raw Mix plays: both voices clashing concurrently]`  
> 
> As you heard, their voices are completely collided. Now, let’s solo our separated **Stem 1**:  
> 🔊 `[Stem 1 plays: crisp, isolated Male Hindi speech without background bleed]`  
> 
> The male voice is cleanly extracted. And here is **Stem 2**:  
> 🔊 `[Stem 2 plays: pristine Female Hinglish speech]`  
> 
> Both speakers are disentangled with a massive **+21.8 dB SI-SDR gain**, with zero phase distortion."*

---

#### 3B. Stage 2: Speaker Diarization Timeline (`1:25 – 1:40`, 15s)
**Visual / Mouse Actions**:
- Scroll down smoothly to **Stage 2: Speaker Diarization**.
- Hover cursor across the interactive colored timeline showing `SPEAKER_00` and `SPEAKER_01` segment blocks.
- Point to the telemetry badges: `9.20% DER` and `0.006 RTF (166.7x speed)`.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Next, in Stage 2, our diarization engine answers 'Who spoke when'. Using ResNet-based voice embeddings and automated Eigengap spectral clustering, the system draws precise speaker turn boundaries. 
> 
> It operates at a blazing **0.006 Real-Time Factor**—over 160 times faster than real-time—achieving an exceptional **9.20% Diarization Error Rate**, even through rapid speaker handoffs."*

---

#### 3C. Stage 3: Bilingual Hinglish ASR (`1:40 – 1:55`, 15s)
**Visual / Mouse Actions**:
- Scroll to **Stage 3: Multilingual Hinglish ASR**.
- Hover over the synchronized dual-channel transcripts.
- Highlight the code-switched terms: *"rehab"* and *"fitness test"* embedded within Devanagari text.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"In Stage 3, each separated stem is transcribed using AI4Bharat IndicWav2Vec coupled with Unicode NFKC normalization.
> 
> Notice how the transcript captures both pure Hindi grammar and English technical vocabulary—correctly recognizing words like 'rehab' and 'fitness test' alongside Devanagari script. We achieve a **21.88% Word Error Rate**, completely eliminating the hallucination loops that plague standard Whisper models."*

---

#### 3D. Stage 4: Grounded RAG Conversational Chatbot (`1:55 – 2:15`, 20s)
**Visual / Mouse Actions**:
- Scroll to **Stage 4: LLM Structuring & Conversational Intelligence**.
- **`1:57`**: Click the quick-query pill: `What rehabilitation instructions were given?`.
- The assistant displays the answer with a clickable badge: `[SPEAKER_00 • 0.0s–5.7s ▶]`.
- **`2:04`**: **Click the timestamp citation badge**. The exact snippet plays aloud from that precise audio second.
- Point to the `Hallucination Rate: 0.00%` badge and Pydantic Contract 5 validation check.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Finally, Stage 4 turns raw audio transcripts into executive intelligence. We query our grounded RAG assistant: 'What rehabilitation instructions were given?'. 
> 
> Notice how the AI answers with an exact, clickable audio citation:  
> 🖱️ `[Click citation badge: audio instantly plays back the exact spoken sentence]`  
> 
> Backed by deterministic Pydantic Schema Contracts, our system achieves a **0.00% hallucination rate**—every single claim is mathematically verified against the recorded audio."*

---

### PART 4: Results, Ablation Studies & Breakdown Threshold
**Timestamp**: `2:15 – 2:45` (30 seconds)  
**Screen**: Stage 5 Diagnostics & Summary Dashboard (`#diagnostics`)  
**Visual / Mouse Actions**:
- Scroll down to **Stage 5: Diagnostics & Field Validation**.
- Hover cursor over the benchmark ablation cards:
  - `+21.77 dB SI-SDR` (Conv-TasNet vs Demucs)
  - `9.20% DER` (vs 33.3% raw mixture error)
  - `21.88% WER` (vs 100% collision collapse)
- Point to the critical threshold callout card: `Critical Acoustic Breakdown Threshold: +4.20 dB SI-SDR`.
- Point to the test suite indicator: `80/80 Tests Passing` on All India Radio and Project Vaani.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Let's summarize our core scientific findings in Stage 5:
> - **Separation**: Conv-TasNet achieved a **+21.77 dB SI-SDR gain**, outperforming Meta Hybrid Demucs while using 30% fewer parameters.
> - **Diarization**: We achieved a **9.20% DER**, representing a 33.3% relative error reduction over unseparated baselines.
> - **ASR Recovery**: Downstream Word Error Rate dropped from 100% attention collapse down to **21.88%**.
> - **Critical Acoustic Threshold**: We discovered a sharp non-linear phase transition at **+4.20 dB SI-SDR**. Above this threshold, downstream ASR operates flawlessly; below it, error rates spike exponentially.
> - Finally, our entire pipeline passed **80 out of 80 automated test suites**, confirming robustness across real-world All India Radio debates and Project Vaani telephone calls."*

---

### OUTRO: Academic Sign-Off
**Timestamp**: `2:45 – 2:50` (5 seconds)  
**Screen**: Page Footer & GitHub Link  
**Visual / Mouse Actions**:
- Scroll to clean bottom footer showing Team 14 attributions and the GitHub repository link.

> 🗣️ **SPOKEN SCRIPT**:
> 
> *"Our complete codebase, models, and interactive benchmark are fully open-source on GitHub. Thank you very much for your time!"*

---

## 📊 Summary of Verified Results & Key Metrics (Presenter Cheat-Sheet)

Keep these exact numbers memorized or visible on your secondary screen during recording:

| Metric | Measured Value | Baseline Comparison | Significance |
| :--- | :---: | :---: | :--- |
| **Separation Gain (SI-SDR)** | **$+21.77 \text{ dB}$** | $+10.74 \text{ dB}$ on real broadcast cross-talk | Disentangles overlapping voices with zero acoustic artifact distortion. |
| **Diarization Error Rate (DER)** | **$9.20\%$** | $33.3\%$ relative error reduction vs raw mixture | Identifies speaker turn handoffs even during rapid interruptions. |
| **Diarization Speed (RTF)** | **$0.006 \text{ RTF}$** | $166.7\times$ faster than real time | Enables instantaneous, sub-second timeline rendering. |
| **Downstream Bilingual ASR (WER)** | **$21.88\%$** | $100.0\%$ total attention collapse on raw mix | Restores broken Whisper ASR to production-grade transcription. |
| **LLM Hallucination Rate** | **$0.00\%$** | Guaranteed by Pydantic Contract 5 | Timestamped citations (`[SPEAKER_00 • 0.0s–5.7s ▶]`) prevent false claims. |
| **Critical Breakdown Threshold** | **$+4.20 \text{ dB}$** | Phase transition boundary | Discovered tipping point below which ASR degrades exponentially. |
| **Automated Test Suite** | **80 / 80 Passing** | Unit & integration tests | Rigorously verified across AIR debates and Project Vaani phone calls. |

---

## 🎙️ Recording & Delivery Tips for the Solo Speaker

1. **Audio Auditioning**:
   - In OBS Studio, ensure **Desktop Audio** is routed to your recording track so that when you click `Play` and `Solo` in Stage 1, the judges hear the actual separated audio loud and clear!
2. **Smooth Cursor Work**:
   - Don't rush your mouse movements. Move the cursor with deliberate, smooth motions to guide the viewer’s eye exactly where you want them to look.
3. **Pacing and Breathing**:
   - Speak at an energetic, conversational pace (~135 words per minute). 
   - Take a quick breath at the section transitions (`0:25`, `1:00`, and `2:15`).
4. **Browser Setup**:
   - Open Chrome or Edge in **Full Screen (`F11`)** at 1080p (`1920x1080`).
   - Hide the bookmarks bar (`Ctrl+Shift+B`) for a clean, professional aesthetic.
   - Set zoom to **100%** (our typography and contrast updates ensure crystal-clear legibility).

---

*Compiled and verified for CS F407 Academic Evaluation, BITS Pilani, K.K. Birla Goa Campus.*
