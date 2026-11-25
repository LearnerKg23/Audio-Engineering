# 🎬 Interactive Website 3-Minute Video Presentation Guide & Script
## "Who Spoke What and When: Deep Multistage Speech Disentanglement and Conversational Intelligence for Overlapping Code-Switched Audio"

---

### 📋 Production & Compliance Overview
* **Course**: CS F407 Artificial Intelligence, AY 2026–27 Semester I
* **Institution**: BITS Pilani, K.K. Birla Goa Campus
* **Team**: Team 14
* **Authors**: 
  - Kushal (2023B1A70698G)
  - Vansh (2023B1A70715G)
  - Dorothy (2023B1A70714G)
  - Pranav (2024A7PS0515G)
* **Strict Evaluation Ceiling**: **3 Minutes (180 Seconds) Maximum**  
  > *Professor Rule: "If your video is longer, we will watch and score only the first 3 minutes."*
* **Target Recorded Runtime**: **2 Minutes 48 Seconds (168 Seconds)** *(Leaves a 12-second safety buffer)*
* **Format**: **100% In-Browser Walkthrough** on GitHub Pages (`https://learnerkg23.github.io/Audio-Engineering/`).  
  *No window-switching or PowerPoint needed — the entire presentation is hosted inside the web application.*
* **Open-Source Repository**: [https://github.com/LearnerKg23/Audio-Engineering](https://github.com/LearnerKg23/Audio-Engineering) *(93/93 Test Suites Passing)*

---

## ⏱️ Video Timeline & Scene Breakdown

| Scene | Timestamp | Duration | UI Screen Section | On-Screen Action | Primary Focus |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **Scene 1** | `0:00 – 0:18` | 18s | **Hero Section** (Waveform Visual) | Hover near title & waveform | Project Intro & Team Attribution |
| **Scene 2** | `0:18 – 0:34` | 16s | **Overview: Top Quadrants** | Click `Problem & Architecture` tab | Acoustic Problem & 5 Decoupled Stages |
| **Scene 3** | `0:34 – 0:50` | 16s | **Overview: Bottom Quadrants** | Scroll down; click orange CTA button | Data Strategy & Downstream ASR Proof |
| **Scene 4** | `0:50 – 1:55` | 65s | **Live Workbench (Stages 1–4)** | Play Raw/Stems; test RAG & Citations | Audition Separation & Guardrailed RAG |
| **Scene 5** | `1:55 – 2:35` | 40s | **Stage 5: Diagnostics & Ablations** | Click `Stage 5` in top navbar | Quantitative Gains & +4.20 dB Threshold |
| **Scene 6** | `2:35 – 2:48` | 13s | **Academic Footer & GitHub Link** | Scroll to bottom footer | Codebase, 93 Tests & Conclusion |
| **Buffer** | `2:48 – 3:00` | 12s | End of Video | — | Hard stop safety margin |

---

## 🎙️ Complete Word-for-Word Production Script

```
====================================================================================================
SCENE 1: HERO TITLE & CONTEXT
Duration: 0:00 – 0:18 (18 Seconds) | Word Count: 42 Words
Screen: Top of Website (Title, Embedded Waveform Graphic, 5 Stage Flow, Team Attribution)
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
* Open Chrome in Full Screen (`F11`) at [https://learnerkg23.github.io/Audio-Engineering/](https://learnerkg23.github.io/Audio-Engineering/).
* Camera view captures the clean title: *“Who Spoke What and When”*, the embedded green-to-coral acoustic soundwave, and the team attribution badges (*Kushal • Vansh • Dorothy • Pranav*).
* At **0:15**, move the mouse cursor toward the **`📑 Problem Statement & 5-Stage Architecture`** tab.
* Click the tab at **0:17**.

#### 🗣️ Spoken Dialogue:
> *"Hello everyone. We are Team 14 — Kushal, Vansh, Dorothy, and Pranav.*
>
> *Our project is: **'Who Spoke What and When: Deep Multistage Speech Disentanglement and Conversational Intelligence for Overlapping Code-Switched Audio'**.*
>
> *(Click tab)* *Let's examine the core engineering challenge and our architecture."*

---

```
====================================================================================================
SCENE 2: THE REAL-WORLD ACOUSTIC PROBLEM & ARCHITECTURE
Duration: 0:18 – 0:34 (16 Seconds) | Word Count: 38 Words
Screen: Top Half of Overview Dashboard (Quadrants 1 & 2)
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
* Hover cursor over Quadrant 1 (**The Real-World Acoustic Problem**), highlighting single-channel telephony and Hinglish code-switching.
* Sweep cursor across to Quadrant 2 (**End-to-End Pipeline Architecture**), indicating the 5 verified stages.

#### 🗣️ Spoken Dialogue:
> *"Commercial speech models like Whisper fail in India because real-world audio involves single-channel phone mics, heavy concurrent overlap, and rapid Hindi-English Hinglish code-switching.*
>
> *To solve this, our system **decouples acoustic disentanglement from linguistic decoding** across five verified stages: time-domain source separation, neural diarization, regional Indic ASR, and guardrailed LLM intelligence."*

---

```
====================================================================================================
SCENE 3: DATA STRATEGY & DOWNSTREAM ASR PROOF
Duration: 0:34 – 0:50 (16 Seconds) | Word Count: 36 Words
Screen: Bottom Half of Overview Dashboard (Quadrants 3 & 4) + Orange Action Button
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
* Scroll smoothly down to reveal Quadrant 3 (**Dual-Track Data Strategy**) and Quadrant 4 (**Downstream ASR Proof**).
* Briefly highlight the contrast between *The Unseparated Collision Catastrophe* (100% WER) and *The Separated Stem Recovery* (<1.0s flawless decoding).
* Hover cursor over the glowing orange button: **`🚀 Proceed to Live Working Audio & RAG Demonstration ↓`**.
* Click the button at **0:48 – 0:50** (the page smooth-scrolls straight into Stage 1).

#### 🗣️ Spoken Dialogue:
> *"We evaluate this on calibrated synthetic mixtures from IndicVoices, as well as in-the-wild All India Radio and Project Vaani phone calls.*
>
> *Without separation, overlapping speech causes catastrophic repetition loops and 100% Word Error Rate. Our front-end separation cleanly recovers the stems with zero bleed.*
>
> *(Click orange button)* *Now, let's see this in action on our live working demonstration."*

---

```
====================================================================================================
SCENE 4: LIVE DISENTANGLEMENT & CONVERSATIONAL RAG DEMO
Duration: 0:50 – 1:55 (65 Seconds) | Word Count: 140 Words | Live Audio Stems: 13 Seconds
Screen: Stage 1 DAW, Stage 2 Timeline, Stage 3 Transcripts, Stage 4 Interactive RAG Chatbot
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
1. **0:50 – 1:02**: In Stage 1 on Track A, click **"▶ Play Raw Mixture"**.  
   * **[AUDIO ACTION 1]**: Let raw audio play for **3 seconds** *(auditioning severe concurrent cross-talk)*. Click Pause.
2. **1:02 – 1:16**: Move down to the Multi-Track DAW.  
   * Click **"▶ Solo Stem 1"** $\rightarrow$ *[AUDIO ACTION 2]*: Play **3 seconds** of clean Female Hinglish voice.  
   * Click **"▶ Solo Stem 2"** $\rightarrow$ *[AUDIO ACTION 3]*: Play **3 seconds** of clean Male Hindi voice. Show zero cross-talk bleed.
3. **1:16 – 1:26**: Scroll down past **Stage 2 Diarization** and **Stage 3 Indic Transcripts**, pointing out synchronized speaker attribution pills.
4. **1:26 – 1:55**: In Stage 4, click tab **`💬 Ask LLM (Interactive RAG)`**.  
   * Click suggested query chip: **`"What was said about rehab?"`** $\rightarrow$ Grounded answer renders instantly.  
   * **CRITICAL DEMO ACTION**: Click the audio deep-linking citation pill: **`[SPEAKER_00 • 0.0s–5.7s ▶]`**.  
   * *[AUDIO ACTION 4]*: Audio player instantly seeks and plays the exact citation segment!  
   * Type out-of-domain query: *"What is the capital of France?"* and press Enter $\rightarrow$ Zoom in on **`[🛡️ Guardrail Triggered: Speculation prevented]`**.

#### 🗣️ Spoken Dialogue:
> *(0:50)* *"Here is our live demonstration on Track A.*  
> *(0:53)* *First, listen to this raw overlapping mixture:*  
> **[Audio: 3s entangled mixture]**  
> *(0:58)* *As you heard, both speakers talk concurrently.*  
> *(1:02)* *Now, we audition our isolated stems:*  
> **[Audio: 3s clean Female Stem 1]**  
> **[Audio: 3s clean Male Stem 2]**  
> *(1:16)* *Cross-talk is completely suppressed.*  
> *(1:19)* *Next, Stage 2 and 3 produce precise millisecond timelines and bilingual Devanagari transcripts.*  
> *(1:26)* *Finally, in Stage 4, our Conversational RAG engine allows users to interrogate the meeting.*  
> *(1:34)* *Clicking 'What was said about rehab?' retrieves the grounded answer. Clicking the citation pill instantly seeks and plays that exact audio stem.*  
> *(1:47)* *And if we ask an out-of-domain question, our strict negative guardrails prevent speculation with zero percent hallucination."*

---

```
====================================================================================================
SCENE 5: QUANTITATIVE RESULTS & THE BREAKDOWN THRESHOLD
Duration: 1:55 – 2:35 (40 Seconds) | Word Count: 92 Words
Screen: Stage 5 Diagnostics & Scientific Ablation Table
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
* In the sticky top navigation bar, click **`Stage 5: Diagnostics`**.
* The page instantly smooth-scrolls to the **Progressive Pipeline Architecture Ablation Benchmark Table**.
* Hover cursor over the key quantitative columns:
  - $\mathbf{+21.80\text{ dB}}$ SI-SDR Gain
  - DER reduction from $40.52\%$ to $\mathbf{23.97\%}$
  - WER reduction from $86.28\%$ to $\mathbf{51.57\%}$
  - Highlight the critical **$+4.20\text{ dB}$ SI-SDR Breakdown Threshold**.

#### 🗣️ Spoken Dialogue:
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
SCENE 6: SUMMARY, REPOSITORY & CONCLUSION
Duration: 2:35 – 2:48 (13 Seconds) | Word Count: 32 Words
Screen: Academic Footer & GitHub Repository Link
====================================================================================================
```

#### 🖥️ Visual Track & Mouse Actions:
* Scroll smoothly to the clean footer at the bottom of the page.
* Highlight the GitHub link: `github.com/LearnerKg23/Audio-Engineering`.

#### 🗣️ Spoken Dialogue:
> *"Our complete open-source codebase, passing all 93 test suites, and our live interactive demonstration are publicly available on GitHub.*
>
> *Thank you for your time, and we look forward to your questions."*

```
====================================================================================================
[2:48 – 3:00] VIDEO COMPLETE (12-Second Buffer before 3:00 evaluation hard cutoff)
====================================================================================================
```

---

## 📋 One-Page Teleprompter Cheat Sheet (For Recording)

```text
[0:00 - 0:18] (Hero Banner)
"Hello everyone. We are Team 14 — Kushal, Vansh, Dorothy, and Pranav. 
Our project is: 'Who Spoke What and When: Deep Multistage Speech Disentanglement 
and Conversational Intelligence for Overlapping Code-Switched Audio'.
Let's examine the core engineering challenge and our architecture."
>>> ACTION: Click tab [Problem Statement & 5-Stage Architecture]

[0:18 - 0:34] (Top Quadrants)
"Commercial speech models like Whisper fail in India because real-world audio 
involves single-channel phone mics, heavy concurrent overlap, and rapid 
Hindi-English Hinglish code-switching.
To solve this, our system decouples acoustic disentanglement from linguistic 
decoding across five verified stages: time-domain source separation, neural 
diarization, regional Indic ASR, and guardrailed LLM intelligence."

[0:34 - 0:50] (Bottom Quadrants)
"We evaluate this on calibrated synthetic mixtures from IndicVoices, as well 
as in-the-wild All India Radio and Project Vaani phone calls.
Without separation, overlapping speech causes catastrophic repetition loops 
and 100% Word Error Rate. Our front-end separation cleanly recovers the stems 
with zero bleed.
Now, let's see this in action on our live working demonstration."
>>> ACTION: Click orange button [Proceed to Live Working Audio & RAG Demonstration]

[0:50 - 1:55] (Live Demo)
"Here is our live demonstration on Track A.
First, listen to this raw overlapping mixture:
[Play 3s raw mix]
As you heard, both speakers talk concurrently.
Now, we audition our isolated stems:
[Play 3s Stem 1]
[Play 3s Stem 2]
Cross-talk is completely suppressed.
Next, Stage 2 and 3 produce precise millisecond timelines and bilingual Devanagari transcripts.
Finally, in Stage 4, our Conversational RAG engine allows users to interrogate the meeting.
Clicking 'What was said about rehab?' retrieves the grounded answer. Clicking the citation 
pill instantly seeks and plays that exact audio stem.
And if we ask an out-of-domain question, our strict negative guardrails prevent speculation 
with zero percent hallucination."

[1:55 - 2:35] (Stage 5 Diagnostics & Results)
>>> ACTION: Click top navbar [Stage 5: Diagnostics]
"Quantitatively, across controlled mixtures and All India Radio field data, our pipeline 
achieves an average +21.80 dB SI-SDR separation gain.
Decoupling separation reduces Diarization Error Rate from 40.5% down to 23.97%, and cuts 
downstream Word Error Rate from 86.3% to 51.57%.
Crucially, our ablation study uncovered an empirical acoustic breakdown threshold at 
+4.20 dB SI-SDR. Below this boundary, downstream phoneme decoding collapses catastrophically.
Finally, our end-to-end pipeline executes at 0.49 to 0.67 times real-time, enabling 
high-throughput speech intelligence."

[2:35 - 2:48] (Conclusion)
"Our complete open-source codebase, passing all 93 test suites, and our live interactive 
demonstration are publicly available on GitHub.
Thank you for your time, and we look forward to your questions."
[STOP RECORDING at 2:48]
```

---

## 🛠️ Recording Setup Checklist

1. **Browser**: Open [https://learnerkg23.github.io/Audio-Engineering/](https://learnerkg23.github.io/Audio-Engineering/) in Chrome. Press `F11` for clean full-screen view.
2. **OBS Audio Capture**:
   - Add **Audio Output Capture** (Desktop Audio) in OBS so that browser stem playbacks are loud and clear.
   - Add **Audio Input Capture** (Microphone) with a gentle noise suppression filter.
3. **Pacing**: Speak at a steady, confident conversational pace (~135 words per minute). Allow the live audio stems to play for their full 3 seconds without talking over them.
