# Stage 1 & Stage 2 Live Demonstration Speaking Script
## "Who Spoke What and When" — Multi-Speaker Indian Audio Engineering

---

## 1. Quick Presentation Checklist Before You Begin

Before sharing your screen or presenting in front of professors and evaluators:
- [ ] Open **`demo.html`** in Google Chrome or Microsoft Edge (or open your live GitHub Pages link).
- [ ] Ensure computer audio output is set correctly and volume is audible (~60–70%).
- [ ] Verify you are in full-screen mode (**F11** on Windows or browser zoom set to 90–100% so the entire interface fits comfortably).
- [ ] Confirm you start at the top of the page on **Track A** (Controlled Synthetic Mixture).

---

## 2. Walkthrough Timetable (5-Minute Master Presentation)

| Time | Section | Screen Action | Core Message |
|---|---|---|---|
| **00:00 – 00:45** | **1. The Problem Hook & Project Scope** | Point to Hero Waveform & Highlighted Stage 1 & 2 Cards | The Cocktail Party Problem in Indian speech and our focus on Stage 1 (Separation) & Stage 2 (Diarization). |
| **00:45 – 01:15** | **2. Quantitative Telemetry Summary** | Hover over the 3 Neumorphic Telemetry Cards | +21.77 dB SI-SDR gain, 9.20% DER, and ultra-fast 0.006 RTF. |
| **01:15 – 02:15** | **3. Live Audio Audition: Track A** | Play Mix $\to$ Play Stem 01 $\to$ Play Stem 02 | Audible isolation of overlapping voices with zero bleed. |
| **02:15 – 03:15** | **4. Stress-Testing: Track B (AIR Panel)** | Switch to Track B tab $\to$ Audition real-world stems | Handling 3 speakers, conversational barge-ins, and studio reverberation. |
| **03:15 – 04:15** | **5. Stage 2 Diarization DAW & Timeline** | Scrub Playhead & Click Speaker Turns in Table | Millisecond boundary accuracy, hyperspherical clustering, and separation-aware overlap resolution. |
| **04:15 – 04:45** | **6. Empirical Benchmark Validation** | Toggle Terminal Console Tabs (Stage 1 & Stage 2) | Proving Conv-TasNet superiority over Demucs and rigorous quantitative metrics. |
| **04:45 – 05:00** | **7. Downstream Impact & Conclusion** | Briefly expand ASR Accordion $\to$ Conclude | Upstream separation prevents downstream ASR attention collapse. |

---

## 3. Step-by-Step Spoken Script (Word-for-Word Guide)

### Part 1: Project Overview & The Problem Hook (00:00 – 00:45)
**Screen Location**: Top Hero Section  
**What to Click**: None (keep the page at the top).

> **What to Say:**
> *"Good morning / afternoon professors and evaluators.*
> 
> *Our project is titled **'Who Spoke What and When'** — an end-to-end computational speech engineering pipeline built specifically for challenging multi-speaker Indian conversational audio.*
> 
> *In natural Indian conversations — such as broadcast panel debates, podcasts, or phone helplines — people frequently talk simultaneously. When two or more people speak at once into a single microphone, traditional speech systems collapse. Speech recognition hallucinates or drops words, and speaker diarization assigns wrong timestamps.*
> 
> *To solve this, we implemented a modular deep learning pipeline. As you can see highlighted here in our architecture sequence, our primary focus and core technical contribution for this demonstration is on the first two stages: **Stage 1: Speech Separation** using Time-Domain Conv-TasNet, and **Stage 2: Hyperspherical Speaker Diarization**."*

---

### Part 2: Top Telemetry & Benchmark Metrics (00:45 – 01:15)
**Screen Location**: Top 3 Summary Cards  
**What to Click**: Point your cursor at the three large metric cards.

> **What to Say:**
> *"Before we audition the audio, let's look at the quantitative results achieved by our pipeline:*
> 
> 1. *First, **+21.77 dB SI-SDR Improvement**: In Blind Source Separation, Scale-Invariant Signal-to-Distortion Ratio is the gold-standard metric. An improvement above +20 dB represents over 100-fold suppression of competing acoustic interference.*
> 2. *Second, a **9.20% Diarization Error Rate (DER)**: By integrating upstream source separation ahead of speaker clustering, we resolved overlapping speech collisions that normally cripple standard diarizers.*
> 3. *Third, a **Real-Time Factor of 0.006**: The separation forward pass takes only 6 milliseconds to process 1 second of audio. That is over 160 times faster than real-time, making it viable for live streaming deployment."*

---

### Part 3: Live Audio Demonstration — Track A (01:15 – 02:15)
**Screen Location**: Multi-Stem Audition Console (Track A)  
**What to Click**:
1. Click the **▶ Play button** on the **Mixed Audio (Input Mixture)** card.
2. Let it play for 4–5 seconds so the evaluators clearly hear the overlap.
3. Click the **▶ Play button** on **Speaker 01 (Isolated Stem 01)**.
4. Click the **▶ Play button** on **Speaker 02 (Isolated Stem 02)**.
5. Point to the **3-Panel Spectrogram** below the cards.

> **What to Say:**
> *(Click Play on Input Mixture)*  
> *"Let us listen to the raw input mixture. This is Track A: two speakers — Female Hinglish and Male Hindi — with a 25% conversational overlap window at 0 dB Signal-to-Interference Ratio.*
> 
> *(Let audio play for 3 seconds)*  
> *Notice right in the middle, both speakers speak concurrently. The acoustics collide into an unintelligible mixture.*
> 
> *(Click Play on Speaker 01)*  
> *Now, listen to the output of our Stage 1 Conv-TasNet model for Speaker 01. The interrupting voice has been completely decoupled. The vocal tract formants are preserved with high acoustic fidelity.*
> 
> *(Click Play on Speaker 02)*  
> *And here is the isolated output for Speaker 02. Notice that during the overlap region, Speaker 01's voice is virtually silent. In the spectrogram display below, you can visually observe how the overlapping energy in the mixture has been cleanly partitioned into two discrete time-frequency representations."*

---

### Part 4: Stress-Testing Real-World Audio — Track B (02:15 – 03:15)
**Screen Location**: Environment Tabs $\to$ Track B  
**What to Click**:
1. Click the **"📻 Track B: Real-World AIR Panel (3 Speakers)"** tab.
2. Click **▶ Play** on the **Input Mixture (3-Speaker Debate)**.
3. Click **▶ Play** on **Channel 01 (Host & Guest 2)**.
4. Click **▶ Play** on **Channel 02 (Interrupting Guest 1)**.

> **What to Say:**
> *(Click Track B Tab)*  
> *"Now let's move beyond clean synthetic data to **Track B**, our in-the-wild evaluation suite sourced from All India Radio (AIR) broadcast archives. This clip represents a real 3-speaker panel discussion with room reverberation and multiple aggressive interruptions.*
> 
> *(Click Play on Input Mixture)*  
> *Here, the host begins, Guest 1 interrupts at 3.8 seconds with high vocal energy, and Guest 2 cuts in at 6.8 seconds. This creates two distinct collision intervals.*
> 
> *(Click Play on Channel 02 - Guest 1)*  
> *Our Conv-TasNet architecture operates on continuous speech separation principles. When Guest 1 interjects ('तारीख के दिन मेरे स्वामी जाएंगे...'), the model isolates her higher-pitch vocal energy into Channel 02, suppressing both male speakers.*
> 
> *(Click Play on Channel 01 - Host & Guest 2)*  
> *Meanwhile, Channel 01 cleanly carries the host and Guest 2. This proves that even in uncurated real-world broadcasts with room reverberation, our model successfully untangles overlapping conversational bursts."*

---

### Part 5: Stage 2 Diarization DAW & Timeline (03:15 – 04:15)
**Screen Location**: Stage 2 Diarization Section  
**What to Click**:
1. Scroll down to the **Multi-Track Diarization Timeline**.
2. Click anywhere on the ruler or lane to scrub the playhead.
3. In the **Chronological Speaker Turns Table**, click the **▶ Turn 1**, **▶ Turn 2**, or **▶ Turn 3** buttons to audition individual speech segments.
4. Point to the red **Overlap Collision Indicators (OV 1 & OV 2)**.

> **What to Say:**
> *"Now we move into **Stage 2: Speaker Diarization** — answering precisely 'Who Spoke When'.*
> 
> *Standard diarization algorithms cluster raw audio in the time domain, which means when two people talk at once, the system is forced to pick only one speaker or assign false boundaries. We solved this with **Separation-Aware Diarization**.*
> 
> *(Point to DAW tracks)*  
> *As shown in our interactive DAW timeline, each speaker is allocated a dedicated track lane with millisecond-exact boundaries. Notice the red indicator blocks marked 'OV 1' and 'OV 2'. In naive diarization, these overlap windows cause massive Diarization Error Rate spikes.*
> 
> *(Click Play on Turn 2 in the table)*  
> *Because we extract hyperspherical unit-norm embeddings from the separated stems, our system correctly tracks Speaker 01's interjection in Turn 2, preserving speaker consistency across the entire conversation. Clicking any turn in our verification table instantly auditions that speaker's segment with its verified transcript."*

---

### Part 6: Empirical Benchmark Console (04:15 – 04:45)
**Screen Location**: Benchmark Console Section  
**What to Click**:
1. Scroll down to the **Benchmark Console**.
2. Click **Tab 1: `evaluate_stage1_separation.py`**.
3. Click **Tab 2: `evaluate_stage2_diarization.py`**.

> **What to Say:**
> *"To ensure academic rigor, all our results are verified through automated benchmark test suites in our console:*
> 
> *(Click Tab 1: Stage 1)*  
> *In Stage 1, we compared **Conv-TasNet against Meta's Demucs**. While Demucs is popular for music separation, its spectrogram-masking architecture collapsed on multi-speaker human speech, achieving only +0.94 dB. Conv-TasNet operates in the raw time domain with dilated temporal convolutions, achieving an average gain of **+15.45 dB across all mixtures and +21.77 dB on our primary benchmark**.*
> 
> *(Click Tab 2: Stage 2)*  
> *In Stage 2, our Hyperspherical spectral clustering evaluated across 4 key error components: Missed Speech, False Alarm, and Speaker Confusion, achieving an overall DER of 9.20%."*

---

### Part 7: Downstream Impact & Conclusion (04:45 – 05:00)
**Screen Location**: Downstream Recognition (ASR) Accordion  
**What to Click**: Click the dropdown to briefly expand the ASR section, then close or summarize.

> **What to Say:**
> *(Click to open ASR Accordion)*  
> *"Finally, why is upstream separation so critical? Without separation, passing overlapping audio into an ASR model like AI4Bharat IndicWav2Vec causes complete attention collapse, resulting in repetitions and severe word loss.*
> 
> *By inserting Stage 1 and Stage 2 as an acoustic front-end, we physically decouple overlapping wavefronts, giving clean, single-speaker streams to downstream speech recognition and language models.*
> 
> *Thank you, and we are now ready to take your questions."*

---

## 4. Cheat Sheet: Questions Evaluators Love to Ask & How to Answer

### Q1: "Why did you choose Conv-TasNet over Meta's Demucs?"
> **Your Answer:**
> *"Demucs was originally trained for music stem separation (drums, bass, vocals, other). When presented with two human voices, Demucs considers all human vocal energy as a single stem ('vocals'), so it fails to separate speaker from speaker (+0.94 dB).  
> In contrast, Conv-TasNet is specifically designed for multi-speaker speech separation. It operates directly in the time domain using a 1D convolutional encoder and a dilated Temporal Convolutional Network (TCN). This avoids Short-Time Fourier Transform (STFT) phase estimation errors and isolates individual vocal pitch tracks, giving us +21.77 dB gain."*

### Q2: "In Track B, you have 3 speakers, but Conv-TasNet outputs 2 channels. How does it handle that?"
> **Your Answer:**
> *"In real-world conversations, it is acoustically very rare for 3 people to utter phonemes simultaneously at the exact same millisecond. Interruptions happen in pairwise bursts (e.g., Host vs Guest 1, then Guest 1 vs Guest 2).  
> Conv-TasNet separates the **locally active pair** in each temporal window. In our Stage 2 Diarization, we compute hyperspherical embeddings across the separated streams to stitch and identify that three distinct speaker identities were present over the full recording."*

### Q3: "What is the difference between SDR and SI-SDR, and why did you use SI-SDR?"
> **Your Answer:**
> *"Standard Signal-to-Distortion Ratio (SDR) is sensitive to arbitrary gain changes or volume scaling. If an algorithm simply amplifies the signal by 3 dB, SDR values can be artificially inflated.  
> Scale-Invariant SDR (SI-SDR) mathematically orthogonalizes the estimated signal against the target reference to remove any gain factor $\alpha$ before measuring noise. It is the accepted scientific standard in source separation literature (Le Roux et al., ICASSP 2019)."*

### Q4: "What is Diarization Error Rate (DER) and what does 9.20% mean?"
> **Your Answer:**
> *"DER is the standard NIST metric for speaker diarization. It consists of three additive errors:  
> $\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm Speech} + \text{Speaker Confusion}}{\text{Total Reference Speech Time}}$  
> In overlapping multi-speaker speech, unseparated baselines usually score above 35–40% DER because overlapping regions cause speaker confusion. Our 9.20% DER demonstrates that our separation-aware pipeline successfully resolved the overlap boundaries."*

### Q5: "What makes Indian conversational audio unique or challenging?"
> **Your Answer:**
> *"Indian conversational speech features frequent **code-switching** (Hinglish), wide acoustic variability across regional accents, higher conversational overlap cadence in broadcast debates, and non-ideal recording channels (e.g., 2G mobile phone-in calls with 300–3400 Hz bandpass limits in Project Vaani). Our pipeline was built and stress-tested specifically against these native acoustic conditions."*

---

## 5. Quick Numbers Flashcard (Keep This Next to Your Keyboard)

```
┌────────────────────────────────────────────────────────────────────────┐
│                     KEY BENCHMARK NUMBERS AT A GLANCE                  │
├────────────────────────────────────────────────────────────────────────┤
│ • Separation Algorithm    : Conv-TasNet (Time-domain Dilated TCN)      │
│ • Separation Metric       : +21.77 dB Δ SI-SDR (Controlled Benchmark)  │
│ • Baseline Comparison     : Conv-TasNet (+21.8 dB) vs Demucs (+0.9 dB) │
│ • Diarization Algorithm   : Hyperspherical Cosine + Spectral Clustering│
│ • Diarization Accuracy    : 9.20% DER (Diarization Error Rate)         │
│ • Inference Latency       : 0.006 RTF (>160x faster than real-time)    │
│ • Primary Dataset Corpora : All India Radio (AIR-RS-DB), Project Vaani,│
│                             IndicVoices, and Nirantar (AI4Bharat)      │
│ • Track A Test Scenario   : 2 Speakers, 25% Overlap, 0 dB SIR          │
│ • Track B Test Scenario   : 3 Speakers, Dual Overlap, Studio Reverb    │
└────────────────────────────────────────────────────────────────────────┘
```
