# First Evaluation Flash Talk Script (3–4 Minutes)

**Presenter**: Technical Team / Lead Engineer  
**Presentation Time**: 3 Minutes 30 Seconds (+ 30s Q&A Buffer)  
**Slides**: 2 Slides ([`docs/flash_talk_slides.md`](docs/flash_talk_slides.md))  
**Visual / Audio Demo**: [`outputs/first_eval_demo/spectrogram_comparison.png`](outputs/first_eval_demo/spectrogram_comparison.png)  

---

## [0:00 – 0:45] Minute 1: The Problem & System Objective (Slide 1)

> *"Good morning, Professor and peers.*
>
> *In machine learning engineering, acoustic speech models are almost always tested in clean, studio conditions with single-speaker turn-taking. But in the real world—especially across the Indian subcontinent in community radio broadcasts, rural panel discussions, and agricultural phone-in programs—speech is chaotic.*
>
> *Multiple speakers talk over each other with heavy overlap; speakers dynamically code-switch between regional languages and English—specifically **Hinglish**; and recordings suffer from uncalibrated mobile microphones and non-stationary ambient noise.*
>
> *Standard commercial ASR engines fail completely under these conditions.*
>
> *Our project, **Advanced Computational Speech Engineering**, tackles this by building an end-to-end multi-stage pipeline designed to answer: **'Who Spoke What and When?'**"*

---

## [0:45 – 1:30] Minute 2: System Architecture & First Evaluation Boundary (Slide 1)

> *"To solve this, our proposed system integrates four specialized stages:*
>
> *1. **Stage 1: Blind Source Separation** to isolate overlapping acoustic waveforms into single-speaker channels.*  
> *2. **Stage 2: Acoustic Speaker Diarization** using `pyannote.audio` neural embeddings and Spectral Clustering to track speaker turn timelines.*  
> *3. **Stage 3: Regional & Code-Switched ASR** using AI4Bharat's `IndicWav2Vec` to transcribe Hinglish phonemes directly into native script.*  
> *4. **Stage 4: Semantic Post-Processing & Guardrailing** using localized foundational LLMs to clean disfluencies and extract structured executive reports without hallucinating.*
>
> *For this **First Evaluation Checkpoint**, we have established our complete data strategy, selected our models, defined our mathematical evaluation framework, and **fully implemented and verified Stage 1 Source Separation**."*

---

## [1:30 – 2:30] Minute 3: Data Strategy & Live Working Demonstration (Slide 2)

> *(Advance to Slide 2)*
>
> *"First, our data strategy: In real-world radio recordings, clean individual speaker tracks do not exist in isolation. Therefore, to evaluate source separation mathematically, we extracted authentic, human conversational speech slices from open Indian corpora—specifically `IndicVoices` and the `Hindi-English Bilingual` dataset—and synthesized a **Controlled Evaluation Suite** with calibrated overlap ratios (25% and 50%) and relative volume levels (0 dB and +6 dB SIR).*
>
> *To evaluate separation, we implemented **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)** with **Permutation Invariant Training (PIT)** matching, ensuring that arbitrary output channel ordering is not falsely penalized.*
>
> *Now, let me demonstrate our working Stage 1 prototype:*
>
> *(Play or point to Audio Demo: `controlled_mix_01`)*
>
> *Here is the **original mixture**: a female Hinglish speaker talking about fitness tests while a male Hindi speaker overlaps concurrently. In the raw mixture, the voices are acoustically entangled.*
>
> *Here is our **separated output channel 0**: the female Hinglish voice, fully isolated.*  
> *And here is our **separated output channel 1**: the male Hindi voice, with the cross-talk completely suppressed.*
>
> *As you can see in our spectrogram analysis on the slide, the overlapping harmonic formants have been cleanly disentangled."*

---

## [2:30 – 3:30] Minute 4: Quantitative Results & Phase B Roadmap (Slide 2)

> *"Quantitatively, across our 6 controlled evaluation conditions on Apple Silicon Metal GPU, our primary model—**Conv-TasNet**—achieved an average SI-SDR of **+15.45 dB**, representing a **+15.45 dB improvement** over the unseparated baseline.*
>
> *Crucially, we also evaluated the proposal's alternative candidate, **Meta's Demucs (`htdemucs`)**, and discovered an important scientific insight: Demucs maps all human vocal energy into a single 'vocals' stem to separate voice from music, rather than separating two distinct human talkers. Therefore, Conv-TasNet is our primary multi-speaker separator, while Demucs serves as our front-end environmental denoiser.*
>
> *Furthermore, Conv-TasNet achieved an average Real-Time Factor (RTF) of **0.075**—processing a 9.5-second audio clip in just 0.7 seconds, over 13 times faster than real-time playback.*
>
> *With our data pipeline verified, models selected, SI-SDR evaluation operational, and Stage 1 working demo complete, we have achieved 100% of our First Evaluation requirements.*
>
> *Following your feedback today, we immediately proceed to **Phase B**: feeding these isolated stems into `pyannote` for timeline diarization, followed by regional ASR transcription and LLM structuring.*
>
> *Thank you. We welcome your questions."*

---

## Anticipated Questions & Answers for Q&A Buffer (30 Seconds)

### Q: "Why use synthetic mixtures instead of just evaluating on real radio broadcasts?"
> **Answer**: *"Real radio broadcasts don't have separate isolated microphone tracks for each guest. Without clean reference tracks, you cannot compute mathematical metrics like SI-SDR or SDR. By constructing controlled mixtures from real human Indian speech, we know the exact ground truth, allowing rigorous mathematical proof of separation. In Phase B Milestone M11, we evaluate real-world broadcasts qualitatively."*

### Q: "Why was Permutation Invariant Training (PIT) necessary?"
> **Answer**: *"Neural separation models don't know who Speaker A or Speaker B is—they output Channel 1 and Channel 2. If the model outputs Speaker B on Channel 1 and Speaker A on Channel 2, a fixed comparison would register near-infinite error despite perfect physical separation. PIT evaluates all permutations and selects the optimal match."*
