# Voice AI in India

India has 1.45 billion people, most of whom prefer voice in regional languages. The voice AI stack splits into cascade (ASR → LLM → TTS) and native speech-to-speech models. 95% of Indian voice AI startups run cascade because the Sarvam stack (Saaras V3 ASR + Sarvam-M + Bulbul TTS) outperforms global models on Indian languages at lower cost. This page covers the architectures, the stack components, the Indian startup landscape, and the metrics that matter.

!!! tip "Rapid Recall"
    **Cascade architecture** (ASR → LLM → TTS): each stage swappable, mature tooling, 500ms-2s latency, easy to debug, full control. **95% of Indian voice AI in 2026.** **Native speech-to-speech** (OpenAI gpt-realtime, Gemini Live, Kyutai Moshi, NVIDIA PersonaPlex): single model processes raw audio → raw audio, 160-800ms latency, full-duplex barge-in, preserves paralinguistic cues, but bad at structured tasks (Moshi 1.26/5 on procedural) and no Indian-language support yet. **Hybrid is 2026 production consensus**: native for conversational shell, cascade for transactional shell. **Sarvam's structural moat**: training data, speaker diversity, code-mixing ("Haan ji, mera account number 12345 hai, I want to check balance"), telephony audio (8kHz, codec, noise), low-resource Indian languages. **Metrics:** WER, MOS, AHT, containment rate.

## §1 The two architectures

### 1.1 Cascade (ASR → LLM → TTS) — legacy but still dominant

```
[Audio] → ASR → [Text] → LLM → [Text] → TTS → [Audio]
```

**Pros:**

- Each stage independently swappable, upgradeable, debuggable.
- Mature tooling (Deepgram, AssemblyAI, ElevenLabs, Whisper).
- Easy to add business logic between stages.
- Full control over each step.

**Cons:**

- **High latency:** 500 ms to 2 s total (typically ASR 200 ms + LLM 300-1000 ms + TTS 200-500 ms).
- Loses paralinguistic info (tone, urgency, emotion) at ASR stage.
- No natural interruption handling.
- Awkward turn-taking — user must fully stop before system responds.

**This is what 95% of Indian voice AI startups run in 2026**, because:

- Sarvam Saaras V3 + Sarvam-M + Sarvam Bulbul = full Indian-language cascade at lower cost than OpenAI.
- Cost control: you are not paying for full-duplex audio streaming.
- Compliance: easier to log transcripts for audit.

### 1.2 Speech-to-speech (native / full-duplex) — the frontier

```
[Audio stream] → Single model → [Audio stream]
```

No text in the middle. The model processes raw audio tokens and generates raw audio tokens. Examples:

- **OpenAI gpt-realtime / gpt-realtime-mini.**
- **Google Gemini Live.**
- **Kyutai Moshi** (open-source, French non-profit, 160 ms latency).
- **NVIDIA PersonaPlex** (built on Moshi, adds persona/role controls, 205 ms).

**Pros:**

- **Ultra-low latency:** 160-800 ms end-to-end (5-10× better than cascade).
- Full-duplex: model listens while speaking, can be interrupted naturally.
- Preserves paralinguistic cues (emotion, urgency, backchanneling).
- "Hmm," pauses, natural conversation rhythm.

**Cons:**

- **Bad at structured instruction-following:** Moshi scores 1.26/5 on procedural tasks.
- Harder to debug (no intermediate text).
- Much more expensive per second of audio.
- Voice is typically fixed or limited (PersonaPlex lets you control it, Moshi does not).
- Tool calling / function calling less reliable.
- **No support for Indian languages yet in native models.**

### 1.3 Hybrid patterns (2026 production consensus)

Most real products in 2026 use a hybrid:

- **Native for the "conversational shell":** greeting, smalltalk, handling interruptions.
- **Cascade with structured prompts for the "transaction shell":** identity verification, appointment scheduling, payment.
- **Tool calls on ASR text, response synthesized via native TTS.**

OpenAI's gpt-realtime specifically optimized for this: complex instruction following, tool calls, **word-for-word script reading** for regulated domains.

## §2 The voice AI stack you will see in Indian interviews

### 2.1 ASR (Speech-to-Text)

- **Global:** Whisper v3, Deepgram Nova-3, OpenAI gpt-4o-transcribe.
- **India:** Sarvam Saaras V3 (best on IndicVoices, Svarah for Indian-accented English).
- **Key metric:** Word Error Rate (WER), on **your specific accents and domain**.
- 2026 realities: 8 kHz telephony support is make-or-break for call centers; streaming is now standard (token starts flowing while audio plays).

### 2.2 LLM / Dialog Manager

- Same as any LLM task — route complexity, enforce structured output for transactions.
- **India-specific:** Sarvam-M for regional languages, code-mixing support.

### 2.3 TTS (Text-to-Speech)

- **Global:** ElevenLabs, OpenAI gpt-4o-mini-tts, Cartesia, Rime.
- **India:** Sarvam Bulbul (Indian languages, natural prosody).
- **Key metrics:** naturalness (MOS score), latency to first audio, voice steerability.
- Custom voices / voice cloning: available on most providers with ethics guardrails.

### 2.4 VAD (Voice Activity Detection)

- Detects when user is speaking vs silent.
- Silero VAD (open-source) is standard.
- Tuning matters: aggressive = cutting user off; conservative = slow response.

### 2.5 Barge-in / Interruption Handling

- User starts talking → system must stop speaking and listen.
- Native models handle this automatically.
- Cascade needs explicit engineering.

### 2.6 Turn-Taking

- When does the user "finish"? Silence threshold + semantic detection ("did they finish a thought?").
- Hard problem. The thing that makes Moshi / PersonaPlex feel magical.

## §3 Indian Voice AI startup landscape

- **Sarvam AI** — the sovereign stack (22 Indian languages, edge models, telephony-optimized).
- **Eltropy** — voice AI agents for credit unions, cascade architecture.
- **Gnani.ai** — enterprise voice assistants, Indian languages.
- **CoRover.ai** — Hindi bhashini for government, conversational AI.
- **Haptik** — Jio-owned, customer support, multilingual.
- **Vernacular.ai** (acquired by SaaSBoomi) — voice agents.
- **Skit.ai** — contact center AI.

**What they all struggle with (and Sarvam is solving):**

- **Code-mixing.** "Haan ji, mera account number 12345 hai, I want to check balance."
- **Telephony audio.** 8 kHz, compressed codec, background noise.
- **Speaker diarization** in crowded environments.
- **Low-resource languages.** Oriya, Kashmiri, Maithili.
- **Real-time streaming** at Indian internet quality.

## §4 The interview framing for voice AI roles

1. **Know the cascade vs native tradeoff cold.**
2. **Speak Sarvam's stack by name** (Saaras, Bulbul, Sarvam-M).
3. **Quantify latency budgets.** Telephony users perceive 800 ms as natural; 1.5 s feels robotic; 2 s+ feels broken.
4. **Know the evaluation stack:** WER, MOS, task completion rate, average handle time (AHT), call containment rate.
5. **Have an opinion on ROI:** voice AI is typically sold on AHT reduction (5-15 min → 3 min) and containment rate (percent resolved without human transfer, 40-70% target).

## Interview Questions

**Q1: What is the difference between a cascade and a native voice AI system, and when would you pick each?**

Cascade is ASR → LLM → TTS; each stage independent, swappable, mature tooling. Native (speech-to-speech) uses a single model that ingests and produces audio tokens directly — no text intermediate. Cascades dominate production because they have lower cost, better tool use and instruction following, and easier debugging; latency is 500 ms to 2 s. Native models (gpt-realtime, Moshi, PersonaPlex) achieve 160-800 ms and handle interruptions naturally but cost more and are weaker at structured tasks. For Indian languages there are no frontier native models yet, so you are forced into cascade with Sarvam's stack: Saaras V3 ASR, Sarvam-M, Bulbul TTS. The 2026 production pattern is hybrid: native for conversational shell, cascade for transactional tasks.

**Q2: Why is the Sarvam stack the right answer for an Indian-language voice product?**

Three reasons. (1) Training data: Saaras V3 was trained on 1M+ hours of Indian-language audio including 22 scheduled languages and code-mixing (which global models never see). (2) Telephony optimization: 8 kHz, compressed codecs, noisy backgrounds — what call centers actually deal with. (3) Benchmark dominance: Saaras V3 beats Gemini 3 Pro and GPT-4o on IndicVoices and Svarah, the standard Indian-language ASR benchmarks. Plus the full stack: Sarvam-M for the LLM with code-mixing support, Bulbul for natural Indian-language TTS. End-to-end Indian-language pipeline at lower cost than OpenAI.

**Q3: What latency budget should you target for a voice AI call center agent?**

Telephony users perceive 800 ms as natural conversation, 1.5 s as robotic, 2 s+ as broken. The ASR + LLM + TTS budget needs to fit under 800 ms for natural feel: ASR 200 ms streaming, LLM 300-400 ms (with prefix caching on system prompts), TTS 200-300 ms streaming first audio. For complex transactions (looking up an account) you can buffer with a filler phrase ("let me check that for you") to keep the perceived latency low while the actual task completes.

**Q4: How would you evaluate a voice AI deployment?**

Four metrics. (1) WER on production audio (your accents, your domain), not just public benchmarks — common pitfall is to optimize Whisper on academic test sets while it tanks on your customers. (2) MOS for TTS naturalness, scored by humans on a 1-5 scale. (3) AHT reduction (average handle time): voice AI should cut human agent time from 5-15 min to 3 min via pre-handling routine queries. (4) Containment rate: percent of calls resolved without human transfer, 40-70% target depending on domain. ROI calculation is usually AHT reduction × agent headcount + containment savings.

**Q5: What is code-mixing and why is it the hardest part of Indian voice AI?**

Code-mixing is mixing two languages in one utterance: "Haan ji, mera account number 12345 hai, I want to check balance." Standard ASR trained on monolingual data fails because it expects either Hindi or English, not both. Production-grade Indian voice systems must handle code-mixing natively, which requires training data with that pattern (which only Indian-trained models have). It is also why Sarvam's stack outperforms Whisper for Indian call centers despite Whisper's general quality — Whisper trips on the Hindi-English transitions in a way a Sarvam model trained on Indian telephony does not.

**Q6 (Trap): A vendor pitches a native speech-to-speech model for an Indian-language customer service bot. Should you buy?**

Not in 2026, no. Native speech-to-speech models (gpt-realtime, Moshi, PersonaPlex) have no Indian-language support yet — they were trained primarily on English. Cascade with Sarvam's stack gives you Indian-language coverage, code-mixing handling, telephony-optimized ASR, and lower cost. The interruption-handling benefit of native models is real but achievable with engineered cascade + VAD. Re-evaluate when Indian-language native models exist (likely 2027+).
