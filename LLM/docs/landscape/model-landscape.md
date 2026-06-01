# Model Landscape — April 2026

The four axes that define every model: open vs closed, frontier vs mid-tier vs small, native multimodal vs text-first, reasoning vs instant. The 2026 plot twist: the gap between open and closed has nearly closed. DeepSeek V4, GLM-5.1, Qwen 3.5 are within a few points of GPT-5.4 / Claude Opus 4.7 on most benchmarks at 1/20 to 1/50 the price.

!!! tip "Rapid Recall"
    **Closed frontier:** GPT-5.4 (unified routing, OSWorld 75%, $2.50/$15), Claude Opus 4.7 (coding, Constitutional AI), Gemini 3.1 Pro (GPQA 94.3%, cheapest frontier at $2/$12), Grok 4 (HLE 50.7%). **Open-weight frontier:** DeepSeek V4 (1T MoE, $0.28/$0.50, ~50× cheaper), Llama 4 Scout (10M context), Qwen 3.5, Mistral, GLM-5.1, MiniMax M2.7, Kimi K2.5. **India:** Sarvam-M outperforms DeepSeek R1 on some benchmarks at 1/6 size; Saaras V3 ASR beats Gemini 3 Pro and GPT-4o on IndicVoices. **2026 reality:** routing 70% Sonnet + 20% Haiku + 10% Opus replaces "Opus for everything" at 60%+ cost reduction. API prices dropped ~80% from 2025 to 2026.

## §1 The four axes

Four axes define every model.

- **Open vs closed** (can you download weights and self-host?).
- **Frontier vs mid-tier vs small** (capability tier).
- **Native multimodal vs text-first** (audio/video/image as first-class inputs?).
- **Reasoning vs instant** (explicit chain-of-thought with test-time compute vs fast single-pass).

The **"gap between open and closed has nearly closed"** in 2026. Six months ago, top closed models held a commanding lead. As of April 2026, open-weight models from Chinese labs (DeepSeek V4, GLM-5.1, MiniMax M2.7, Kimi K2.5) and Meta (Llama 4) are within a few points on most benchmarks at **1/20 to 1/50 the price**.

## §2 Closed / proprietary frontier

### 2.1 OpenAI — GPT-5.4 (released March 2026)

- Replaces GPT-5.3 Codex as flagship.
- **Key feature:** unified routing architecture — the model itself decides fast-reply vs deep-reasoning based on query complexity.
- 1M token context in Codex mode, native computer use, 47% tool-call token reduction.
- Pricing: $2.50 / $15 per M tokens (standard), $30 / $180 (Pro with max reasoning).
- **Best at:** OSWorld computer use (75%, above human expert), terminal/DevOps, structured reasoning.
- **GPT-4o** is still around as the cheaper multimodal workhorse but is legacy now. When people say "GPT-4o" in 2026 they usually mean "the realtime voice/audio model."

### 2.2 Anthropic — Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5

- **Claude Opus 4.7** (most recent): top of Anthropic's line, highest reasoning depth and code quality.
- **Sonnet 4.6:** ~98% of Opus quality at a fraction of the cost — the workhorse most production apps use.
- **Haiku 4.5:** cheap/fast tier for classification, simple extraction.
- Pricing (Opus 4.6 as reference): $5 / $25 per M tokens, 1M token context.
- **Best at:** coding (powers Cursor, Windsurf, Claude Code), nuanced writing with natural prose, long-context coherence, complex instruction following.
- **Distinctive:** trained with Constitutional AI (see [Constitutional AI and RLAIF](constitutional-ai-and-rlaif.md)) rather than pure RLHF.

### 2.3 Google DeepMind — Gemini 3.1 Pro (released Feb 19, 2026)

- Leads 12 of 18 tracked benchmarks.
- Pricing: $2 / $12 per M tokens — **cheapest frontier model**.
- **Best at:** scientific reasoning (GPQA Diamond 94.3%), abstract reasoning (ARC-AGI-2 77.1%, double predecessor), native multimodal across text/image/audio/video, 1M context.
- **Gemini 2.5 Deep Think** is the reasoning-specific variant.
- **Gemini 3 Flash** is the cheap/fast tier.

### 2.4 xAI — Grok 4 / Grok 4.1 Fast

- **Leads HLE (Humanity's Last Exam) at 50.7%** — the hardest reasoning benchmark.
- Native integration with X/Twitter real-time data.
- Less censored on sensitive topics (deliberate positioning).
- Pricing: $2 / $15 per M tokens.

## §3 Open-weight frontier (the 2026 story)

### 3.1 DeepSeek V4 / DeepSeek-R1

- **V4** released March 2026, trained on Huawei Ascend chips (no NVIDIA at all).
- 1T parameters MoE, $0.28 / $0.50 per M tokens — ~50× cheaper than GPT-5.4.
- **R1** is their reasoning model — uses pure RL with GRPO (see [GRPO](../alignment/grpo.md)), no SFT cold start in R1-Zero variant.
- OpenAI publicly accused DeepSeek in Feb 2026 of distilling from GPT outputs (see [Distillation](distillation.md)).
- **Known limits:** censorship on politically sensitive topics (Chinese regulation), sometimes slow due to long CoT.

### 3.2 Meta — Llama 4 (Scout / Maverick / Behemoth)

- **Scout:** 10M token context window (largest in industry), 17B active params MoE.
- **Maverick:** higher capability tier.
- **Behemoth:** reportedly in training, not released as of April 2026.
- Fully open-weight with commercial license, strong for self-hosting.
- **Real-world coding performance has been disputed** — benchmarks vs actual developer use show a gap.

### 3.3 Alibaba — Qwen 3.5

- Multilingual strength (Chinese, Arabic, Hindi, Spanish).
- Efficient MoE (397B total, 17B active).
- Sonnet-class quality runnable on 2× RTX 4090.
- Ultra-cheap API ($0.11 / 1M input).
- **Distilled variants** (0.8B, 2B, 4B) dominate the small-model leaderboard for latency / cost.

### 3.4 Mistral, Microsoft Phi-4, others

- **Mistral** (French lab): Mistral Small 3, Magistral, Codestral. European alternative, GDPR-friendly.
- **Phi-4** (Microsoft): ~14B params competitive with 70B on reasoning, trained on synthetic textbook-quality data. Great for edge.
- **Other Chinese frontier open-weight** worth knowing: GLM-5 / GLM-5.1 (Z.ai, 94% of Claude Opus 4.6 coding), MiniMax M2.5 / M2.7 (80.2% SWE-bench), Kimi K2 / K2.5 Thinking (Moonshot), Baichuan, Yi, Hunyuan.

## §4 India-specific: Sarvam AI

The one set of models you should be able to speak about fluently as a Hyderabad-based AI PM. Sarvam AI (Bengaluru, backed by IndiaAI Mission).

- **Sarvam-M** (multilingual reasoning LLM): 30B and 105B variants, launched Feb 18, 2026. The 105B model **outperforms DeepSeek R1 (600B) on several benchmarks at 1/6 the size**. 22 scheduled Indian languages plus English, 128K context, efficient MoE (128 experts, top-6 / top-8 routing).
- **Saaras V3** (ASR): **outperforms Gemini 3 Pro and GPT-4o on IndicVoices and Svarah benchmarks** for Indian languages and Indian-accented English. Streaming real-time, trained on 1M+ hours of multilingual audio, optimized for code-mixing and noisy telephony.
- **Bulbul** (TTS for Indian languages).
- **Saarika** (speech-to-text, transcription-focused).
- **Mayura** (text translation).
- **Sarvam Vision** (OCR + document understanding, 93.28% on OmniDoc).
- **Sarvam Audio**: speech-to-command directly (skips the ASR → LLM → TTS cascade).
- Edge models running on feature phones (Nokia/HMD partnership), Bosch cars, Sarvam Kaze smart glasses.

Why this matters for India: 1.45B people, most prefer voice in regional languages. Sarvam has a structural moat — global models cannot match it on Indian languages because training data, speaker diversity, and code-mixing are deeply local.

## §5 The 2026 decision tree

```
What is your task?
├── Need data sovereignty (HIPAA, GDPR, India PDP)?
│   → Self-host Llama 4 / Qwen 3.5 / DeepSeek V4
│   → Or Sarvam-M for Indian language apps
├── Need maximum reasoning?
│   → GPT-5.4 Pro, Claude Opus 4.7 with thinking,
│     Gemini 2.5 Deep Think, or DeepSeek-R1 if budget-constrained
├── Need cheap high-volume?
│   → Gemini 3.1 Pro ($2/$12), Haiku 4.5,
│     or open-weight on vLLM (DeepSeek V4, Qwen 3.5)
├── Need real-time voice?
│   → gpt-realtime (OpenAI), Gemini Live,
│     Sarvam Saaras V3 for Indian languages
├── Need long context (1M+ tokens)?
│   → Gemini 3.1 Pro, Claude Opus 4.7, Llama 4 Scout (10M)
├── Need coding agent?
│   → Claude Opus 4.7 (or Sonnet 4.6 for cost),
│     GPT-5.4, GLM-5.1 if open-weight required
└── Need small/edge?
    → Phi-4, Qwen 3.5 4B, Gemma 3n, or Sarvam edge models
```

## §6 The 2026 pricing collapse

One stat to anchor every cost conversation: **API prices dropped ~80% from 2025 to 2026.** What cost $500/month in mid-2025 runs $50-100 today.

- Input tokens: $0.11 to $30 per M (50× range).
- Output tokens: 3-8× input pricing (median 4×).
- Cached input: 50-90% discount if you hit the cache.

This is why **routing systems now pay back in weeks**. A workload that uses Opus for everything at $25/M output can route 70% to Sonnet ($15/M) and 20% to Haiku ($4/M), keeping 10% on Opus, and drop the bill by 60%+ with negligible quality loss. See [Caching and routing](caching-and-routing.md).

## §7 April 2026 frontier pricing (per M tokens)

| Model | Input | Output | Context | Notes |
|---|---|---|---|---|
| GPT-5.4 | $2.50 | $15 | 272K / 1M | Pro variant $30/$180 |
| GPT-5.4 Pro | $30 | $180 | 1M | Max reasoning |
| Claude Opus 4.7 | $15 | $75 | 1M | Top Anthropic |
| Claude Sonnet 4.6 | $3 | $15 | 1M | 98% Opus quality |
| Claude Haiku 4.5 | $0.25 | $1.25 | 200K | Cheap tier |
| Gemini 3.1 Pro | $2 | $12 | 1M | Cheapest frontier |
| Grok 4 | $2 | $15 | 256K | HLE leader |
| DeepSeek V4 | $0.28 | $0.50 | 128K | 50× cheaper |
| Qwen 3.5 Plus | $0.11 | $0.33 | 128K | Cheapest competent |
| Llama 4 Maverick (self-host) | GPU cost | GPU cost | 1M | Open-weight |
| Sarvam-M 105B | Indian rates | — | 128K | Indian-language specialist |

## §8 The "what should I use" quick picks

- **Default chat app** → Claude Sonnet 4.6 (quality/cost sweet spot).
- **Code generation** → Claude Opus 4.7, Sonnet 4.6 for cost, or GLM-5.1 open-weight.
- **Graduate-level reasoning** → Gemini 3.1 Pro or GPT-5.4 Pro.
- **Math olympiad** → DeepSeek-R1 or Gemini 2.5 Deep Think.
- **Real-time voice (English)** → OpenAI gpt-realtime.
- **Real-time voice (Indian languages)** → Sarvam cascade (Saaras + Sarvam-M + Bulbul).
- **Cheap classification/extraction** → Haiku 4.5, Gemini Flash, or Qwen 3.5 small.
- **Long-document analysis** → Gemini 3.1 Pro or Claude Opus 4.7 (1M context).
- **Agent with tool use** → GPT-5.4 (best OSWorld) or Opus 4.7.
- **Self-host for data sovereignty** → Qwen 3.5 or Llama 4 on vLLM.

## §9 Post-training recipe stack (who uses what)

| Lab | Recipe | Signature technique |
|---|---|---|
| OpenAI (GPT-5.x) | SFT + RLHF (PPO) + RLAIF + RLVR for reasoning | Unified router, CoT compression |
| Anthropic (Claude) | SFT + **Constitutional AI** + RLAIF | Self-critique, constitution |
| Google (Gemini) | SFT + RLHF + RLAIF + multimodal RLVR | Deep Think mode |
| DeepSeek | SFT + **GRPO + RLVR** | Pure-RL reasoning (R1-Zero), rule-based rewards |
| Meta (Llama 4) | SFT + DPO + RLAIF | Open-weight release |
| Alibaba (Qwen 3.5) | SFT + DPO + GRPO | MoE efficiency |
| Sarvam | SFT on Indian data + DPO + domain RLHF | Multilingual from-scratch pretraining |

## Interview Questions

**Q1: Walk me through the 2026 model landscape and how you would choose one for a new product.**

Four axes: open vs closed, capability tier, multimodal vs text, reasoning vs instant. For a new product in 2026, I would build a routing architecture rather than pick one model. Cheap queries (intent classification, simple extraction) go to Haiku 4.5 or Gemini Flash at under $1/M tokens. Medium queries go to Sonnet 4.6 or Gemini 3.1 Pro. Complex reasoning escalates to Opus 4.7, GPT-5.4 Pro, or DeepSeek-R1 for cost-sensitive cases. If data sovereignty matters, self-host Llama 4 or Qwen 3.5 via vLLM. For Indian-language voice workloads, Sarvam's stack is non-negotiable. The decision is 50% about cost-per-task, 30% about latency budget, 20% about data/compliance.

**Q2: What does the open/closed gap "nearly closing" actually mean operationally?**

Six months ago, top closed models (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5) held a commanding lead. As of April 2026, open-weight models from Chinese labs (DeepSeek V4, GLM-5.1, MiniMax M2.7, Kimi K2.5) and Meta (Llama 4) are within a few points on most benchmarks at 1/20 to 1/50 the price. Operationally: for data-sovereignty workloads or cost-sensitive verticals, self-hosting an open-weight model on vLLM is now a real production option, not just a cost-savings experiment. The closed-model premium is justified only for the very hardest reasoning and tool-use tasks.

**Q3: Why is routing the winning architecture and not a single best model?**

Cost. Output token pricing spans 50× across providers (Qwen 3.5 at $0.33 to GPT-5.4 Pro at $180). Most user queries are simple and can be answered by Haiku or Flash; a small fraction need Opus or GPT-5.4 Pro. Routing captures that distribution: send simple queries to cheap models, escalate when complexity demands. Reported savings 50-80% vs "Opus for everything." Plus, routing systems are easier to evolve as new models drop into the catalog.

**Q4: What is Sarvam's structural advantage and why can global frontier models not catch up?**

Three things. Training data: Sarvam has access to deeply local Indian-language data (22 scheduled languages, code-mixing, telephony noise, regional accents) that is not in the web crawls global labs use. Speaker diversity: Saaras V3's training set has 1M+ hours of Indian-accent audio. Code-mixing: "Haan ji, mera account number 12345 hai, I want to check balance" is normal Indian speech; global models trained mostly on monolingual English do not handle this naturally. Result: Sarvam-M outperforms DeepSeek R1 (~6× bigger) on Indian benchmarks; Saaras V3 beats Gemini 3 Pro and GPT-4o on IndicVoices.

**Q5: How would you reduce a company's AI API costs by 50% in 2026?**

Two levers in order. First, model routing: send simple queries to Haiku 4.5 or Gemini Flash, escalate to Sonnet 4.6 or GPT-5, reserve Opus and GPT-5.4 Pro for genuinely hard cases. Typically captures 50 to 70% savings on its own. Second, semantic caching: compute embeddings of incoming queries, match against a vector cache, serve cached responses when similarity exceeds threshold. Customer-support workloads often hit 40 to 70% cache rates. Together, 70 to 80% cost reduction is realistic. Tertiary optimizations: prompt caching at the provider level (50 to 90% off on cached prefixes), moving high-volume workloads to self-hosted vLLM with quantized open-weight models. See [Caching and routing](caching-and-routing.md).
