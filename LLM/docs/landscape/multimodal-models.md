# Multimodal Models — How They Are Trained

Modern multimodal models (GPT-4o, Gemini 3.1, Claude Opus 4.7) do not have separate "vision towers" and "text towers" stapled together. They have a **single transformer** that ingests all modalities as tokens — text tokens, image patch tokens, audio frame tokens — in one stream. Training is joint from the start. This page covers the three production architectures, the four-stage training recipe, and the design knobs that matter.

!!! tip "Rapid Recall"
    Three architectures in production. (1) **Encoder-cascade** (legacy, GPT-4V, early LLaVA): separate vision encoder (CLIP ViT) projects to LLM token space; LLM processes text + visual tokens. One-way only. (2) **Unified transformer** (current SOTA, GPT-4o, Gemini 3.1 Pro, Claude Opus 4.7): single model, single weights, no separate encoders. Each modality has a tokenizer (text BPE, image ViT patches, audio EnCodec / Mimi, video keyframes). Tokens interleaved in one sequence. Model can understand AND generate across modalities. (3) **MoE multimodal** (Llama 4): modality-specific experts. **Training recipe:** multimodal pretrain → modality-balanced SFT → multimodal RLHF/DPO/RLVR → specific-modality alignment. **Effective context ≠ advertised context** (RULER benchmark) — multimodal makes this worse because image tokens are dense.

## §1 The simple story

Modern multimodal models do not have separate "vision towers" and "text towers" stapled together. They have a single transformer that ingests all modalities as tokens — text tokens, image patch tokens, audio frame tokens — in one stream. Training is joint from the start.

## §2 The three architectures

### 2.1 Encoder-cascade (legacy — GPT-4V, early LLaVA)

- Separate vision encoder (usually CLIP ViT) → projects to LLM's token space.
- LLM processes text + "visual tokens."
- **Limits:** one-way (LLM understands images, cannot generate them); awkward for video/audio.
- Still common for cheap VQA deployments.

### 2.2 Unified transformer (current SOTA — GPT-4o, Gemini 3.1 Pro, Claude Opus 4.7)

- **Single model**, single set of weights, no separate encoders.
- Each modality has a **tokenizer**:
    - Text: standard BPE.
    - Image: ViT-style patches OR VQ-VAE tokens (Chameleon-style).
    - Audio: EnCodec / Mimi discrete audio tokens OR raw mel-spectrograms.
    - Video: image tokens per frame + temporal positional encoding.
- All tokens interleaved in a single sequence.
- Model can both **understand and generate** across modalities.

### 2.3 Mixture-of-Experts multimodal (Llama 4, many Chinese models)

- MoE layers with **modality-specific experts** — text tokens routed to text experts, image tokens to image experts.
- Conditionally activated: only 17B of 397B parameters fire for any token.
- Scale capacity without proportional compute.

## §3 Training recipe — how a frontier multimodal model gets built

### 3.1 Stage 1 — multimodal pretraining

- **Interleaved data:** web pages with images/text mixed, video frames + audio + transcripts, documents + figures.
- Objective: **next-token prediction across all modalities**.
- Scale: trillions of text tokens + billions of images + millions of hours of audio/video.
- Cost: $100M+.

### 3.2 Stage 2 — modality-balanced SFT

- Curated instruction-tuning data per modality:
    - Image understanding: "Describe this chart" / "What is in this photo?"
    - Audio: "Transcribe this" / "Summarize this meeting."
    - Cross-modal: "Based on the video and this document, answer..."
- Critical: prevent **modality collapse** (model ignoring one input type).

### 3.3 Stage 3 — multimodal RLHF / DPO / RLVR

- Preference data across modalities.
- Verifiable rewards where possible (image correctly described → OCR check).

### 3.4 Stage 4 — specific modality alignment

- Audio: reduce hallucinations in transcription (OpenAI's 2025 snapshots cut this 90% vs Whisper v2).
- Video: temporal consistency ("what happened between 0:30 and 1:00?").
- Document: layout-aware reading order (Sarvam Vision's 93% on OmniDoc).

## §4 Key design decisions

### 4.1 Patch size for vision

- Smaller patches (14×14 or 16×16) → more tokens per image → more compute but finer detail.
- Larger patches → fewer tokens → cheaper but loses small text in images.

### 4.2 Audio tokenization

- Discrete (EnCodec, Mimi) → allows audio generation; fixed vocabulary.
- Continuous (mel-spectrograms) → higher quality analysis but cannot be generated directly.

### 4.3 Cross-modal grounding

- "Which pixel does 'the red dog' refer to?" — required for editing, pointing tasks.
- Achieved via segmentation-aware training (SAM 2 integration in some models).

## §5 Limitations interviewers will probe

1. **Effective context ≠ advertised context.** RULER benchmark shows models reliably use 50-65% of their advertised context window; multimodal makes this worse because image tokens are dense.
2. **Temporal reasoning in video.** Still weak. Most "video" models really process keyframes + audio, not full motion.
3. **Fine-grained text in images.** Tokenization loses small text. OCR pipelines still outperform native multimodal on document parsing.
4. **Generation quality.** Native image generation quality (GPT-4o, Gemini 3) catching up to dedicated diffusion models (Flux, Imagen) but not yet at parity for professional use.
5. **Cross-modal hallucination.** Model sees one thing, describes another. Harder to detect than text hallucination.

## §6 The contrastive-learning backbone (CLIP)

The vision-language joining trick most modern multimodal models inherit comes from CLIP (OpenAI, 2021). Two encoders (image and text) projecting into one shared vector space, trained on 400M (image, caption) pairs. The InfoNCE / symmetric contrastive loss:

\[ L = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{e^{\text{sim}(u_i, v_i)/\tau}}{\sum_j e^{\text{sim}(u_i, v_j)/\tau}} + \log \frac{e^{\text{sim}(u_i, v_i)/\tau}}{\sum_j e^{\text{sim}(u_j, v_i)/\tau}} \right] \]

`sim` is cosine similarity; `τ` is a learnable temperature. See [Architecture families §3](../foundations/architecture-families.md) for the full CLIP treatment.

Modern unified multimodal models do not use CLIP directly but inherit the contrastive idea: shared embedding space such that "a photo of a dog" and the text "a dog" end up close together.

## §7 For an AI PM in 2026

Multimodal is the **default assumption** for any new product in 2026, not a nice-to-have. The question is not "should we support images?" — it is "what multimodal tasks specifically unlock value in our domain?" For India: voice + images of handwritten notes, schedules, receipts in regional scripts is the killer combo.

## Interview Questions

**Q1: Explain how a unified multimodal transformer differs from a cascade of encoders.**

Cascade architecture (GPT-4V, LLaVA) has a separate vision encoder (typically CLIP ViT) that produces visual tokens, then the LLM processes a sequence of text tokens + visual tokens. One-way only — the model can describe images but cannot generate them. Unified transformer (GPT-4o, Gemini 3.1, Claude Opus 4.7) uses one model with one set of weights; each modality has its own tokenizer (text BPE, image ViT patches, audio EnCodec) and all tokens flow into the same transformer. Result: the model can both understand and generate across modalities, and cross-modal reasoning is native rather than mediated by the cascade.

**Q2: What is modality collapse and how do you prevent it?**

When a multimodal model learns to ignore one input modality and rely only on the others. Common failure: trained on lots of "describe this image" data, the model learns that text descriptions are mostly predictable from the question itself plus general world knowledge, so it stops actually attending to image tokens. Detection: ablate the image input — if performance does not drop, the model was ignoring it. Prevention: modality-balanced SFT data with tasks that absolutely require each modality (counting objects in images, transcribing specific audio, reading exact text from screenshots).

**Q3: Why does "effective context" matter more than advertised context for multimodal?**

The RULER benchmark shows most LLMs reliably use 50 to 65% of their advertised context. Multimodal makes it worse because image tokens are dense — a single image at high resolution can consume 1000+ tokens, eating context budget fast. A 1M-token advertised window with one high-res screenshot in the middle might effectively give you usable attention over only the first 100K tokens. Operational implication: down-sample images, use lower-resolution variants for less critical context, and benchmark on RULER (not raw context length) when picking a model for long multimodal documents.

**Q4: Where does CLIP-style contrastive learning fit in modern multimodal training?**

It is the vision-language joining trick that made multimodal possible. CLIP trains two encoders to produce embeddings such that matching (image, caption) pairs have high cosine similarity and non-matching pairs do not. The shared embedding space lets you do zero-shot classification, retrieval, and cross-modal search. Modern unified multimodal models do not use CLIP as a separate encoder but inherit the contrastive idea: their training data includes (image, caption) pairs and the loss implicitly enforces the same alignment. Many production systems still use CLIP or SigLIP as a fast bi-encoder for image search and retrieval-augmented multimodal pipelines.

**Q5 (Trap): Are unified multimodal models always better than cascade architectures?**

No, for two reasons. Cost: cascade systems can use a tiny vision encoder (CLIP-Tiny) plus a small LLM, much cheaper than a frontier unified model for simple VQA. Maturity: dedicated diffusion models (Flux, Imagen) still beat unified multimodal generation on professional-grade images. The unified architecture's selling point is cross-modal reasoning and generation at once — for narrow tasks like "extract text from a screenshot," a cascade with OCR + LLM can be both cheaper and more accurate than asking a unified multimodal model to do it natively.
