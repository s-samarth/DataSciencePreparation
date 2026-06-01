# Architecture Families and the Task → Architecture Map

Attention is just a mechanism. What matters is what you connect via attention. Encoder-only, decoder-only, and encoder-decoder are the three families; ViT, BERT, CLIP, and Whisper are the canonical examples; bi-encoder and cross-encoder are the retrieval/ranking patterns built on top.

!!! tip "Rapid Recall"
    Three families: **encoder-only** for understand-then-label (BERT), **decoder-only** for generate-anything (GPT, Claude), **encoder-decoder** for source-to-different-target (T5, Whisper). Self-attention has Q=K=V from one sequence; cross-attention has Q from a decoder and K, V from an encoder. **Bi-encoder vs cross-encoder is a different concept**: bi-encoder encodes inputs separately and compares vectors (fast retrieval, billions of docs); cross-encoder concatenates inputs and encodes jointly (slow, accurate, ~100 candidates). Production RAG = bi-encoder retrieve → cross-encoder rerank → LLM generate.

## §1 Vision Transformer (ViT)

"What if we just apply a transformer to images?" And it worked, at scale.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 150" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="11">
<g transform="translate(20,35)">
<rect width="80" height="80" fill="#ebe0cf" stroke="#1a1410" stroke-width="1.3"/>
<line x1="0" y1="27" x2="80" y2="27" stroke="#d4c4ab"/><line x1="0" y1="54" x2="80" y2="54" stroke="#d4c4ab"/>
<line x1="27" y1="0" x2="27" y2="80" stroke="#d4c4ab"/><line x1="54" y1="0" x2="54" y2="80" stroke="#d4c4ab"/>
<text x="40" y="100" text-anchor="middle" font-size="10">16×16 patches</text>
</g>
<text x="130" y="78" font-size="20">→</text>
<g transform="translate(160,55)">
<rect width="26" height="40" fill="#1a1410"/><text x="13" y="25" fill="#f4ede2" text-anchor="middle" font-size="9">CLS</text>
<rect x="30" width="26" height="40" fill="#1f5e5b"/><rect x="60" width="26" height="40" fill="#1f5e5b"/>
<rect x="90" width="26" height="40" fill="#1f5e5b"/><rect x="120" width="26" height="40" fill="#1f5e5b"/>
<text x="85" y="58" text-anchor="middle" font-size="9">patch embeds + pos</text>
</g>
<text x="320" y="78" font-size="20">→</text>
<rect x="350" y="45" width="120" height="60" rx="5" fill="#b5462a" stroke="#1a1410" stroke-width="1.3"/>
<text x="410" y="72" text-anchor="middle" fill="#f4ede2" font-size="11">Transformer</text>
<text x="410" y="88" text-anchor="middle" fill="#f4ede2" font-size="10">encoder ×12</text>
<text x="490" y="78" font-size="20">→</text>
<rect x="520" y="50" width="150" height="50" rx="5" fill="#c08a2d" stroke="#1a1410" stroke-width="1.3"/>
<text x="595" y="72" text-anchor="middle" font-size="10">CLS → MLP head</text>
<text x="595" y="88" text-anchor="middle" font-size="10">→ class</text>
</svg>
<figcaption>Image → 16×16 patches → patch tokens + CLS + positions → transformer encoder → CLS head → class.</figcaption>
</figure>

**The pipeline:** split image into 16×16 patches (224² → 196 patches) → linearly project each flattened patch to an embedding (treat each patch as a token) → prepend a learnable `[CLS]` token → add learned positional embeddings → standard transformer encoder (bidirectional, no mask) → CLS final state → MLP head → class. **The only conv-flavored op is the patch projection.**

!!! abstract "The data-scaling shock"
    On ImageNet-1K, ViT *loses* to ResNet. On ImageNet-21K it matches. On JFT-300M it *beats* ResNet. CNNs hardcode locality, translation-equivariance, and hierarchy as priors that help with small data; with enough data, ViT learns better priors from scratch. DeiT later made ViT work on ImageNet-1K alone via aggressive augmentation plus distillation.

Why ViT mattered most: it made vision *interoperable* with the transformer ecosystem. Two transformers (vision + language) compose trivially, which is what made modern multimodal LLMs possible. Family: DeiT, Swin (hierarchical, shifted windows), MAE (masked self-supervised), DINOv2, SAM, and CLIP/SigLIP encoders inside every VLM.

## §2 BERT — the encoder-only era

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers (Google, Oct 2018). Just the encoder half: bidirectional attention, no causal mask. It established "pretrain once, fine-tune everywhere." Base: 12 layers / 110M; Large: 24 layers / 340M. Post-norm, GELU, WordPiece (~30K vocab), learned positions, plus a *segment embedding* (sentence A vs B).

### Two pretraining objectives

**Masked LM (MLM):** Mask 15% of tokens, predict them from bidirectional context. The famous **80/10/10** split: of the chosen 15%, 80% → `[MASK]`, 10% → random token, 10% → unchanged. Prevents train/inference mismatch (`[MASK]` never appears at inference) by forcing useful representations for *every* position.

**Next Sentence Prediction (NSP):** Predict whether sentence B follows A. Turned out to be a *bad* objective; RoBERTa removed it and improved. Too easy (mostly topic detection), taught little.

### How it is used

- **Classification:** `[CLS]` rep → linear → softmax. Loss: cross-entropy.
- **Token classification (NER):** per-token rep → tag (BIO scheme). Loss: per-token CE.
- **Extractive QA (SQuAD):** two heads predict start and end span positions. Loss: CE on start + CE on end.
- **Sentence pairs:** `[CLS] A [SEP] B [SEP]` → relation (cross-encoder), or Sentence-BERT bi-encoder for similarity at scale.

**Inference:** single forward pass, no autoregression, no KV-cache. Fast and cheap. Still used heavily in 2026 for high-volume classification (spam, moderation, ranking, embeddings) where LLMs are overkill. Family: RoBERTa (BERT done right), ALBERT, DistilBERT, ELECTRA, DeBERTa, Sentence-BERT, mBERT.

## §3 CLIP — contrastive vision-language

Two *independent* encoders (image: ViT/ResNet; text: GPT-2-style) projecting into one shared vector space. Trained on 400M web (image, caption) pairs. After L2-normalization, similarity is just a dot product. **CLIP is a bi-encoder.**

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 460 230" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="11">
<text x="35" y="20" font-size="11" fill="#b5462a" font-weight="700">images ↓</text>
<text x="210" y="20" font-size="11" fill="#1f5e5b" font-weight="700">texts →</text>
<g transform="translate(70,30)">
<text x="20" y="-6" text-anchor="middle" font-size="10">T1</text><text x="60" y="-6" text-anchor="middle" font-size="10">T2</text>
<text x="100" y="-6" text-anchor="middle" font-size="10">T3</text><text x="140" y="-6" text-anchor="middle" font-size="10">T4</text>
<text x="-12" y="24" text-anchor="end" font-size="10">I1</text><text x="-12" y="64" text-anchor="end" font-size="10">I2</text>
<text x="-12" y="104" text-anchor="end" font-size="10">I3</text><text x="-12" y="144" text-anchor="end" font-size="10">I4</text>
<g stroke="#1a1410" stroke-width="1">
<rect x="0" y="0" width="40" height="40" fill="#b5462a"/><rect x="40" y="0" width="40" height="40" fill="#ebe0cf"/><rect x="80" y="0" width="40" height="40" fill="#ebe0cf"/><rect x="120" y="0" width="40" height="40" fill="#ebe0cf"/>
<rect x="0" y="40" width="40" height="40" fill="#ebe0cf"/><rect x="40" y="40" width="40" height="40" fill="#b5462a"/><rect x="80" y="40" width="40" height="40" fill="#ebe0cf"/><rect x="120" y="40" width="40" height="40" fill="#ebe0cf"/>
<rect x="0" y="80" width="40" height="40" fill="#ebe0cf"/><rect x="40" y="80" width="40" height="40" fill="#ebe0cf"/><rect x="80" y="80" width="40" height="40" fill="#b5462a"/><rect x="120" y="80" width="40" height="40" fill="#ebe0cf"/>
<rect x="0" y="120" width="40" height="40" fill="#ebe0cf"/><rect x="40" y="120" width="40" height="40" fill="#ebe0cf"/><rect x="80" y="120" width="40" height="40" fill="#ebe0cf"/><rect x="120" y="120" width="40" height="40" fill="#b5462a"/>
</g>
</g>
<text x="240" y="100" font-size="12" fill="#3a2e22">↑ diagonal = correct</text>
<text x="240" y="120" font-size="12" fill="#3a2e22">pairs (push UP)</text>
<text x="240" y="150" font-size="12" fill="#3a2e22">off-diagonal = wrong</text>
<text x="240" y="170" font-size="12" fill="#3a2e22">pairs (push DOWN)</text>
</svg>
<figcaption>Contrastive training: for a batch of N pairs, build the N×N cosine-similarity matrix. Diagonal = correct (push up), off-diagonal = wrong (push down).</figcaption>
</figure>

**Training:** for a batch of N pairs, build the N×N cosine-similarity matrix. Symmetric cross-entropy: each image should pick its true caption (row-wise softmax) and each caption its true image (column-wise). A learnable temperature sharpens it.

The full math (InfoNCE / symmetric contrastive):

\[ L = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{e^{\text{sim}(u_i, v_i)/\tau}}{\sum_j e^{\text{sim}(u_i, v_j)/\tau}} + \log \frac{e^{\text{sim}(u_i, v_i)/\tau}}{\sum_j e^{\text{sim}(u_j, v_i)/\tau}} \right] \]

`sim` is cosine similarity; `τ` is a learnable temperature.

!!! note "Why huge batches matter"
    Contrastive signal strength scales with negatives per positive. Batch of 32K → each image is contrasted against 32,767 wrong captions. Small batches = weak signal. This is why CLIP-scale training needs massive multi-GPU infra. **Zero-shot classification:** encode class prompts ("a photo of a {class}") once, encode the image, take the nearest by cosine. No task training. CLIP matched supervised ResNet-50 on ImageNet zero-shot.

Successors: SigLIP (sigmoid loss, no huge-batch need), EVA-CLIP, CoCa, InternVL.

## §4 Whisper — speech as seq2seq

An **encoder-decoder** transformer (OpenAI, Sep 2022). Speech recognition is fundamentally seq2seq: audio → text. Trained on 680K hours (V3: 5M+ hours) of weakly supervised web audio.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 150" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="10">
<rect x="15" y="55" width="80" height="40" rx="4" fill="#ebe0cf" stroke="#1a1410" stroke-width="1.3"/>
<text x="55" y="72" text-anchor="middle" font-size="10">audio 16kHz</text>
<text x="55" y="86" text-anchor="middle" font-size="10">30s chunk</text>
<text x="105" y="80" font-size="16">→</text>
<rect x="125" y="50" width="95" height="50" rx="4" fill="#fbf6ee" stroke="#b5462a" stroke-width="1.3"/>
<text x="172" y="70" text-anchor="middle" font-size="10">log-Mel</text>
<text x="172" y="84" text-anchor="middle" font-size="10">(80, 3000)</text>
<text x="228" y="80" font-size="16">→</text>
<rect x="248" y="50" width="100" height="50" rx="4" fill="#1f5e5b" stroke="#1a1410" stroke-width="1.3"/>
<text x="298" y="70" text-anchor="middle" fill="#f4ede2" font-size="10">Encoder</text>
<text x="298" y="84" text-anchor="middle" fill="#f4ede2" font-size="9">2×conv→layers</text>
<text x="356" y="80" font-size="16">→</text>
<rect x="376" y="50" width="110" height="50" rx="4" fill="#b5462a" stroke="#1a1410" stroke-width="1.3"/>
<text x="431" y="68" text-anchor="middle" fill="#f4ede2" font-size="10">Decoder</text>
<text x="431" y="82" text-anchor="middle" fill="#f4ede2" font-size="9">self+cross attn</text>
<text x="494" y="80" font-size="16">→</text>
<rect x="514" y="55" width="170" height="40" rx="4" fill="#1a1410" stroke="#1a1410"/>
<text x="599" y="79" text-anchor="middle" fill="#f4ede2" font-size="10">text tokens (autoregressive)</text>
</svg>
<figcaption>Audio (16 kHz, 30-second chunk) → log-Mel spectrogram → encoder → decoder (self + cross attention) → autoregressive text tokens.</figcaption>
</figure>

**Preprocessing:** raw 16 kHz audio is too long (480K samples / 30s) for O(N²) attention. Convert to a log-Mel spectrogram (80/128 bins, 25 ms windows, 10 ms stride) → (80, 3000). Mel scale matches human pitch perception; log compresses dynamic range. Like patches for an image.

**Encoder:** two strided 1D convs (downsample to 1500 frames) + sinusoidal positions + transformer layers (bidirectional). **Decoder:** causal self-attention over generated text + *cross-attention to the 1500 acoustic frames* + FFN, autoregressive. Loss: cross-entropy on transcript tokens.

!!! abstract "Task-controlling special tokens"
    One model, 99 languages, multiple tasks, via the decoder prompt: `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`. Swap `<|en|>` → `<|fr|>` for French; `<|transcribe|>` → `<|translate|>` to translate to English. Same idea as instruction tuning, but predates it. Skip the language token and the model predicts it (language ID).

**Limits:** 30-second chunks (boundary issues), hallucination on silence/noise, no speaker diarization, slow autoregressive decode, not streaming. **Landscape:** Whisper Large V3 / Turbo, Distil-Whisper, NVIDIA Parakeet (low-latency non-AR), SeamlessM4T (speech-to-speech), and audio LLMs (Qwen-Audio, GPT-4o) that make audio a native LLM modality.

## §5 Self vs cross attention, bi vs cross encoders

### Self vs cross attention — same op, different inputs

| Self-attention | Cross-attention |
|---|---|
| Q, K, V all from the *same* sequence. The sequence searches itself. `Q = K = V = X`. Maintains coherence within a sequence. | Q from sequence A, K and V from sequence B. One sequence searches another. Grounds a decoder in an encoder's output. |

An **encoder-decoder decoder block has THREE sub-blocks:** (1) masked self-attention over its own past tokens (autoregressive coherence), (2) cross-attention to the encoder output (grounding, no mask), (3) FFN. **Decoder-only models (GPT) use only self-attention:** the prompt and output are one sequence, no separate encoder to cross-attend to. Cross-attention also powers Stable Diffusion (image queries text), Perceiver, and Flamingo.

### Bi-encoder vs cross-encoder — a retrieval pattern

These are retrieval/ranking *patterns*, not attention types. (Note: a cross-encoder internally uses *self*-attention over the concatenated pair. The shared word "cross" is an unfortunate coincidence.)

| | Bi-encoder | Cross-encoder |
|---|---|---|
| How | Encode the two inputs **separately**, compare vectors by cosine | Concatenate both inputs, encode **jointly** with full attention → one score |
| Speed | Very fast (precompute one side) | Slow (one forward pass per pair) |
| Accuracy | Lower | Higher (fine-grained interaction) |
| Precompute? | Yes — store doc vectors | No |
| Scale | Billions of docs | ~100 candidates |
| Use for | Retrieval / candidate generation | Reranking / final scoring |

!!! abstract "The production pattern: retrieve → rerank"
    **Bi-encoder** retrieves top ~100 from millions in milliseconds (precomputed vectors). **Cross-encoder** reranks those 100 accurately. You cannot cross-encode millions (a forward pass per pair does not scale). This is Google Search, every RAG system, every modern search stack. CLIP is a bi-encoder; the SQuAD-style `[CLS] A [SEP] B` setup is a cross-encoder.

## §6 Task → architecture decision map

Three families, three I/O shapes. The selection rule to burn into memory.

| You have | Use | Trained how |
|---|---|---|
| Text → label | Encoder + classifier head | MLM pretrain → CE on head |
| Text → per-token tag (NER) | Encoder + token head | Per-token CE |
| Text → span (QA) | Encoder + start/end heads | CE on start + CE on end |
| Text → fixed vector | Encoder as bi-encoder | Contrastive (InfoNCE) |
| Pair → similarity score | Encoder as cross-encoder | CE on relation |
| Audio → text | Enc-Dec (Whisper) | CE on target tokens |
| Lang A → Lang B | Enc-Dec | CE, teacher forcing |
| Doc → summary | Enc-Dec or Decoder LLM | CE on summary |
| Prompt → generation | Decoder (GPT) | Next-token pretrain |
| Instruction → answer | Decoder + instruction tuning | Pretrain → SFT → RLHF/DPO |
| Chat with tools | Decoder + function calling | SFT on tool-use traces |
| Tabular → prediction | **XGBoost first**; transformer only if conditions met | — |

### Transformers for tabular data

!!! abstract "The 2026 verdict"
    **Default to XGBoost.** Tree models handle tabular inductive biases better (robust to uninformative features, no normalization needed) and they fit the naturally jagged decision boundaries that NNs oversmooth. Tabular datasets are usually small, where transformers' data hunger hurts.

Transformers (FT-Transformer, TabPFN, SAINT) treat each feature as a token; self-attention learns feature interactions; CLS → prediction. Consider them only when: **>1M rows**, **mixed-modality** tabular (free text/images alongside columns), need for **row embeddings** for downstream retrieval, or **temporal/sequential** structure. Otherwise XGBoost / CatBoost / LightGBM win on speed, robustness, and deployment simplicity.

## §7 Contrastive learning losses (quick reference)

Learn representations by **pulling similar things together and pushing dissimilar things apart** in vector space, instead of predicting a label. Each step needs an **anchor**, a **positive** (similar), and **negatives** (dissimilar, usually the rest of the batch).

| Triplet loss | InfoNCE (modern default) |
|---|---|
| `L = max(0, d(a,p) − d(a,n) + margin)` | `L = −log( exp(sim(a,p)/τ) / Σᵢ exp(sim(a,xᵢ)/τ) )` |
| "Anchor should be closer to positive than to negative, by at least `margin`." If already true → loss 0. Margin prevents collapse to zero distance. Uses one negative per step. Older (FaceNet). | Cross-entropy in disguise: "pick the positive out of N candidates." Softmax over similarities with the positive as the label. Uses N−1 negatives per step → richer signal. CLIP, SimCLR, embedding models. |

!!! note "Core mental model"
    Triplet = "make this pair closer than that pair, by a margin" (pairwise). InfoNCE = "classify the positive out of a lineup" (softmax framing). Same goal, different framing. InfoNCE won because scaling negatives is trivial (bigger batch = more in-batch negatives), giving a much richer gradient. Whenever you see "contrastive loss" today, assume InfoNCE.

## Interview Questions

**Q1 (Trap): ViT is permutation-invariant over patches — does shuffling change the output?**

Attention is invariant, but ViT adds positional embeddings. Shuffle patches without their positional embeddings → output changes (each patch now carries a different position). Shuffle both together → invariant, but positions are baked to specific locations during training, so it never happens in practice.

**Q2 (Trap): If BERT is bidirectional, can it generate text?**

Technically yes (iteratively unmask), but practically no. It was not trained for coherent left-to-right generation. That is why generation became GPT's domain and modern LLMs descend from GPT, not BERT.

**Q3 (Trap): Why not one shared encoder for image and text in CLIP?**

Images and text have totally different structure and need different architectures. CLIP shares the *output space*, not parameters. Separate encoders, jointly trained so their vectors are comparable. The shared latent space, not shared weights, is what enables contrastive learning and zero-shot transfer.

**Q4 (Trap): Why does Whisper hallucinate on silence?**

It is a generative AR language model; it must emit *some* token. On silent or noisy audio, cross-attention gives an ambiguous signal and the decoder's LM side fills in whatever is most likely from prior text context. Fix with voice-activity detection (VAD) preprocessing and confidence thresholding.

**Q5 (Trap): If decoder-only does everything via prompting, why do encoder-only models still exist?**

Three reasons: **cost/latency** (a fine-tuned BERT is 10 to 100× cheaper than prompting an LLM at high volume), **embeddings** (you need a bi-encoder for fixed-size retrieval vectors), and **reliability** (extractive QA cannot hallucinate; it must return a span). Encoder-only is the production workhorse; decoder-only is the flexible frontier.
