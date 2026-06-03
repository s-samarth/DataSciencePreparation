# Tokenization & Byte Pair Encoding

Networks eat numbers, not text, so before any modeling you split text into units and map them to integer IDs. The winning unit is the subword, and Byte Pair Encoding is the greedy merge algorithm behind almost every modern tokenizer.

!!! tip "Rapid Recall"
    Word tokens explode the vocabulary and hit out-of-vocabulary words; character tokens give tiny vocab but brutally long sequences; subwords are the winner (common words whole, rare words split, bounded vocab, no OOV). BPE trains by repeatedly merging the most frequent adjacent pair; encoding applies the learned merges greedily, which is the no-OOV property in action. Byte-level BPE (GPT-2+) starts from raw bytes so any string is representable; WordPiece (BERT) merges by likelihood; Unigram/SentencePiece (T5) prunes top-down and is language-agnostic. Special tokens like `<PAD>` (must be masked), `<SOS>`, `<EOS>`, `<CLS>`, `<SEP>` are structural markers.

## §1 Levels of tokenization

| Level | Token | Problem it has |
| --- | --- | --- |
| **Word** | a word | Vocabulary explosion (millions of words); **out-of-vocabulary** (any unseen word → `<UNK>`); no morphology ("run"/"running" unrelated IDs). |
| **Character** | a character | Tiny vocab, zero OOV, but sequences become brutally long ("internationalization" = 20 tokens); the model must reassemble meaning from letters. |
| **Subword** ✓ | a frequent chunk | The winner. Common words stay whole, rare words split ("tokenization"→"token"+"ization"). Bounded vocab (30k to 100k), **no OOV ever**, partial morphology. Every modern LLM uses this. |

## §2 Byte Pair Encoding (BPE), a greedy merge algorithm

**Training the tokenizer** (once, on a corpus):

1. Base vocab = all individual characters; represent each word as a char sequence (often with an end-of-word marker `</w>`).
2. Count every adjacent symbol pair's frequency across the corpus.
3. Merge the **single most frequent pair** into one new symbol; add to vocab; record the merge rule.
4. Repeat steps 2 to 3 until vocab hits the target size (for example 30,000 merges).

**Worked micro-example.** Corpus: `low`×5, `lower`×2, `newest`×6, `widest`×3.

```
low    ×5 → l o w </w>        Pair "e s" appears in newest(×6)+widest(×3) = 9 → most frequent
lower  ×2 → l o w e r </w>
newest ×6 → n e w e s t </w>    Merge 1: e s → es      newest = n e w es t
widest ×3 → w i d e s t </w>    Merge 2: es t → est    newest = n e w est
                                Merge 3: l o → lo      then lo w → low ...
```

**Encoding new text**: split into chars, then apply the learned merges *in order*, greedily. "lowest" (never seen) → `l o w e s t` → es → est → lo → low → result `["low","est"]`. **That's the no-OOV property in action.**

| Variant | How it differs |
| --- | --- |
| **Byte-level BPE** (GPT-2+) | Start from the **256 raw bytes**, not characters. *Any* string, emoji, Chinese, code, is representable with zero OOV. The de-facto LLM standard. |
| **WordPiece** (BERT) | Merge the pair that most increases **training-data likelihood** (not raw frequency). Marks continuations with `##`: "playing" → `play`, `##ing`. |
| **Unigram / SentencePiece** (T5) | Top-down: start with a big vocab, **prune** tokens that hurt corpus likelihood least. SentencePiece treats space as a symbol (▁), so it is language-agnostic and reversible (crucial for Japanese/Chinese). |

## §3 Special tokens (the practicalities)

Added to the vocabulary as structural markers:

- `<PAD>`, pad shorter sequences to the batch's max length so tensors are rectangular. **Must be masked**, excluded from loss and from attention, or the model wastes capacity on padding and the loss is corrupted.
- `<SOS>`/`<BOS>`, start of sequence; kicks off a generator (the decoder's first input).
- `<EOS>`/`<EOT>`, end of sequence; lets the model **learn when to stop**. Generate until it emits `<EOS>`.
- `<UNK>`, fallback for unrepresentable tokens (obsolete with byte-level BPE).
- `<CLS>`/`<SEP>`, BERT: CLS's final vector summarizes the sequence; SEP separates two sentences.

A translation training pair (note the decoder input vs target are the same sequence shifted by one):

```
encoder input: <SOS> the cat sat <EOS> <PAD> <PAD>
decoder input: <SOS> le chat s' est assis
decoder target: le chat s' est assis <EOS>
```

The next step is turning these IDs into vectors, covered on the [embeddings](embeddings.md) page.
