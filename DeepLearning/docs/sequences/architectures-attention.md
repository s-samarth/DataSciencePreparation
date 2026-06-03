# Architectures & Attention

Three ways to wire a sequence model serve three task families, and one idea, attention, removes the bottleneck that crippled the original seq2seq model and seeds the entire Transformer. This page is the bridge from recurrence to self-attention.

!!! tip "Rapid Recall"
    Three wirings: decoder-only (causal, left-to-right, GPT, trained by next-token prediction), encoder-only (bidirectional, BERT, trained by masked language modeling), and encoder-decoder (seq2seq, teacher forcing, exposure bias). Classic LSTM seq2seq crammed the whole source into one fixed vector, which collapses on long sentences. Attention fixes this: keep all encoder states and, at each decode step, score them against the decoder state, softmax into weights, and take the weighted sum as a focused context vector. This is the query-key-value pattern, and letting a sequence attend to itself in parallel is the Transformer.

## §1 Encoder-only, decoder-only, encoder-decoder

Three ways to wire a sequence model for three task families. The distinction predates Transformers (it applies to RNNs/LSTMs) and carries straight into them.

| Structure | How it reads | Task fit & training |
| --- | --- | --- |
| **Decoder-only** (autoregressive) | Left-to-right, **causal**, token $t$ sees only $1..t-1$ (can't peek at the future it must predict). | Generation (LM, completion, dialogue; GPT = Transformer version, LSTM-LM = recurrent). Trained by **next-token prediction**; input = sequence shifted right (start `<SOS>`), target = shifted left (end `<EOS>`). **Self-supervised**, labels are the text itself, so it scales to the whole internet. |
| **Encoder-only** (representation) | Whole sequence at once, **bidirectionally**, every token sees both sides. | Understanding (classification, sentiment, NER, embeddings; BERT). Trained by **masked language modeling**, hide ~15% of tokens, predict them from both-side context. Bidirectionality is allowed only because it's not generating. `<CLS>` output = sentence representation. |
| **Encoder-decoder** (seq2seq) | Encoder reads + compresses input; decoder generates output one token at a time, conditioned on the encoder. | Transforming one sequence to another (translation, summarization, speech-to-text). **Teacher forcing**: feed the decoder the true previous target token during training (stabilizes). At inference it consumes its own outputs, so the train/inference mismatch is **exposure bias**. |

!!! note "Classic Seq2Seq (LSTM encoder-decoder) and its fatal bottleneck"
    The 2014 model: an LSTM encoder reads the whole source and dumps its **final hidden state**, a single fixed-size vector, as "the meaning." The decoder starts from that vector and generates.

    ```
    Encoder: the → cat → sat → on → mat → [c] ──┐ (c = ONE fixed vector)
    Decoder: [c] → le → chat → s'est → assis → <EOS>
    ```

    The *entire* source, 5 words or 50, must be crammed into one fixed vector. For long sentences this is hopeless; early words are forgotten by the time the encoder finishes. Translation quality collapsed with length. This bottleneck is exactly what attention was invented to fix.

## §2 Attention (the build-up, stopping before the Transformer)

The problem is forcing all source information through one vector. The fix is obvious in hindsight: **don't compress; keep all the encoder's hidden states (one per source word) and let the decoder look back at the relevant ones each time it generates a word.** When producing the French word for "cat," focus on the encoder state for "cat" and ignore the rest. Attention computes a **weighted average of all encoder hidden states**, the weights being "how relevant is each source word right now," recomputed per step. No bottleneck, direct focused access to the entire source.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg">
  <style>
    .es{fill:rgba(94,168,196,.18);stroke:#5ea8c4;stroke-width:1.6}
    .et{fill:#cdd6e3;font-family:'Spline Sans Mono',monospace;font-size:12px;text-anchor:middle}
    .wt{font-family:'Spline Sans Mono',monospace;font-size:12px;text-anchor:middle}
    .cv{fill:rgba(201,122,90,.18);stroke:#c97a5a;stroke-width:1.8}
    .lb{fill:#8a8778;font-family:'Spline Sans Mono',monospace;font-size:11px;text-anchor:middle}
  </style>
<rect x="40" y="40" width="70" height="40" class="es"/><text x="75" y="65" class="et">h1</text><text x="75" y="32" class="lb">the</text><path d="M 75 80 L 330 165" stroke="#e0b341" stroke-width="1.5" fill="none" opacity="0.26"/><text x="75" y="98" class="wt" fill="#e0b341">0.1</text><rect x="130" y="40" width="70" height="40" class="es"/><text x="165" y="65" class="et">h2</text><text x="165" y="32" class="lb">cat</text><path d="M 165 80 L 330 165" stroke="#e0b341" stroke-width="5.7" fill="none" opacity="0.92"/><text x="165" y="98" class="wt" fill="#e0b341">0.7</text><rect x="220" y="40" width="70" height="40" class="es"/><text x="255" y="65" class="et">h3</text><text x="255" y="32" class="lb">sat</text><path d="M 255 80 L 330 165" stroke="#e0b341" stroke-width="1.5" fill="none" opacity="0.26"/><text x="255" y="98" class="wt" fill="#e0b341">0.1</text><rect x="310" y="40" width="70" height="40" class="es"/><text x="345" y="65" class="et">h4</text><text x="345" y="32" class="lb">on</text><path d="M 345 80 L 330 165" stroke="#e0b341" stroke-width="1.2" fill="none" opacity="0.21"/><text x="345" y="98" class="wt" fill="#e0b341">0.05</text><rect x="400" y="40" width="70" height="40" class="es"/><text x="435" y="65" class="et">h5</text><text x="435" y="32" class="lb">mat</text><path d="M 435 80 L 330 165" stroke="#e0b341" stroke-width="1.2" fill="none" opacity="0.21"/><text x="435" y="98" class="wt" fill="#e0b341">0.05</text>
  <circle cx="330" cy="180" r="26" class="cv"/><text x="330" y="184" class="et" style="fill:#c97a5a">c_t</text>
  <text x="330" y="225" class="lb">context vector = Σ α·h  (different every decode step)</text>
  <text x="330" y="120" class="lb" style="fill:#e0b341">attention weights α (softmax, sum to 1) — mostly "cat"</text>
</svg>
<figcaption>At each decode step the decoder scores every encoder state, softmaxes the scores into weights α, and takes their weighted sum as a focused context vector.</figcaption>
</figure>

The math (additive / Bahdanau attention). At decoder step $t$, with decoder state $s_{t-1}$ and encoder states $h_1 \dots h_n$:

**1. Score** each encoder state against the current decoder state (alignment):

$$
e_{t,i} = \text{score}(s_{t-1}, h_i) = v^\top \tanh(W_s s_{t-1} + W_h h_i)
$$

(Bahdanau = additive; Luong's simpler dot product $s_{t-1}^\top h_i$ = multiplicative, the seed of Transformer attention.)

**2. Normalize** into weights summing to 1 (softmax over source positions):

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n}\exp(e_{t,j})}
$$

**3. Context vector** = weighted sum of encoder states:

$$
c_t = \sum_{i=1}^{n} \alpha_{t,i}\, h_i
$$

**4. Generate** the next token from the decoder state combined with $c_t$. Since $\alpha$ is recomputed every step, the decoder attends to different source words as it proceeds.

## §3 Why this is the bridge to Transformers

Three things attention generalizes:

1. The bottleneck is gone, the decoder reaches the entire source directly, with learned focus.
2. The weights $\alpha_{t,i}$ are interpretable, plot them and see a soft word-alignment matrix.
3. The **query-key-value** pattern is implicit: the decoder state is a **query**, encoder states are **keys** (to match) and **values** (to average).

The Transformer's leap: if attention is this powerful, drop the RNN entirely, let a sequence attend to **itself** (self-attention), and process all positions **in parallel** (no sequential bottleneck), which is what made massive scaling possible. That's the Transformer, where we stop.

The build-up is complete: fixed-vector bottleneck → attention as a dynamic weighted lookup over all encoder states → the realization that attention alone, applied to a sequence attending to itself, could replace recurrence entirely. The loss that trains these generative models is [next-token cross-entropy](../losses/llm-losses.md).
