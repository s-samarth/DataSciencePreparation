# Sequences & Attention

A sequence model handles data where order matters and elements depend on each other. Text is the canonical case because of one reframing: language modeling = next-token prediction. This section walks the full arc from turning text into numbers up to the doorstep of the Transformer.

!!! tip "Rapid Recall"
    Language modeling is next-token prediction, and the probability of a sequence factorizes by the chain rule into per-token conditionals. Tokenization splits text into subword units (BPE merges frequent pairs, byte-level never hits OOV); embeddings map IDs to vectors where geometry encodes meaning. RNNs carry a hidden state but suffer vanishing/exploding gradients; LSTMs add an additive cell state (a gradient highway) for long context, GRUs are the lighter sibling. Models wire as decoder-only (causal, GPT), encoder-only (bidirectional, BERT), or encoder-decoder (seq2seq). Attention removes the fixed-vector bottleneck and seeds the query-key-value pattern that became self-attention, the Transformer.

## §1 Why sequence modeling, and why text

A sequence model handles data where **order matters and elements depend on each other**. Text is canonical because of one reframing: **language modeling = next-token prediction.** Given "The cat sat on the ___", predict the next token. The probability of a whole sequence factorizes by the chain rule:

$$
P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1})
$$

Every model in this arc, RNN, LSTM, Transformer, GPT, is a different machine for estimating that one conditional. The differences are *how* they compress "the context so far."

## §2 What is in this section

- [Tokenization and BPE](tokenization-bpe.md): splitting text into subword units and the greedy-merge algorithm behind every modern tokenizer.
- [Embeddings](embeddings.md): mapping token IDs to vectors where geometry encodes meaning (Word2Vec, GloVe, contextual, sentence embeddings).
- [RNNs and gated cells](rnns-and-gates.md): the recurrent hidden state, its vanishing-gradient flaw, and the LSTM/GRU fix.
- [Architectures and attention](architectures-attention.md): decoder-only / encoder-only / encoder-decoder wiring, the seq2seq bottleneck, and attention as the bridge to the Transformer.
