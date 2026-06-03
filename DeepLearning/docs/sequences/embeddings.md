# Embeddings

A token ID is an arbitrary label you cannot do math on. An embedding maps each ID to a dense vector where geometric closeness encodes semantic similarity, meaning becomes geometry. This is the layer that turns discrete symbols into something a network can reason over.

!!! tip "Rapid Recall"
    An embedding is a $[\text{vocab} \times d]$ lookup table, ID $i$ → row $i$, learned by gradient descent, where distance encodes meaning ($\text{king} - \text{man} + \text{woman} \approx \text{queen}$). Word2Vec learns embeddings as a byproduct of context prediction (CBOW or skip-gram) made trainable by negative sampling; GloVe factorizes global log co-occurrence counts. The fundamental limit is that these are *static* (one vector per word forever), which contextual embeddings (RNN/LSTM, ELMo, BERT) fix. Sentence embeddings use SBERT (mean-pooled, contrastively fine-tuned) and are compared by cosine similarity for search and RAG.

## §1 Intuition

A token ID like 4291 is an arbitrary label, you can't do math on it. An **embedding** maps each ID to a dense vector (for example 300 or 768 dims) where **geometric closeness encodes semantic similarity**. "king" and "queen" land near each other; "king" and "banana" far apart. Mechanically it's a **lookup table**: a $[\text{vocab} \times d]$ matrix, ID $i$ → row $i$, rows learned by gradient descent. Meaning becomes geometry:

$$
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})
$$

The "royalty" and "gender" relationships become actual directions in the vector space.

## §2 Word2Vec

Word2Vec rests on the **distributional hypothesis**: "a word is known by the company it keeps." Train a model to predict context; the byproduct, the weights, become the embeddings. No hand-labeling; meaning falls out of the prediction task. Two mirror architectures:

- **CBOW**, given surrounding context, predict the center word. Faster, better for frequent words.
- **Skip-gram**, given the center word, predict the surrounding context. Better for rare words / small data. The famous one.

**Negative sampling**, the trick that made it trainable: a full softmax over a million-word vocabulary every step is astronomical. Reframe "predict the right context word out of a million" as a tiny binary question: "is this (center, context) pair real or fake?" For each real pair, sample $k$ random **negative** words and train real high, fakes low:

$$
\log\sigma(v_{c}\cdot v_{w}) + \sum_{i=1}^{k}\mathbb{E}_{n_i\sim P(w)}\big[\log\sigma(-v_{c}\cdot v_{n_i})\big]
$$

where $\sigma$ = sigmoid and $v_c \cdot v_w$ = similarity. That is $k{+}1$ cheap binary checks instead of a million-way softmax. (Negatives are sampled with frequency$^{0.75}$, which dampens "the".)

## §3 GloVe, the count-based alternative

Word2Vec is predictive (local windows). GloVe is global and count-based: build a co-occurrence matrix $X_{ij}$ (how often word $j$ appears near $i$ across the whole corpus), then learn embeddings whose dot products match the *log* counts:

$$
v_i \cdot v_j + b_i + b_j \approx \log X_{ij}
$$

Similar quality to Word2Vec; Word2Vec streams over windows, GloVe factorizes global statistics.

## §4 The fundamental limitation: static embeddings

Word2Vec/GloVe are **static**, each word gets exactly one vector forever. But "bank" in "river bank" vs "savings bank" should differ. The fix is **contextual embeddings**: the vector is computed on the fly from the whole sentence (RNN/LSTM hidden states, then ELMo, then BERT). This is the bridge into the [recurrent models](rnns-and-gates.md) and beyond.

## §5 Sentence embeddings "these days"

- **Naive baseline**, average the word embeddings. Loses order, surprisingly OK.
- **Modern standard, Sentence-BERT (SBERT)**, take a Transformer encoder, **mean-pool** its contextual token outputs, and **fine-tune with a contrastive objective** (siamese setup: push paraphrases together, non-paraphrases apart). A raw BERT `<CLS>` token is a poor sentence vector without this.
- These power semantic search and RAG: embed documents + query, retrieve by **cosine similarity** $\frac{a\cdot b}{\|a\|\|b\|}$ (angle, magnitude-invariant). The contrastive objectives behind SBERT are the [InfoNCE / triplet](../losses/ranking-losses.md) losses.
