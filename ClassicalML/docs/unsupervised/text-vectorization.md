# Text Vectorization

Classical ML cannot read text, so you turn it into numbers. TF-IDF weights each word by how distinctive it is to a document, producing a wide sparse matrix, and Truncated SVD (LSA) reduces that matrix to dense latent topics without the centering that would destroy its sparsity. This page covers both and the pipeline that connects them.

!!! tip "Rapid Recall"
    TF-IDF scores a word high when it is frequent in a document but rare across the corpus, turning a text column into a wide sparse matrix of documents by vocabulary. The classic pairing is TF-IDF with a linear SVM, logistic regression, or multinomial Naive Bayes, using cosine similarity on L2-normalized vectors. Do not run vanilla PCA on TF-IDF: centering turns every zero nonzero and densifies the matrix. Use Truncated SVD instead, which is PCA without centering; applied to a document-term matrix it is LSA, and its top components are latent topics because they capture word co-occurrence.

## §1 TF-IDF

TF-IDF turns a word into a number answering: "**how important is this word to this specific document, relative to the whole corpus?**" It's a smarter word-count that automatically crushes filler words and boosts distinctive ones.

### The components

- **TF** (term frequency) — how often the word appears in *this* document.
- **IDF** (inverse document frequency) — how rare the word is *across all* documents. "the/is/and" everywhere → IDF about 0 → crushed. "thrombocytopenia" in 3 docs → high IDF → boosted.
- **TF-IDF = TF × IDF** — high only when a word is frequent *here* but rare *elsewhere*.

A common smoothed form:

$$\text{tf-idf}(t,d)=\underbrace{f_{t,d}}_{\text{count in }d}\;\times\;\underbrace{\log\!\frac{N}{1+n_t}}_{\text{rarity across corpus}}$$

\(N\) = total documents, \(n_t\) = documents containing term \(t\). The \(+1\) avoids division by zero.

### The key move: text to tabular data

TF-IDF vectorization converts a column of raw text into a numeric matrix classical ML can consume. **Rows = documents, columns = vocabulary words, values = TF-IDF.** Each document becomes a point in "word space."

|  | approval | cheap | deal | loan | meeting | noon | offer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| doc1 "cheap loan offer" | 0 | 0.52 | 0 | 0.52 | 0 | 0 | 0.67 |
| doc2 "meeting at noon" | 0 | 0 | 0 | 0 | 0.47 | 0.6 | 0 |
| doc3 "loan approval meeting" | 0.6 | 0 | 0 | 0.47 | 0.47 | 0 | 0 |
| doc4 "cheap cheap deal" | 0 | 0.85 | 0.52 | 0 | 0 | 0 | 0 |

!!! note "The defining property: huge and sparse"
    Real vocab = 20k to 100k+ unique words → that many **columns**. Any single document uses only a few hundred → most columns are zero. Stored as a **sparse matrix**. This sparsity plus dimensionality shapes every downstream choice. (Your "one feature is text" instinct was right but undersized, it's one text column exploding into thousands of word-features.)

### How it's used in classical ML

1. **Text classification** (bread-and-butter): spam, sentiment, topic, ticket routing. Pipeline: `raw text → TF-IDF → classifier`. Classic pairing: **TF-IDF + Linear SVM** or **Logistic Regression** (linear models thrive in high-D sparse space; data is usually linearly separable there). **Multinomial Naive Bayes** is the other strong, fast baseline. This combo was SOTA pre-deep-learning and is still a brutally strong baseline in 2026.
2. **Document clustering**: feed TF-IDF to **K-Means** (with cosine, not Euclidean) to group articles/complaints by topic.
3. **Information retrieval / search / similarity**: **cosine similarity** between TF-IDF vectors. The backbone of pre-neural search; the classical cousin of embeddings plus vector DBs (text → vector → similarity), just sparse-lexical instead of dense-semantic.

!!! warning "PCA on TF-IDF, the gotcha"
    Don't use vanilla PCA. PCA **centers** the data (subtracts the column mean), which turns every zero into a nonzero, destroying sparsity and exploding memory. Use **Truncated SVD** (= LSA), which does linear reduction *without* centering and preserves sparsity. Corrected pipeline:

    `text → TF-IDF (50k sparse) → Truncated SVD/LSA (→ 100-300 dense) → classifier / clustering / t-SNE / UMAP`

### Normalize and use cosine

TF-IDF vectors are usually **L2-normalized** (unit length) to remove document-length effects, a long doc shouldn't be "more" of a topic just for being long. Once unit-length, **dot product = cosine similarity**, which measures *direction* (which words, in what proportion) not *magnitude*. For text, direction is what matters, hence cosine, not Euclidean, and configure K-Means accordingly.

### Where it stands in 2026

| TF-IDF | Embeddings (BERT, sentence-transformers) |
| --- | --- |
| Sparse, lexical, fast, interpretable, zero training cost. "Dumb": "loan" and "mortgage" are different columns with zero overlap, pure surface word-matching. | Dense, semantic, understands synonyms/meaning, but needs a model, compute, less interpretable. |

Reach for TF-IDF as a fast, strong, explainable baseline; for keyword-driven tasks (spam, legal/medical where exact terms matter); when compute is limited; when you need interpretability ("which words drove this?"). Honest answer: *start with TF-IDF + linear model; move to embeddings if you need semantic understanding and can afford the compute.*

## §2 Truncated SVD and LSA

Truncated SVD is "keep the top-\(k\) singular vectors from SVD." The interesting part is *why* that's the same thing as PCA, *when* it isn't, and *why* doing it to a TF-IDF matrix produces "topics."

### From SVD

Any matrix \(X\) (\(n\) docs × \(d\) words) factors as:

$$X=U\Sigma V^{\top}$$

- **\(V\)** (right singular vectors) — directions in *feature space*, i.e. combinations of words; axes of maximum variation.
- **\(\Sigma\)** (singular values) — how much energy/variance each direction carries, sorted descending.
- **\(U\)** — the documents expressed in those directions.

Truncated SVD keeps the top \(k\):

$$X\approx U_k\Sigma_k V_k^{\top},\qquad\text{reduced data}=U_k\Sigma_k$$

The top singular vectors capture dominant co-variation; the long tail of tiny singular values is mostly noise. By the **Eckart-Young theorem**, truncated SVD is the *best possible rank-\(k\) approximation* of \(X\) (smallest reconstruction error).

!!! note "PCA is SVD on centered data"
    They're the same algorithm with one preprocessing difference:

    - **PCA**: subtract the column mean (center), *then* SVD. Finds directions of maximum **variance** (defined around the mean, so centering is mandatory).
    - **Truncated SVD**: SVD on raw, **uncentered** \(X\). Finds directions of maximum **energy** (around the origin).

    The *only* difference: **does the data get centered first?** Center → PCA. Don't → truncated SVD. On already-centered data they're identical.

### Why it matters for text, the sparsity collision

Centering means subtracting each column's (small positive) mean from every entry, so every zero becomes \(0-\text{mean}\neq 0\) → the sparse matrix densifies into a full 50k-column matrix that won't fit in RAM. PCA *requires* centering → unusable on large sparse text. Truncated SVD *skips* centering → operates directly on the sparse matrix. **That's why it exists as a distinct tool: it's the dimensionality reducer that survives sparse data.**

### Why "LSA / Latent Semantic Analysis"?

LSA = truncated SVD applied specifically to a document-term matrix. Each top right-singular vector \(V_k\) is a direction in word-space, a weighted combination of words that tend to **co-occur** across documents:

- Component 1 might weight {loan, mortgage, credit, interest, approval} → a "finance" direction.
- Component 2 might weight {patient, dose, symptom, diagnosis} → a "medical" direction.

Nobody labeled these, SVD's top directions capture co-variation, and in a doc-term matrix that *is* word co-occurrence. These are the **latent semantic dimensions**, hidden meaning-axes inferred from co-occurrence. Hence *Latent* (hidden), *Semantic* (meaning), *Analysis*.

!!! note "LSA partially solves synonymy"
    Plain TF-IDF: "car" and "automobile" are different columns with zero overlap, so two docs using one each look unrelated (cosine about 0). But both co-occur with {road, drive, engine, wheel}, so LSA places them along the *same* latent direction, the docs land near each other despite sharing no literal words. The crude, linear ancestor of dense embeddings (LSA → word2vec → modern embeddings).

### Truncated SVD / LSA vs PCA

**Advantages over PCA:**

- Handles **sparse** data without densifying, the decisive win on text.
- Scales better (randomized SVD computes only top-\(k\)).
- No centering step (the step that breaks on sparse data).

**Disadvantages vs PCA:**

- Components aren't "principal components" in the variance sense, directions maximize energy around the origin; on uncentered data the leading component often just points toward the overall mean/magnitude (a "size" direction).
- "Explained variance" framing is exact for PCA, only approximate here.
- For dense, modest-dimensional data, PCA is the more principled choice.

!!! note "One-liner"
    Truncated SVD is PCA without the centering step, which is precisely what lets it work on huge sparse matrices like TF-IDF; applied to a document-term matrix it's called LSA, and its top components are interpretable as latent topics because they capture word co-occurrence.

## Interview questions

**Q1: What does TF-IDF compute, and what does the resulting matrix look like?**
It scores each word in each document as term frequency times inverse document frequency, so a word scores high only when it is frequent in that document but rare across the corpus, which crushes filler words and boosts distinctive ones. The result is a matrix of documents by vocabulary, which is huge (tens of thousands of word columns) and sparse, since any document uses only a few hundred words. This sparsity and dimensionality drive every downstream choice.

**Q2: Why must you not run vanilla PCA on a TF-IDF matrix?**
Because PCA centers the data by subtracting each column's mean, and since most TF-IDF entries are zero, subtracting a small positive mean turns every zero nonzero, densifying a 50k-column sparse matrix into one that will not fit in memory. Truncated SVD does the same linear reduction without centering, so it operates directly on the sparse matrix, which is exactly why it exists as a separate tool.

**Q3: How are Truncated SVD, PCA, and LSA related?**
PCA is SVD on centered data and finds directions of maximum variance around the mean; Truncated SVD is SVD on uncentered data and finds directions of maximum energy around the origin, so the only difference is whether you center first. LSA is Truncated SVD applied to a document-term matrix specifically, where the top right singular vectors are latent topics because, in that matrix, captured co-variation is word co-occurrence.

**Q4: Why does cosine similarity, not Euclidean, suit TF-IDF, and what does LSA add?**
TF-IDF vectors are L2-normalized to remove document-length effects, and once unit-length the dot product equals cosine similarity, which measures direction, which words in what proportion, rather than magnitude, which is what matters for text. Plain TF-IDF is purely lexical, so synonyms like car and automobile share no column, whereas LSA places them on the same latent direction because they co-occur with the same context words, partially solving synonymy.
