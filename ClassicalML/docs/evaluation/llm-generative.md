# LLM & Generative AI Metrics

Traditional NLP metrics were designed for tasks with clear reference answers, but LLMs generate open-ended text where there is no single correct answer. This pushed the field toward semantic metrics, framework-based evaluation, and LLM-as-judge. This page covers perplexity in depth, then BLEU, ROUGE, BERTScore, RAGAS, and human evaluation.

!!! tip "Rapid Recall"
    Perplexity is exponentiated cross-entropy, the effective branching factor of the next-token distribution, and structurally it is the inverse geometric mean of per-token probabilities, so one confidently-wrong token blows it up. It is only comparable under an identical tokenizer and test set, and it scores teacher-forced fit, not generation, so it says nothing about factuality. BLEU is precision-focused for translation, ROUGE is recall-focused for summarization, both purely lexical, while BERTScore matches meaning via embeddings. RAGAS splits RAG evaluation into faithfulness and retrieval quality, and frontier models serve as scalable LLM-as-judge graders.

## §1 Perplexity as effective branching

**Intuition:** "How surprised is the model by the text it sees?" A language model assigns a probability to each token; if it predicts the actual next token with high probability, it is not very perplexed. Beyond "exponentiated average negative log-likelihood," perplexity is a quantity living in the space of probability distributions, measured in *effective vocabulary size*.

$$\text{Perplexity} = \exp\!\Big(-\frac{1}{N}\sum_i \log P(w_i\mid w_1,\dots,w_{i-1})\Big)$$

!!! note "The one-sentence reframe"
    Perplexity is the *effective branching factor* of the model's predictive distribution. Perplexity 20 means that, averaged over every position, the model is as uncertain as if choosing uniformly among 20 equally-likely next tokens. The word "effective" is the whole concept: not literally 20 choices, but the uniform distribution that would leave you *equally* confused.

**Building it from entropy.** Entropy is average surprise; for a uniform distribution over \(k\) outcomes, \(H = \log k\). Invert it: a distribution with entropy \(H\) (in bits) is as uncertain as a uniform distribution over \(k = 2^{H}\). This \(k\) is the perplexity. Entropy lives in log-space (additive, abstract); perplexity in count-space (multiplicative, intuitive). Same information, two coordinate systems.

**Subtlety 1, it's a cross-entropy, not an entropy.** You don't know the true distribution \(p\); you have samples (the test text) and your model \(q\). What you compute is the cross-entropy:

$$H(p,q) = -\sum_x p(x)\log q(x) \approx -\frac{1}{N}\sum_{i=1}^{N}\log q(w_i\mid w_{<i})$$

!!! note "The floor you can never beat"
    By Gibbs' inequality, \(H(p,q)\ge H(p)\), equality iff \(q=p\). Cross-entropy is bounded below by the true entropy of language, so perplexity has a floor of \(2^{H(\text{language})}\). The excess decomposes exactly as KL divergence: \(H(p,q) - H(p) = D_{\text{KL}}(p\,\|\,q)\). So perplexity measures an *irreducible floor set by language itself* plus your model's *avoidable KL divergence from the truth*. Minimizing perplexity drives \(q\to p\).

**Subtlety 2, the geometric-mean structure.** By the chain rule \(q(w_1,\dots,w_N) = \prod_i q(w_i\mid w_{<i})\). Perplexity is the inverse geometric mean of per-token probabilities:

$$\text{PPL} = \left(\prod_{i=1}^{N}\frac{1}{q(w_i\mid w_{<i})}\right)^{1/N} = \left(\prod_{i=1}^{N} q(w_i\mid w_{<i})\right)^{-1/N}$$

Each token contributes \(1/q\), its local branching factor. Geometric, not arithmetic, because probabilities multiply along the sequence, so the natural average is additive in log-space.

!!! warning "The brutal asymmetry"
    Geometric mean is dominated by its smallest terms. One token assigned probability 0.0001 contributes local perplexity 10,000 and drags the whole geometric mean up violently. *One confidently-wrong token poisons the perplexity of an entire document.* This is why perplexity is so sensitive to tail behavior.

**The four caveats that separate a real answer from a recited one:**

- **Only comparable under identical tokenization.** Perplexity is per-token normalized. A 256K-vocab model chops text into fewer tokens than a 32K-vocab one, spreading the joint probability over fewer factors, mechanically lowering per-token perplexity without being "better." Comparable only with identical test set *and* tokenizer. (Bits-per-byte normalizes by raw bytes to escape this.)
- **Measures distributional fit, not generation.** Computed teacher-forced (every position conditions on the true history). It never watches the model generate, so a model can have great perplexity yet produce repetitive or degenerate text under sampling.
- **Blind to what you care about in a chat model.** Factuality, helpfulness, safety, instruction-following, none captured. Confidently predicting the next token of a factually wrong test sentence gets *rewarded*. Central for pretraining; near-useless for an aligned assistant.
- **The KL-floor explains diminishing returns.** Since perplexity bottoms out at \(2^{H(\text{language})}\), early training crushes huge KL cheaply; late training fights for the last fraction of a bit, the information-theoretic shadow of the scaling-law plateau.

## §2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Intuition:** "How much of the reference summary appears in the generated summary?" ROUGE is recall-focused: it measures what fraction of the reference n-grams the model reproduced. Variants: ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), ROUGE-L (longest common subsequence, capturing sentence-level structure).

$$\text{ROUGE-N Recall} = \frac{\text{matching n-grams between generated and reference}}{\text{total n-grams in reference}}$$

**When to use:** summarization evaluation, the standard in summarization research; ROUGE-2 is the most discriminative in practice. **When it fails:** ROUGE is purely lexical. "The cat sat on the mat" and "A feline rested on the rug" get ROUGE-1 = 0 despite being semantically identical. This is why BERTScore was invented.

## §3 BERTScore

**Intuition:** Instead of matching exact words like ROUGE, BERTScore matches meaning. It passes both generated and reference texts through a pre-trained BERT, gets contextual embeddings for each token, and computes the best matching between tokens by cosine similarity. Precision: for each candidate token, find the most similar reference token and average; recall: the reverse; F1: their harmonic mean.

**Why it's better than ROUGE:** "Dog" and "canine" have cosine similarity about 0.85 in BERT space, so they get credit, while ROUGE gives zero. **Gotchas:** slower (requires BERT inference), depends on the pre-trained model used, and can be tricked by semantically similar but factually wrong text.

## §4 BLEU (Bilingual Evaluation Understudy)

**Intuition:** The oldest automatic metric, designed for machine translation. "How many of the n-grams in the generated text also appear in the reference?" BLEU is precision-focused, the opposite of ROUGE.

$$\text{BLEU} = \text{BP}\cdot\exp\!\Big(\sum_n w_n\log p_n\Big),\qquad \text{BP} = \min\!\Big(1,\ \exp\big(1 - \tfrac{\text{ref\_len}}{\text{gen\_len}}\big)\Big)$$

where \(p_n\) is the modified n-gram precision and \(w_n\) the weights (typically 1/4 for n = 1,2,3,4). The brevity penalty BP prevents cheating by generating very short text that could have high precision.

| | BLEU | ROUGE |
|---|------|-------|
| Focus | Precision (generated n-grams in reference) | Recall (reference n-grams in generated) |
| Designed for | Machine translation | Summarization |
| Brevity penalty | Yes | No |
| N-gram range | Typically 1-4 (combined) | Individual (ROUGE-1, ROUGE-2) |

**When to use:** machine translation benchmarking. **When not to use:** open-ended generation, creative writing, anything where the output need not closely match a reference.

## §5 RAG metrics (RAGAS framework)

RAG systems have two components that can fail independently: the retriever and the generator. RAGAS provides metrics for each.

- **Faithfulness:** "Did the model make stuff up, or only say things supported by the retrieved context?" Each claim in the answer is checked against the context: \(\text{Faithfulness} = \frac{\text{claims supported by context}}{\text{total claims in answer}}\). This is the hallucination detector.
- **Answer Relevancy:** "Does the answer actually address what the user asked?" Generate hypothetical questions from the answer, then measure semantic similarity to the original question.
- **Context Precision** = Relevant chunks retrieved / Total chunks retrieved. "Were the retrieved chunks actually useful?"
- **Context Recall** = Relevant chunks retrieved / Total relevant chunks in corpus. "Did the retriever find everything it needed?"
- **Hallucination Rate** = Ungrounded claims / Total claims, essentially 1 minus faithfulness.

These are typically computed using an LLM-as-judge (a frontier model evaluating the outputs). Not exact, but a scalable way to evaluate RAG pipelines.

## §6 Human and advanced evaluation

**LLM-as-Judge:** use a strong frontier model to evaluate a weaker model's outputs against a rubric (helpfulness, accuracy, coherence, safety). High correlation (0.8 to 0.9) with human judgments, much faster and cheaper. Failure modes: position bias (prefer the first response), verbosity bias (longer rated higher), self-preference (a model rates its own family higher), and rubric sensitivity.

**Elo Rating:** borrowed from chess. Two models answer the same prompt, a human or judge picks the better one, and over many matchups each accumulates an Elo score: \(R_{\text{new}} = R_{\text{old}} + K(\text{actual} - \text{expected})\) with \(\text{expected} = 1/(1 + 10^{(R_{\text{opp}} - R_{\text{self}})/400})\). Used by Chatbot Arena, the most influential LLM benchmark.

**Side-by-Side and Likert:** SBS shows two outputs and a human picks the better one (simple but needs many comparisons); Likert rates each output 1 to 5 on specific dimensions (more granular but introduces inter-rater variability).

## Interview questions

**Q1: Give the deep definition of perplexity.**
Perplexity is the exponential of the cross-entropy between the data and the model, which makes it the effective branching factor of the next-token distribution, the size of the uniform distribution that would leave you equally uncertain. Because cross-entropy splits into the entropy of language plus the KL divergence from model to truth, perplexity is an irreducible floor plus the model's avoidable divergence, so minimizing it drives the model distribution toward the true one. Structurally it is the inverse geometric mean of per-token probabilities.

**Q2: Why is perplexity dominated by the worst-predicted tokens?**
Because it is a geometric mean of per-token probabilities, and a geometric mean is dominated by its smallest terms. A single token assigned probability 0.0001 contributes a local branching factor of 10,000 and hauls the whole product up, so one confidently-wrong token poisons the perplexity of an entire document. This is why perplexity is so sensitive to tail behavior.

**Q3: When is perplexity comparable, and what doesn't it measure?**
It is only comparable across models evaluated on an identical test set and tokenizer, because it is per-token normalized and a larger vocabulary mechanically lowers it by splitting text into fewer tokens. It is computed teacher-forced, so it measures distributional fit, not generation quality, and it captures nothing about factuality, helpfulness, or safety. It is central for pretraining but is replaced by downstream and human evaluations for aligned models.

**Q4: Why did the field move from BLEU and ROUGE toward BERTScore and LLM-as-judge?**
Because BLEU and ROUGE are purely lexical, so two sentences that mean the same thing with different words score near zero, which fails badly on open-ended generation. BERTScore matches contextual embeddings by cosine similarity, so synonyms get credit, capturing semantic rather than surface similarity. For fully open-ended outputs, LLM-as-judge uses a frontier model against a rubric and correlates highly with humans at far lower cost, though it has position, verbosity, and self-preference biases.
