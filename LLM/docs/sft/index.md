# Post-Training: SFT

Supervised fine-tuning is the bridge between a giant autocomplete engine (the pretrained base) and a model that follows instructions. It is mechanically simple, next-token cross-entropy on labeled data, but every production system gets at least one detail wrong somewhere. The pages in this section cover the four details that matter.

!!! tip "Rapid Recall"
    SFT is the same next-token cross-entropy as pretraining, but with three structural changes that you must internalize. (1) The data is instruction-response pairs wrapped in the model's chat template. (2) Loss is masked on user/system tokens (the `-100` ignore_index trick). (3) The learning rate is 10 to 100× smaller than pretraining because the model already sits at a good optimum and you only want to nudge it. The deeper why is statistics: every loss in this section, and most losses in machine learning, is maximum likelihood under some distributional assumption (Gaussian → MSE, Bernoulli → BCE, categorical → cross-entropy, Bradley-Terry → reward-model loss).

## Pages in this section

- **[Three-stage pipeline](three-stage-pipeline.md)** — pretrain → SFT → alignment, what question each stage answers, and the 2026 plot twist of RLVR added as a fourth stage.
- **[Chat templates](chat-templates.md)** — why hand-rolled "USER:" / "ASSISTANT:" tags lose to atomic special tokens, and how `tokenizer.apply_chat_template()` is the single source of truth.
- **[Loss masking](loss-masking.md)** — the single most important SFT concept, with the off-by-one bug everyone hits and how to verify masking visually.
- **[MLE and MAP backbone](mle-map-backbone.md)** — the statistics behind every loss in this whole site, including the L2 = Gaussian-prior insight.

For the runnable end-to-end version (Qwen full SFT plus TinyLlama LoRA on Dolly-15K), see [SFT Walkthrough](../build-from-scratch/sft-walkthrough.md).
