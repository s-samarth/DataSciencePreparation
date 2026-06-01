# Evaluation Metrics

Every model above is only as trustworthy as the number you judge it by, and the wrong metric hides exactly the failure you care about. This section is the cross-cutting reference: classification, regression, ranking, generative, vision, RL, clustering, and drift, each metric explained as intuition, formula, when to use it, and when it lies.

!!! tip "Rapid Recall"
    The first question for any metric is what error is expensive: a false positive or a false negative. Accuracy lies under imbalance; use F1 or MCC, and AUC-PR rather than AUC-ROC when positives are rare and you do not care about true negatives. AUC is the probability a random positive outranks a random negative. For regression, RMSE targets the mean and MAE the median, while R² is the scale-free quality-versus-baseline number. Ranking uses NDCG for graded relevance and MAP for binary. Perplexity is exponentiated cross-entropy, and SHAP is the provably fair per-feature decomposition that sums exactly to prediction minus baseline.

## How to use this section

Each page takes a family of metrics and gives you both the breadth (every metric, its formula, when it lies) and the depth for the few you will be asked to defend hardest. The interactive widgets from the original study notes are captured as static figures next to the derivation they illustrate.

- **[Classification & Threshold](classification.md)**: accuracy through MCC, log loss, and the threshold-sweep metrics AUC-ROC and AUC-PR, with the confusion matrix as the foundation.
- **[Regression Metrics](regression.md)**: MSE, RMSE, MAE, MAPE, R², adjusted R², and Huber, with the R² decomposition.
- **[Ranking & Recommendation](ranking.md)**: NDCG, the precision@k to AP to MAP ladder, MRR, and online business metrics.
- **[LLM & Generative AI](llm-generative.md)**: perplexity in depth, plus BLEU, ROUGE, BERTScore, RAGAS, and human evaluation.
- **[Vision & RL Metrics](vision-rl.md)**: IoU, mAP, Dice, FID, SSIM, PSNR, top-k; and RL return, success rate, and offline estimators.
- **[Clustering, Drift & Explainability](clustering-drift.md)**: silhouette, Davies-Bouldin, ARI; PSI, KS, KL drift; and SHAP.

## Which metric for which task

| Task | Default Metric | When Imbalanced | When Stakeholders Need Simplicity |
|------|---------------|-----------------|-----------------------------------|
| Binary Classification | F1 or AUC-ROC | AUC-PR or MCC | Accuracy (if balanced) |
| Multi-class Classification | Macro-F1 | Weighted-F1 | Accuracy |
| Regression | RMSE | RMSE (with Huber for training) | MAPE (percentage) |
| Ranking / RecSys | NDCG@K | NDCG@K | MRR |
| Object Detection | mAP (COCO) | - | IoU |
| Segmentation | Dice / IoU | Dice | Pixel Accuracy |
| Image Generation | FID | - | SSIM |
| Summarization | ROUGE-2 | - | BERTScore |
| Translation | BLEU | - | BERTScore |
| RAG Systems | Faithfulness + Context Precision | - | Hallucination Rate |
| LLM Comparison | Elo (Arena) | - | LLM-as-judge |
| Clustering | Silhouette Score | - | - |
| Drift Detection | PSI + KS Test | - | PSI alone |
| Explainability | SHAP | - | Feature importance bar chart |

## Interview power moves

1. **"95% accuracy" trap:** Always ask about class distribution first.
2. **"Which metric?":** Start with "What's more expensive, a false positive or false negative?" This determines precision vs recall emphasis.
3. **AUC-ROC vs AUC-PR:** "I use AUC-PR when the positive class is rare because AUC-ROC is inflated by true negatives."
4. **RMSE vs MAE:** "It depends on whether I want my model to target the mean or the median of the distribution."
5. **BLEU/ROUGE vs BERTScore:** "Lexical overlap metrics miss paraphrases. BERTScore captures semantic similarity but is slower."
6. **Offline vs Online metrics:** "I optimize NDCG offline but validate with A/B tests on CTR/conversion because the offline-online gap is real."
7. **Drift detection:** "I monitor PSI on key features weekly. PSI > 0.2 triggers investigation and potential retraining."
8. **Explainability:** "SHAP values decompose each prediction into per-feature contributions that sum to the prediction. This satisfies both debugging and compliance needs."

## The reflexes (rapid fire)

- **"95% accuracy, good?"** → "Depends on class distribution."
- **"Which metric?"** → "What's more expensive, a false positive or a false negative?"
- **ROC vs PR** → "PR when positives are rare and I don't care about true negatives."
- **AUC meaning** → "Probability a random positive outranks a random negative, a Mann-Whitney U."
- **RMSE vs MAE** → "MSE targets the mean, MAE the median of the target distribution."
- **R² gotcha** → "Never decreases when adding features; use adjusted R² to compare complexity."
- **NDCG vs MAP** → "Graded relevance → NDCG; binary → MAP."
- **Perplexity caveat** → "Only comparable under identical tokenizer and test set."
- **SHAP key property** → "Efficiency: contributions sum exactly to prediction minus baseline."
