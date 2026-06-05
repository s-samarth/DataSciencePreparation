# ML Evaluation Metrics: The Complete Interview-Ready Reference

> **How to use this doc:** Each metric gets: intuition (why it exists), the formula, when to use it, when it lies to you, and how it compares to alternatives. Written so you can explain any metric in 60 seconds under interview pressure.

---

## 1. Classification Metrics

### The Big Picture

Classification is about predicting categories. But "did my model get it right?" has many flavors depending on what "right" means to your business. A cancer screening model that catches 99% of cancers but flags 50% of healthy people as sick is great at recall but terrible at precision. Every classification metric is a different lens on the same confusion matrix.

**The Confusion Matrix (Foundation for Everything)**

```
                    Predicted
                 Positive  Negative
Actual Positive [   TP   |   FN   ]
Actual Negative [   FP   |   TN   ]
```

- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Incorrectly predicted positive (Type I error)
- **FN (False Negative):** Incorrectly predicted negative (Type II error)

Every classification metric below is just a different ratio of these four numbers.

---

### 1.1 Accuracy

**Intuition:** The most naive question: "What fraction of predictions did I get right?" It treats every correct prediction equally, whether it's a positive or negative.

**Formula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:** Only when classes are roughly balanced AND the cost of false positives ≈ cost of false negatives. Example: predicting if a coin flip is heads or tails.

**When it lies to you:** Imbalanced datasets destroy accuracy's meaning. If 99% of emails are not spam, a model that predicts "not spam" for everything gets 99% accuracy but catches zero spam. This is the single most important gotcha in all of ML metrics.

**Interview insight:** If an interviewer asks "your model has 95% accuracy, is it good?" the correct answer is always "it depends on the class distribution." This is a trap question.

---

### 1.2 Precision

**Intuition:** Of all the things I flagged as positive, how many were actually positive? Precision answers: "When I raise an alarm, can you trust me?"

**Formula:**

```
Precision = TP / (TP + FP)
```

**When to use:** When false positives are expensive. Examples:
- Spam filter: Marking a legitimate email as spam (FP) means your user misses an important email. You'd rather let some spam through than eat real emails.
- Recommender systems: Recommending irrelevant products erodes trust.

**Relationship to other metrics:** Precision is in tension with recall. You can trivially get perfect precision by only predicting positive when you're 100% sure, but then you'll miss a ton of actual positives (low recall). This tradeoff is the heart of the precision-recall curve.

---

### 1.3 Recall (Sensitivity / True Positive Rate)

**Intuition:** Of all the actual positives in the data, how many did I catch? Recall answers: "Am I missing things that matter?"

**Formula:**

```
Recall = TP / (TP + FN)
```

**When to use:** When false negatives are expensive. Examples:
- Cancer screening: Missing a cancer diagnosis (FN) is catastrophic. You'd rather have extra false positives (unnecessary biopsies) than miss a real case.
- Fraud detection: Missing a fraudulent transaction is worse than flagging some legitimate ones.

**Precision vs Recall (the core tradeoff):**
- **High precision, low recall:** The model is conservative. It only flags things it's very sure about, so it misses a lot.
- **High recall, low precision:** The model is aggressive. It catches almost everything but also flags a lot of noise.
- You can always trade one for the other by adjusting the classification threshold.

---

### 1.4 F1 Score

**Intuition:** You want a single number that balances precision and recall. A simple average would let one dominate (precision=1.0, recall=0.01, average=0.505 looks fine but is terrible). The harmonic mean punishes imbalance, so the F1 score is only high when BOTH precision and recall are high.

**Formula:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Equivalently:

```
F1 = 2*TP / (2*TP + FP + FN)
```

**Why harmonic mean, not arithmetic mean?**
- Arithmetic mean of (1.0, 0.01) = 0.505 (misleadingly high)
- Harmonic mean of (1.0, 0.01) = 0.0198 (correctly reflects the disaster)

The harmonic mean is always ≤ arithmetic mean, and equals it only when both values are identical. It gravitates toward the smaller value.

**When to use:** When you want a balanced view and classes are imbalanced. It's the default "go-to" metric for imbalanced classification when you don't have a clear business preference between precision and recall.

**Generalization (F-beta score):**

```
F_β = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```

- β = 1: Standard F1, equal weight
- β = 2: F2 score, weights recall 2x more than precision (use when missing positives is costly)
- β = 0.5: F0.5 score, weights precision 2x more than recall (use when false alarms are costly)

---

### 1.5 AUC-ROC

**Intuition:** Most classifiers output a probability (e.g., "70% chance of spam"). The ROC curve answers: "If I sweep my classification threshold from 0 to 1, how does my true positive rate vs false positive rate change?" The AUC is the area under this curve, a single number summarizing performance across ALL possible thresholds.

**The ROC Curve:**
- X-axis: False Positive Rate = FP / (FP + TN) = 1 - Specificity
- Y-axis: True Positive Rate = TP / (TP + FN) = Recall

Imagine sweeping a threshold from 1.0 (predict nothing positive) to 0.0 (predict everything positive). At threshold=1.0, you're at origin (0,0). At threshold=0.0, you're at (1,1). A good model climbs steeply toward (0,1), the top-left corner.

**Formula:**

```
AUC = ∫₀¹ TPR(FPR) d(FPR)
```

**Probabilistic interpretation:** AUC = the probability that a randomly chosen positive example is ranked higher (gets a higher predicted score) than a randomly chosen negative example. This is the cleanest way to explain it in an interview.

**Key values:**
- AUC = 1.0: Perfect model
- AUC = 0.5: Random guessing (the diagonal line)
- AUC < 0.5: Worse than random (your labels might be flipped)

**When to use:** When you want a threshold-independent evaluation. Great for comparing models before you decide on a threshold. Very common in Kaggle competitions and research papers.

**When it lies to you:** When classes are heavily imbalanced. With 99.9% negatives, even a small FPR like 0.01 means a huge number of false positives in absolute terms. The ROC curve doesn't show this because FPR is a rate, not a count. In such cases, use AUC-PR instead.

---

### 1.6 AUC-PR (Area Under the Precision-Recall Curve)

**Intuition:** Same idea as AUC-ROC, but plots Precision (y-axis) vs Recall (x-axis) as you sweep the threshold. Unlike ROC, the PR curve focuses entirely on the positive class, so it doesn't get inflated by a large number of true negatives.

**The PR Curve:**
- X-axis: Recall = TP / (TP + FN)
- Y-axis: Precision = TP / (TP + FP)

**When to use:** When classes are heavily imbalanced (fraud detection, rare disease diagnosis, information retrieval). AUC-PR is the metric that actually tells the truth in these scenarios.

**AUC-ROC vs AUC-PR (critical comparison):**

| Scenario | AUC-ROC | AUC-PR |
|----------|---------|--------|
| Balanced classes | Great | Great |
| Imbalanced classes (rare positives) | Overly optimistic | Honest |
| Dominated by true negatives | Hides the problem | Exposes it |
| Baseline (random) | Always 0.5 | Varies (equals positive rate) |
| Threshold-independent | Yes | Yes |

**Decision rule:** Use AUC-ROC for balanced problems. Use AUC-PR when the positive class is rare (<5% of data).

---

### 1.7 Log Loss (Cross-Entropy Loss)

**Intuition:** Accuracy asks "were you right or wrong?" Log loss asks "how confident were you, and were you right to be?" A model that says "90% spam" on a spam email is better than one that says "51% spam" on the same email. Log loss rewards well-calibrated probabilities.

**Formula (Binary):**

```
Log Loss = -1/N * Σ [yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ)]
```

Where yᵢ ∈ {0, 1} is the true label and ŷᵢ ∈ (0, 1) is the predicted probability.

**Why the log?**
- If true label = 1 and you predict ŷ = 0.99, loss = -log(0.99) ≈ 0.01 (tiny penalty)
- If true label = 1 and you predict ŷ = 0.01, loss = -log(0.01) ≈ 4.6 (massive penalty)
- The log creates an asymmetric, explosive penalty for confident-and-wrong predictions. It's infinitely bad to predict 0.0 for a true positive.

**When to use:** When you care about probability calibration, not just the final label. Common in logistic regression training, Kaggle competitions, and any system where the probability itself is used downstream (e.g., ranking by confidence, expected value calculations).

**Log Loss vs Accuracy:**
- Accuracy ignores HOW confident the model is. A model predicting 0.51 for everything gets the same accuracy as one predicting 0.99 (given the same threshold).
- Log loss differentiates them. The 0.99 model is better calibrated and gets a lower (better) log loss.

---

### 1.8 Specificity (True Negative Rate)

**Intuition:** Of all the actual negatives, how many did I correctly identify as negative? It's recall for the negative class.

**Formula:**

```
Specificity = TN / (TN + FP)
```

**Relationship to Recall:**
- Recall = How good am I at finding positives?
- Specificity = How good am I at leaving negatives alone?
- FPR = 1 - Specificity (which is why the ROC x-axis is sometimes labeled "1 - Specificity")

**When to use:** Medical screening, where you need to know: "Of all healthy people, how many did the test correctly clear?" A test with low specificity means lots of healthy people get unnecessary follow-up.

---

### 1.9 Matthew's Correlation Coefficient (MCC)

**Intuition:** Accuracy, precision, recall, F1 all have blind spots. MCC is a correlation coefficient between the observed and predicted binary classifications that uses ALL FOUR confusion matrix values (TP, TN, FP, FN). It's the most balanced metric for binary classification, especially with imbalanced classes.

**Formula:**

```
MCC = (TP * TN - FP * FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range:** [-1, +1]
- MCC = +1: Perfect prediction
- MCC = 0: Random prediction (no better than chance)
- MCC = -1: Total disagreement (labels flipped)

**Why MCC over F1?**
- F1 ignores true negatives entirely. A model that correctly identifies negatives gets no credit.
- MCC considers all four quadrants of the confusion matrix.
- In extreme class imbalance, F1 can be high even when the model is poor at negatives. MCC catches this.

**When to use:** When you want the single most robust metric for binary classification, especially with imbalanced data. Papers in bioinformatics and medical ML increasingly prefer MCC over F1.

**Gotcha:** MCC is undefined when any row or column of the confusion matrix sums to zero (denominator = 0). This happens when the model predicts only one class.

---

### Classification Metrics Comparison Table

| Metric | Handles Imbalance? | Threshold-Independent? | Considers TN? | Best For |
|--------|-------------------|----------------------|---------------|----------|
| Accuracy | No | No | Yes | Balanced problems only |
| Precision | Somewhat | No | No | False positives are costly |
| Recall | Somewhat | No | No | False negatives are costly |
| F1 | Somewhat | No | No | Balance of precision/recall |
| AUC-ROC | Poorly | Yes | Indirectly (via FPR) | Model comparison, balanced data |
| AUC-PR | Yes | Yes | No | Imbalanced data, rare positives |
| Log Loss | Yes | Yes (probability-based) | Yes | Calibrated probabilities |
| MCC | Yes | No | Yes | Most robust single metric |

---

## 2. Regression Metrics

### The Big Picture

Regression is about predicting numbers. The question is always: "How far off was my prediction?" But there are many ways to define "far off." Some metrics punish big errors more (MSE), some treat all errors equally (MAE), some are scale-independent (MAPE), and one tells you about explained variance (R²).

---

### 2.1 MSE (Mean Squared Error)

**Intuition:** Average the squared differences between predicted and actual values. Squaring does two things: (1) makes all errors positive, and (2) amplifies large errors disproportionately. A prediction that's off by 10 contributes 100 to MSE, while one that's off by 1 contributes only 1.

**Formula:**

```
MSE = (1/N) * Σ (yᵢ - ŷᵢ)²
```

**Why squared, not just absolute?**
- Squaring is differentiable everywhere (smooth gradient), making it easy to optimize with gradient descent. The absolute value function has a kink at zero.
- Squaring heavily penalizes outlier predictions, which can be a feature (you WANT the model to avoid big errors) or a bug (a single outlier dominates the metric).

**When to use:** Default for regression when:
- Large errors are genuinely worse than small ones (house price prediction where being off by ₹10L is 100x worse than being off by ₹1L)
- You're training a model with gradient descent (MSE gives clean gradients)

**When NOT to use:**
- When outliers exist in your target variable and you don't want them dominating the metric (use MAE or Huber instead)
- When you need interpretable units (use RMSE)

---

### 2.2 RMSE (Root Mean Squared Error)

**Intuition:** MSE gives you squared units (if you're predicting prices in ₹, MSE is in ₹²). RMSE just takes the square root to bring it back to the original units. That's it. Same behavior as MSE, just interpretable.

**Formula:**

```
RMSE = √MSE = √((1/N) * Σ (yᵢ - ŷᵢ)²)
```

**RMSE vs MAE:**
- RMSE ≥ MAE always (by Jensen's inequality)
- The gap between RMSE and MAE tells you about error variance. If RMSE >> MAE, you have some very large errors mixed in with small ones.
- RMSE = MAE only when all errors are exactly the same magnitude.

**When to use:** When you want MSE's behavior but need to report errors in the original unit. "Average error is ₹2.3 lakhs" is more meaningful than "MSE is 5.29 lakh-squared."

---

### 2.3 MAE (Mean Absolute Error)

**Intuition:** The simplest possible error metric. Just average the absolute differences. Unlike MSE, MAE treats all errors linearly, so a prediction off by 10 is exactly 10x worse than one off by 1 (not 100x).

**Formula:**

```
MAE = (1/N) * Σ |yᵢ - ŷᵢ|
```

**MAE vs MSE (critical comparison):**

| Property | MSE/RMSE | MAE |
|----------|----------|-----|
| Outlier sensitivity | High (squares amplify outliers) | Low (linear penalty) |
| Differentiability | Smooth everywhere | Kink at zero (not differentiable) |
| Gradient behavior | Gradient proportional to error size | Constant gradient magnitude |
| Interpretation | In squared units (MSE) or original (RMSE) | In original units |
| Optimal prediction | Mean of target distribution | Median of target distribution |

**Key insight for interviews:** The "optimal prediction" row is gold. If you minimize MSE, your model converges toward predicting the mean. If you minimize MAE, it converges toward predicting the median. This matters when the target distribution is skewed.

**When to use:** When outliers exist and you don't want them dominating. When you want a more robust central tendency (median-like behavior). Common in financial forecasting where occasional extreme values shouldn't distort the overall error picture.

---

### 2.4 MAPE (Mean Absolute Percentage Error)

**Intuition:** "What percentage was I off by, on average?" This makes errors scale-independent. Being off by ₹100 on a ₹1,000 item (10%) is worse than being off by ₹100 on a ₹10,000 item (1%). MAPE captures this.

**Formula:**

```
MAPE = (100/N) * Σ |yᵢ - ŷᵢ| / |yᵢ|
```

**When to use:** When you need a scale-independent error metric. Common in supply chain forecasting, sales predictions, and anywhere stakeholders think in percentages ("our forecasts are off by about 15%").

**When it lies to you (critical failure modes):**
1. **Division by zero:** If any actual value yᵢ = 0, MAPE blows up to infinity. This is not a theoretical concern. Demand forecasting for a product with zero sales on some days? MAPE is useless.
2. **Asymmetric penalty:** MAPE penalizes over-predictions more than under-predictions for the same absolute error. Predicting 150 when actual is 100 gives 50% error. Predicting 50 when actual is 100 also gives 50% error. But predicting 200 when actual is 100 gives 100%, while predicting 0 when actual is 100 gives 100% too. The asymmetry kicks in for different base values.
3. **Biased toward under-forecasting:** Because of point 2, models optimized on MAPE tend to systematically under-predict.

**Alternative:** SMAPE (Symmetric MAPE) tries to fix the asymmetry but introduces its own issues. WMAPE (Weighted MAPE) aggregates better across series.

---

### 2.5 R² (Coefficient of Determination)

**Intuition:** "How much of the variance in the data does my model explain?" R² compares your model's error to the dumbest possible baseline: always predicting the mean. If your model has the same error as the mean baseline, R² = 0. If your model is perfect, R² = 1. If your model is WORSE than the mean, R² < 0 (yes, negative R² is possible).

**Formula:**

```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ (yᵢ - ŷᵢ)²       (residual sum of squares = your model's error)
SS_tot = Σ (yᵢ - ȳ)²        (total sum of squares = variance of the data)
```

**Key values:**
- R² = 1: Model explains all variance (perfect predictions)
- R² = 0: Model is no better than predicting the mean
- R² < 0: Model is WORSE than predicting the mean (this happens; it means your model is actively harmful)

**When to use:** When you want to know what fraction of variance your model captures. Great for communicating to non-technical stakeholders: "Our model explains 85% of the variation in house prices."

**Gotchas:**
1. **R² always increases with more features** (even useless ones). Use Adjusted R² instead:
   ```
   Adjusted R² = 1 - [(1-R²)(N-1) / (N-p-1)]
   ```
   where p = number of features. Adjusted R² penalizes for adding features that don't help.

2. **R² doesn't tell you if predictions are biased.** A model that's consistently off by +₹5L has low R² but might be fixable with a simple offset.

3. **R² is scale-dependent on the problem.** R² = 0.3 might be amazing for stock prediction but terrible for physical measurement.

---

### 2.6 Huber Loss (Smooth L1 Loss)

**Intuition:** MSE is great for small errors but gets wrecked by outliers. MAE is robust to outliers but has a kink at zero (gradient issues). Huber loss is the best of both worlds: it behaves like MSE for small errors (smooth, easy to optimize) and like MAE for large errors (doesn't blow up).

**Formula:**

```
L_δ(y, ŷ) = 
    0.5 * (y - ŷ)²              if |y - ŷ| ≤ δ
    δ * |y - ŷ| - 0.5 * δ²      if |y - ŷ| > δ
```

Where δ (delta) is the threshold that separates "small" from "large" errors.

**Visualization:** Picture the MSE parabola for the center of the curve (|error| ≤ δ), smoothly transitioning to straight MAE lines on both sides (|error| > δ). The transition is smooth, no kinks.

**When to use:** When your data has outliers but you still want smooth gradients for optimization. Used extensively in object detection (Smooth L1 Loss in Faster R-CNN) and robust regression.

**Hyperparameter δ:** Controls the crossover point. Smaller δ = more MAE-like (more robust). Larger δ = more MSE-like (more sensitive to large errors). Typical default: δ = 1.0.

---

### Regression Metrics Comparison

| Metric | Outlier Robust? | Scale-Independent? | Differentiable? | Optimal Prediction |
|--------|----------------|-------------------|-----------------|-------------------|
| MSE | No | No | Yes | Mean |
| RMSE | No | No | Yes | Mean |
| MAE | Yes | No | No (kink at 0) | Median |
| MAPE | Yes | Yes | No | Weighted median |
| R² | No | Somewhat (0-1) | Yes | Mean |
| Huber | Configurable (δ) | No | Yes | Between mean and median |

---

## 3. Ranking & Recommendation Metrics

### The Big Picture

Recommendation systems don't just classify (relevant/not). They produce ORDERED LISTS. The position of an item in the list matters enormously. Showing the right movie at position #1 is worth way more than showing it at position #50. These metrics evaluate list quality with position awareness.

---

### 3.1 NDCG (Normalized Discounted Cumulative Gain)

**Intuition:** Not all relevant items are equally relevant (a 5-star movie is more relevant than a 3-star). And items at the top of the list matter more. NDCG handles both: it weights relevance by position, heavily discounting items further down the list.

**Formula (3 steps):**

**Step 1: Cumulative Gain (CG)** -- just sum up the relevance scores, ignoring position.
```
CG = Σ relᵢ
```

**Step 2: Discounted Cumulative Gain (DCG)** -- weight each item's relevance by its position using a log discount.
```
DCG@K = Σᵢ₌₁ᴷ relᵢ / log₂(i + 1)
```

Why log₂(i+1)? Position 1 gets discount 1/log₂(2) = 1.0. Position 2 gets 1/log₂(3) ≈ 0.63. Position 10 gets 1/log₂(11) ≈ 0.29. The log creates a gentle decay: top positions matter a lot, but later positions still contribute something.

**Step 3: Normalize** -- divide by the IDEAL DCG (what you'd get if items were perfectly sorted by relevance).
```
NDCG@K = DCG@K / IDCG@K
```

**Range:** [0, 1]. NDCG = 1 means the ranking is perfect.

**When to use:** When relevance is graded (not binary). Search engines, Netflix recommendations, any "top K" list where items have different quality levels. This is THE standard metric for learning-to-rank systems.

**When NOT to use:** When relevance is binary (clicked/not clicked). Use MAP instead.

---

### 3.2 MAP (Mean Average Precision)

**Intuition:** For binary relevance (relevant or not), MAP asks: "On average across queries, when you find relevant items, how clean is the list above them?" It computes precision at each position where a relevant item appears, then averages.

**Formula:**

```
For a single query:
AP = (1/R) * Σₖ₌₁ᴺ Precision@k * rel(k)

Where R = total relevant items, rel(k) = 1 if item at position k is relevant.

MAP = (1/Q) * Σ AP_q  (average across Q queries)
```

**Walk-through:** Suppose you have a ranked list: [Relevant, Irrelevant, Relevant, Irrelevant, Relevant]
- At position 1: Precision@1 = 1/1 = 1.0 (relevant hit!)
- At position 2: Skip (irrelevant)
- At position 3: Precision@3 = 2/3 = 0.67 (relevant hit!)
- At position 4: Skip (irrelevant)
- At position 5: Precision@5 = 3/5 = 0.60 (relevant hit!)
- AP = (1/3) * (1.0 + 0.67 + 0.60) = 0.76

**When to use:** Information retrieval, binary relevance, when you care about the overall quality of the ranked list. Standard metric in TREC evaluations.

**MAP vs NDCG:**

| | MAP | NDCG |
|---|-----|------|
| Relevance type | Binary (relevant/not) | Graded (1-5 stars) |
| Position discount | Implicit (via precision@k) | Explicit (log discount) |
| Use case | Search, document retrieval | Recommendations, LTR |

---

### 3.3 MRR (Mean Reciprocal Rank)

**Intuition:** "How far down the list do I have to scroll to find the FIRST relevant result?" MRR only cares about the first hit. If the first relevant result is at position 1, that's perfect. At position 5, you score 1/5 = 0.2.

**Formula:**

```
MRR = (1/Q) * Σ 1/rank_i

Where rank_i = position of the first relevant result for query i.
```

**When to use:** When only the top result matters. Question answering ("did the correct answer appear first?"), voice search (the device reads only the top result), autocomplete suggestions.

**When NOT to use:** When multiple relevant results matter (use MAP or NDCG). MRR ignores everything after the first hit.

**MRR vs MAP vs NDCG:**

| Metric | What it measures | Cares about position? | Multiple relevant items? |
|--------|-----------------|----------------------|------------------------|
| MRR | First relevant result | Yes (via reciprocal rank) | No (only first) |
| MAP | All relevant results | Yes (via precision@k) | Yes (binary relevance) |
| NDCG | All results with graded relevance | Yes (log discount) | Yes (graded relevance) |

---

### 3.4 Online / Business Metrics

These are what your product manager actually cares about. Offline metrics (NDCG, MAP) tell you about ranking quality. Online metrics tell you about business impact.

**CTR (Click-Through Rate):**
```
CTR = Clicks / Impressions
```
- Measures: "Did users engage with what we showed them?"
- Gotcha: High CTR doesn't mean high satisfaction. Clickbait has high CTR.

**Conversion Rate:**
```
Conversion Rate = Desired Actions / Total Visitors (or Clicks)
```
- Measures: "Did users do the thing we wanted?" (purchase, signup, subscribe)
- More meaningful than CTR because it captures downstream value.

**Dwell Time / Session Length:**
- Measures: "How long did users engage with the recommended content?"
- Proxy for satisfaction. Long dwell time on a recommended article = probably relevant.
- Gotcha: Long dwell time on a confusing page = bad UX, not good content.

**Connection between offline and online:**
You optimize offline metrics during model development, then A/B test to see if improvements in NDCG/MAP actually translate to CTR/conversion improvements. Sometimes they don't (the offline-online gap), which is one of the hardest problems in recommender systems.

---

## 4. Computer Vision Metrics

### 4.1 Object Detection

#### IoU (Intersection over Union)

**Intuition:** You predicted a bounding box around a dog. The ground truth is another box around the same dog. How much do they overlap? IoU measures the overlap area as a fraction of the total area covered by both boxes.

**Formula:**

```
IoU = Area of Intersection / Area of Union
```

**Visualization:** Picture two overlapping rectangles. The intersection is the area where both rectangles exist. The union is the total area covered by either rectangle. A perfect prediction has IoU = 1.0 (boxes are identical). No overlap = IoU = 0.

**Thresholds in practice:**
- IoU ≥ 0.5: PASCAL VOC standard ("loose" match)
- IoU ≥ 0.75: "strict" match
- IoU @ [0.5:0.95]: COCO standard (average over multiple thresholds)

**Why IoU and not just distance between centers?**
Two boxes can have the same center distance but very different overlap (one might be much larger than the other). IoU captures both position AND size accuracy.

#### mAP (Mean Average Precision for Object Detection)

**Intuition:** For each object class, compute AP (precision-recall curve area) at various IoU thresholds, then average across all classes. This is the standard metric for object detection benchmarks (COCO, PASCAL VOC).

**Formula:**

```
mAP = (1/C) * Σ AP_c

Where C = number of classes, and AP_c is computed at a specific IoU threshold (or averaged across thresholds for COCO).
```

**PASCAL VOC mAP:** AP at IoU ≥ 0.5 (single threshold)
**COCO mAP:** Average AP across IoU thresholds [0.5, 0.55, 0.60, ..., 0.95] (much stricter)

---

### 4.2 Image Classification

**Top-1 Error:**
```
Top-1 Error = (Predictions where argmax ≠ true label) / N
```
The model's highest probability prediction is wrong. This is just 1 - Accuracy for single-label classification.

**Top-5 Error:**
```
Top-5 Error = (Predictions where true label ∉ top 5 predictions) / N
```
The correct class doesn't appear in the model's 5 highest-confidence predictions. Used in ImageNet because with 1000 classes, getting the exact right answer is hard, but getting it in the top 5 is more reasonable.

**Why Top-5 exists:** With 1000 ImageNet classes, many are visually similar (different dog breeds). Top-5 measures: "Does the model at least narrow it down to the right neighborhood?"

---

### 4.3 Image Segmentation

#### Dice Coefficient (Dice-Sorensen)

**Intuition:** The F1 score but for pixels. For segmentation, you're classifying every pixel as belonging to an object or not. The Dice coefficient measures the overlap between your predicted pixel mask and the ground truth mask.

**Formula:**

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

Where A = set of pixels predicted as foreground, B = set of pixels actually foreground.

Equivalently (for binary):
```
Dice = 2*TP / (2*TP + FP + FN)
```

This is literally the F1 score formula. Dice = F1 for binary pixel classification.

**Range:** [0, 1]. Dice = 1 means perfect segmentation.

**Why Dice over IoU for segmentation?**
- Dice = 2*IoU / (1 + IoU). They're monotonically related, so they always rank models the same way.
- Dice is more commonly used as a LOSS function for training (Dice Loss) because it directly optimizes overlap. IoU is harder to differentiate.
- Medical imaging community strongly prefers Dice.

**Pixel Accuracy:**
```
Pixel Accuracy = Correctly classified pixels / Total pixels
```
Same gotcha as classification accuracy: if 95% of the image is background, predicting "all background" gives 95% pixel accuracy but captures nothing.

---

### 4.4 Image Generation (GANs / Diffusion Models)

#### FID (Frechet Inception Distance)

**Intuition:** "How similar is the DISTRIBUTION of generated images to the distribution of real images?" FID doesn't compare individual images. It passes both real and generated images through a pre-trained Inception-v3 network, extracts feature vectors, fits a multivariate Gaussian to each set, and measures the distance between the two Gaussians.

**Formula:**

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r * Σ_g)^(1/2))
```

Where (μ_r, Σ_r) and (μ_g, Σ_g) are the mean and covariance of real and generated feature distributions.

**Lower FID = Better.** FID = 0 means the distributions are identical.

**Why it works:** The Inception features capture high-level semantic content. Similar FID means generated images have similar diversity, quality, and semantic content as real images.

**Gotchas:**
- Depends on Inception-v3 (biased toward ImageNet-like images)
- Needs a large sample (minimum ~10K images for stable estimates)
- Doesn't measure individual image quality, only distributional similarity

#### SSIM (Structural Similarity Index)

**Intuition:** Compares two images the way a human would perceive similarity, by examining luminance, contrast, and structure separately, then combining them.

**Formula:**

```
SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ

Where:
l = luminance comparison = (2*μx*μy + C1) / (μx² + μy² + C1)
c = contrast comparison = (2*σx*σy + C2) / (σx² + σy² + C2)
s = structure comparison = (σxy + C3) / (σx*σy + C3)
```

**Range:** [-1, 1]. SSIM = 1 means the images are identical.

**SSIM vs PSNR:**
- PSNR measures raw pixel-level error (how different are the numbers?)
- SSIM measures perceptual similarity (how different do they LOOK to a human?)
- Two images can have the same PSNR but very different SSIM if the errors are structured (blur) vs random (noise).

#### PSNR (Peak Signal-to-Noise Ratio)

**Intuition:** "How much signal vs noise is there?" Higher PSNR = less noise = better reconstruction. It's based on MSE but expressed in decibels (log scale).

**Formula:**

```
PSNR = 10 * log₁₀(MAX² / MSE)

Where MAX = maximum possible pixel value (255 for 8-bit images)
```

**Range:** Higher is better. PSNR = ∞ means identical images. Typical "good" values: 30-50 dB.

**When to use:** Image compression, super-resolution, denoising. It's simple and widely reported but poorly correlates with human perception (use SSIM for that).

---

## 5. LLM & Generative AI Metrics

### The Big Picture

Traditional NLP metrics (BLEU, ROUGE) were designed for tasks with clear reference answers (translation, summarization). LLMs generate open-ended text where there's no single correct answer. This has pushed the field toward semantic metrics (BERTScore), framework-based evaluation (RAGAS), and LLM-as-judge approaches.

---

### 5.1 Perplexity

**Intuition:** "How surprised is the model by the text it sees?" A language model assigns a probability to each token. If the model predicts the actual next token with high probability, it's not very "perplexed." Low perplexity = the model understands the language well.

**Formula:**

```
Perplexity = exp(-1/N * Σ log P(wᵢ | w₁, ..., wᵢ₋₁))
```

This is just the exponentiated average negative log-likelihood per token.

**Interpretation:** If perplexity = 10, the model is, on average, as confused as if it had to choose uniformly among 10 equally likely words at each step.

**When to use:** Comparing language models on the same dataset. Training monitoring (perplexity should decrease during training). Evaluating how well a model has learned a domain.

**When it lies to you:**
- Low perplexity doesn't mean good generation. A model can predict common patterns well but still produce boring, repetitive text.
- Perplexity is only comparable across models evaluated on the same tokenizer and test set.
- Doesn't measure factuality, coherence, or usefulness at all.

---

### 5.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Intuition:** "How much of the reference summary appears in the generated summary?" ROUGE is recall-focused: it measures what fraction of the reference n-grams the model managed to reproduce.

**Variants:**
- **ROUGE-1:** Unigram overlap (individual words)
- **ROUGE-2:** Bigram overlap (pairs of consecutive words)
- **ROUGE-L:** Longest common subsequence (captures sentence-level structure)

**Formulas:**

```
ROUGE-N Recall = (Matching n-grams between generated and reference) / (Total n-grams in reference)

ROUGE-L = LCS(generated, reference) / Length of reference
```

**When to use:** Summarization evaluation. It's the standard metric in summarization research papers. ROUGE-2 is the most discriminative in practice.

**When it fails:** ROUGE is purely lexical. Two sentences can mean the same thing with zero word overlap. "The cat sat on the mat" and "A feline rested on the rug" get ROUGE-1 = 0 despite being semantically identical. This is why BERTScore was invented.

---

### 5.3 BERTScore

**Intuition:** Instead of matching exact words (like ROUGE), BERTScore matches meaning. It passes both the generated and reference texts through a pre-trained BERT model, gets contextual embeddings for each token, and then computes the best matching between tokens based on cosine similarity.

**Formula:**

```
Precision: For each token in candidate, find most similar token in reference → average
Recall: For each token in reference, find most similar token in candidate → average  
F1: Harmonic mean of the above
```

**Why it's better than ROUGE:** "Dog" and "canine" have cosine similarity ≈ 0.85 in BERT embedding space, so they'd get credit. In ROUGE, they get zero.

**When to use:** Any text generation evaluation where you want semantic similarity, not just lexical matching. Increasingly replacing ROUGE in research.

**Gotchas:**
- Slower than ROUGE (requires BERT inference)
- Depends on the pre-trained model used (RoBERTa-large is the default)
- Can be "tricked" by semantically similar but factually wrong text

---

### 5.4 BLEU (Bilingual Evaluation Understudy)

**Intuition:** The oldest automatic metric for text generation, designed for machine translation. "How many of the n-grams in the generated text also appear in the reference?" BLEU is precision-focused (opposite of ROUGE which is recall-focused).

**Formula:**

```
BLEU = BP * exp(Σ wₙ * log(pₙ))

Where:
pₙ = modified precision for n-grams of length n
wₙ = weights (typically 1/4 for n = 1,2,3,4)
BP = brevity penalty = min(1, exp(1 - ref_length/gen_length))
```

The brevity penalty prevents cheating by generating very short text (which could have high precision).

**BLEU vs ROUGE:**

| | BLEU | ROUGE |
|---|------|-------|
| Focus | Precision (what % of generated n-grams are in reference) | Recall (what % of reference n-grams are in generated) |
| Designed for | Machine translation | Summarization |
| Brevity penalty | Yes | No |
| N-gram range | Typically 1-4 (combined) | Individual (ROUGE-1, ROUGE-2, etc.) |

**When to use:** Machine translation benchmarking. Still widely reported but increasingly criticized for poor correlation with human judgment.

**When NOT to use:** Open-ended generation, creative writing, anything where the output doesn't need to closely match a reference.

---

### 5.5 RAG Metrics (RAGAS Framework)

RAG (Retrieval-Augmented Generation) systems have two components that can fail independently: the retriever and the generator. RAGAS provides metrics for each.

#### Faithfulness

**Intuition:** "Did the model make stuff up, or did it only say things supported by the retrieved context?" Faithfulness measures if the answer is grounded in the provided documents.

**How it's computed:** Each claim in the answer is checked against the retrieved context. Claims not supported by the context are unfaithful.

```
Faithfulness = (Number of claims supported by context) / (Total claims in answer)
```

**Why it matters:** This is the hallucination detector for RAG systems. A model with high faithfulness stays close to its sources.

#### Answer Relevancy

**Intuition:** "Does the answer actually address what the user asked?" A factually correct answer to the wrong question scores low.

**How it's computed:** Generate hypothetical questions from the answer, then measure semantic similarity between those questions and the original question.

#### Context Precision

**Intuition:** "Were the retrieved chunks actually useful?" If your retriever returns 5 chunks but only 1 is relevant, context precision is low.

```
Context Precision = Relevant chunks retrieved / Total chunks retrieved
```

#### Context Recall

**Intuition:** "Did the retriever find everything it needed?" If there are 3 relevant documents in the corpus and the retriever only finds 1, context recall is low.

```
Context Recall = Relevant chunks retrieved / Total relevant chunks in corpus
```

#### Hallucination Rate

```
Hallucination Rate = Ungrounded claims / Total claims
```

This is essentially 1 - Faithfulness.

**RAGAS in practice:** These metrics are typically computed using an LLM-as-judge (GPT-4 or Claude evaluating the outputs). They're not exact but provide a scalable way to evaluate RAG pipelines.

---

### 5.6 Human & Advanced Evaluation

#### LLM-as-Judge

**Concept:** Use a strong frontier model (GPT-4, Claude) to evaluate outputs of a weaker model. You give the judge a rubric and ask it to score the output on criteria like helpfulness, accuracy, coherence, safety.

**Why it works:** High correlation (0.8-0.9) with human judgments in many studies. Much faster and cheaper than hiring human evaluators.

**Failure modes:**
- Position bias (judges prefer the first response in A/B comparisons)
- Verbosity bias (longer = higher rated, even if worse)
- Self-preference (GPT-4 rates GPT-4 outputs higher)
- Rubric sensitivity (small changes in the judging prompt change results)

#### Elo Rating

**Intuition:** Borrowed from chess. Two models are shown the same prompt. A human (or LLM judge) picks the better response. Over many such matchups, each model accumulates an Elo score reflecting its relative strength.

**How it works:**
```
After each comparison:
R_new = R_old + K * (actual - expected)

Where expected = 1 / (1 + 10^((R_opponent - R_self)/400))
```

**Used by:** Chatbot Arena (LMSYS), which is the most influential LLM benchmark as of 2026.

#### Side-by-Side (SBS) & Likert Ratings

- **SBS:** Show two outputs, human picks the better one. Simple but requires many comparisons.
- **Likert:** Rate each output 1-5 on specific dimensions (helpfulness, accuracy, fluency). More granular but introduces inter-rater variability.

---

## 6. Reinforcement Learning Metrics

### The Big Picture

RL agents aren't evaluated on predictions. They're evaluated on actions and outcomes over time. The fundamental question: "Did the agent learn to behave well?"

---

### 6.1 Cumulative Reward (Return)

**Intuition:** The total reward the agent collects during one episode. This is the thing RL is trying to maximize.

**Formula:**

```
G_t = Σₖ₌₀^∞ γᵏ * r_{t+k+1}

Where γ = discount factor (0 < γ ≤ 1)
```

**The discount factor γ:** Controls how much the agent cares about future vs immediate rewards. γ = 0.99 means the agent is patient (values future rewards almost as much as current). γ = 0.5 means the agent is myopic (strongly prefers immediate rewards).

### 6.2 Average Episode Length / Survival Time

**Intuition:** In tasks like CartPole or game playing, longer survival = better policy. If the agent keeps the pole balanced for 500 steps instead of 20, it's doing better.

### 6.3 Success Rate

```
Success Rate = Episodes where goal achieved / Total episodes
```

**When to use:** Goal-conditioned RL where there's a clear binary outcome (reached the target, solved the maze, delivered the package).

### 6.4 Value Loss / Policy Loss

**Intuition:** Internal training metrics. Value loss measures how well the value network estimates future rewards. Policy loss measures how well the policy is being optimized (depends on the algorithm: PPO, A2C, REINFORCE, etc.).

These converge during training. If value loss stops decreasing, the agent's understanding of "what's good" has plateaued.

### 6.5 Offline RL Metrics

**The Problem:** In offline RL, you have a dataset collected by some old policy (e.g., a human doctor's treatment decisions). You want to estimate how a NEW policy would perform WITHOUT actually deploying it. This is counterfactual reasoning.

**IPS (Inverse Propensity Score):**
```
V_IPS(π) = (1/N) * Σ (π(aᵢ|sᵢ) / π_old(aᵢ|sᵢ)) * rᵢ
```

Reweights historical outcomes by how likely the new policy would have taken the same action. High variance because the importance weights can be extreme.

**Doubly Robust Estimator:** Combines IPS with a value function estimate. More stable than IPS alone because if either the importance weights OR the value function estimate is correct, the overall estimate is unbiased.

---

## 7. Clustering Metrics (Unsupervised Learning)

### The Big Picture

Clustering has no labels, so how do you know if it worked? Intrinsic metrics (Silhouette, Davies-Bouldin) measure cluster quality without ground truth. Extrinsic metrics (ARI) require ground truth labels.

---

### 7.1 Silhouette Score

**Intuition:** For each point, ask: "Am I in the right cluster?" Compare how close I am to points in my own cluster (a = intra-cluster distance) vs how close I am to points in the nearest OTHER cluster (b = nearest-cluster distance).

**Formula:**

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Range:** [-1, 1]
- s ≈ 1: Point is well-matched to its cluster, far from others (good)
- s ≈ 0: Point is on the border between two clusters
- s < 0: Point is probably in the wrong cluster

**Average Silhouette Score:** Average s(i) across all points. Used to choose the optimal number of clusters K.

### 7.2 Davies-Bouldin Index

**Intuition:** For each cluster, find the cluster that's most "similar" to it (where similarity = high intra-cluster spread + low inter-cluster distance). Average this worst-case similarity across all clusters.

**Formula:**

```
DB = (1/K) * Σᵢ max_{j≠i} [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]

Where σᵢ = average distance of points in cluster i to centroid, d(cᵢ, cⱼ) = distance between centroids.
```

**Lower is better.** Unlike Silhouette, DBI has no fixed range.

**DBI vs Silhouette:**

| | Silhouette | Davies-Bouldin |
|---|-----------|----------------|
| Range | [-1, 1] (interpretable) | [0, ∞) (relative only) |
| Computation | O(n²) pairwise distances | O(n*K) faster |
| Interpretation | Per-point quality available | Only cluster-level |
| Better when | N is small enough for pairwise | N is large |

### 7.3 Adjusted Rand Index (ARI)

**Intuition:** If you have ground truth labels, ARI measures agreement between the true clusters and predicted clusters, adjusted for chance. Without the adjustment, random cluster assignments can score high simply due to many clusters.

**Formula:**

```
ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)
```

Where RI (Rand Index) = fraction of point pairs that are either in the same cluster in both assignments or in different clusters in both.

**Range:** [-1, 1]. ARI = 1 means perfect agreement. ARI ≈ 0 means random. ARI < 0 means worse than random.

**When to use:** When you have ground truth and want to evaluate clustering quality. Common in research papers comparing clustering algorithms.

---

## 8. MLOps & Production Monitoring Metrics

### The Big Picture

Models degrade in production because the world changes. The data your users generate today might look different from the training data. These metrics detect when your model is becoming stale.

---

### 8.1 Data Drift Metrics

#### PSI (Population Stability Index)

**Intuition:** "Has the distribution of my input features shifted since training?" PSI compares the training distribution to the current production distribution, bucket by bucket.

**Formula:**

```
PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
```

Where you bin the feature values and compare the percentage in each bin.

**Thresholds (industry standard):**
- PSI < 0.1: No significant shift
- 0.1 ≤ PSI < 0.2: Moderate shift, investigate
- PSI ≥ 0.2: Significant shift, model likely degraded, retrain

**When to use:** Monitoring production models. Run PSI on key features weekly/monthly. If PSI crosses 0.2, trigger a retraining pipeline.

#### KS Test (Kolmogorov-Smirnov)

**Intuition:** "What's the maximum difference between two cumulative distribution functions?" Unlike PSI which sums differences across bins, KS finds the single point of maximum divergence.

**Formula:**

```
KS Statistic = max |F_train(x) - F_prod(x)|
```

**Visualization:** Plot two CDFs on the same axes. The KS statistic is the largest vertical gap between them.

**KS vs PSI:**
- KS is a formal statistical test with p-values. PSI is a practical index with rules of thumb.
- KS is more sensitive to distributional shape differences. PSI is more sensitive to frequency changes in specific bins.
- Use both together for robust drift detection.

#### KL Divergence (Kullback-Leibler)

**Intuition:** "How much information do I lose if I use distribution Q (my model's assumption) instead of distribution P (reality)?" KL Divergence measures the "surprise" of seeing data from P when you expected Q.

**Formula:**

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

**Key properties:**
- KL ≥ 0 always (Gibbs' inequality)
- KL = 0 iff P = Q
- **NOT symmetric:** KL(P||Q) ≠ KL(Q||P). This is not a true distance metric.
- Undefined if Q(x) = 0 where P(x) > 0 (can't explain data that your model says is impossible)

**KL vs PSI:** PSI is actually a symmetric version of KL divergence: PSI = KL(P||Q) + KL(Q||P). So PSI "fixes" the asymmetry problem.

---

### 8.2 Explainability: SHAP Values

**Intuition:** For a specific prediction, SHAP answers: "How much did each feature contribute to pushing the prediction away from the average?" Based on Shapley values from cooperative game theory (each feature is a "player" contributing to the prediction "game").

**Formula:**

```
φᵢ = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)! / |N|!] * [f(S ∪ {i}) - f(S)]
```

In plain English: For each possible subset of features, compute how much adding feature i changes the prediction. Weight these by the number of possible orderings. Average across all subsets.

**Key properties:**
- **Additivity:** SHAP values for all features sum to (prediction - average prediction). This means they're a complete decomposition.
- **Consistency:** If a feature's contribution increases in a new model, its SHAP value also increases.
- **Local accuracy:** The sum of base value + all SHAP values = the actual prediction.

**When to use:** Model debugging ("why did the model reject this loan application?"), regulatory compliance (EU AI Act requires explanations), feature importance analysis, bias detection.

**Complexity:** Computing exact SHAP values is O(2^n) where n = number of features. In practice, TreeSHAP (for tree models, O(TLD²)) and KernelSHAP (model-agnostic approximation) make it tractable.

---

## Quick Reference: Which Metric for Which Task

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

---

## Interview Power Moves

1. **"95% accuracy" trap:** Always ask about class distribution first.
2. **"Which metric?":** Start with "What's more expensive, a false positive or false negative?" This determines precision vs recall emphasis.
3. **AUC-ROC vs AUC-PR:** "I use AUC-PR when the positive class is rare because AUC-ROC is inflated by true negatives."
4. **RMSE vs MAE:** "It depends on whether I want my model to target the mean or the median of the distribution."
5. **BLEU/ROUGE vs BERTScore:** "Lexical overlap metrics miss paraphrases. BERTScore captures semantic similarity but is slower."
6. **Offline vs Online metrics:** "I optimize NDCG offline but validate with A/B tests on CTR/conversion because the offline-online gap is real."
7. **Drift detection:** "I monitor PSI on key features weekly. PSI > 0.2 triggers investigation and potential retraining."
8. **Explainability:** "SHAP values decompose each prediction into per-feature contributions that sum to the prediction. This satisfies both debugging and compliance needs."
