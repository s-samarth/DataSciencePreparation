# Labels and Ground Truth Are Products Too

A label is the answer you train the model to predict. In production ML, labels are often delayed, noisy, biased, and expensive. Treating them as a product, with a source, a population, and a delay, is what separates a real ML system from a clean notebook.

!!! tip "Rapid Recall"
    Beginners treat labels as a clean CSV; in real systems labels are created by product behavior, human decisions, external systems, or future outcomes, and each source has bias. Label delay changes system design: if fraud labels arrive 30 days later, you cannot know today's accuracy today, so you need proxy metrics. Label bias matters too: if only high-risk transactions get reviewed and labeled, training data overrepresents suspicious traffic, which is selection bias from the previous production policy. In a clean notebook y is given; in production y is engineered, so always ask where it came from, when it arrives, what population it covers, and what policy created it.

## §1 Labels are engineered, not given

Beginners often treat labels as if they are already sitting in a clean CSV. In real systems, labels are created by product behavior, human decisions, external systems, or future outcomes. Each source has bias.

| Label type | Example | What can go wrong |
|---|---|---|
| Explicit user label | User clicks "this is spam" | Only angry or motivated users label; many cases remain unlabeled. |
| Implicit behavioral label | Watch time, click, purchase | Behavior reflects exposure and UI position, not pure preference. |
| Delayed outcome | Chargeback, loan default, churn | Truth arrives weeks later, so monitoring is delayed. |
| Human review | Moderator marks content unsafe | Reviewers disagree; policies change; reviewed items are not random. |
| Weak or synthetic label | Rule-based fraud flag, LLM judge | The model may learn the rule or judge bias instead of the real target. |

## §2 Label delay changes system design

Label delay changes system design. If fraud labels arrive 30 days later, you cannot know today's model accuracy today. You need proxy metrics: approval rate, score distribution, manual review rate, user complaints, payment authorization failure, and early chargeback signals. The [Production Loop](../loop/index.md) section uses this idea for monitoring.

## §3 Label bias

Label bias also matters. Suppose you only send high-risk transactions to human review, and only reviewed transactions get labels. Then your training data overrepresents suspicious transactions. If you train naively, the model may not understand normal traffic well. This is selection bias caused by the previous production policy.

!!! note "Interview note"
    Data-science analogy: in a clean notebook, y is given. In a production system, y is engineered. You must ask where it came from, when it arrives, what population it covers, and what policy created it.

## Interview Questions

**Q1: Why is treating labels as a clean given a mistake in production ML?**
Because labels are created by product behavior, human decisions, external systems, or future outcomes, and every source carries bias: explicit labels come only from motivated users, implicit labels reflect UI exposure, delayed outcomes arrive late, human review is inconsistent, and weak labels encode a rule rather than the truth. You have to ask where the label came from, when it arrives, what population it covers, and what policy created it.

**Q2: How does label delay affect monitoring?**
If the true label arrives weeks later, you cannot compute today's accuracy today, so you fall back on proxy metrics: approval rate, score distribution, manual review rate, user complaints, payment authorization failures, and early chargeback signals. These give an early warning while you wait for ground truth to mature.

**Q3: What is policy-induced selection bias in labels?**
When the previous production policy decides which records get labeled, for example only sending high-risk transactions to human review, the labeled training data overrepresents suspicious cases and underrepresents normal traffic. A model trained naively on it may misunderstand normal behavior, so you must account for how the labeling policy shaped the population.
