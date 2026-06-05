# Features, Freshness, and Leakage

A feature is a fact made available to the model at prediction time. The phrase "at prediction time" is what keeps you honest, and it is the entire defense against leakage, the bug that makes a model look perfect offline and useless in production.

!!! tip "Rapid Recall"
    Features come in three flavors: simple fields, aggregates over a window, and learned embeddings. Feature freshness is a model-quality requirement, not just an infrastructure preference: if the model must react to attacks in seconds, a daily batch pipeline is not enough. Leakage means the model learns from information it will not have in production. The four forms are future leakage (facts updated after prediction time), target leakage (a field that encodes the answer), duplicate leakage (near-identical rows in train and test), and policy leakage (labels created by an old model or rule). The honest phrase is "historical as of prediction time," not "all historical data."

## §1 What a feature is

Features can be simple fields such as transaction amount, country, account age, device type, or payment method. They can be aggregate features such as failed logins in the last 10 minutes, number of cards used in the last 24 hours, average order value in the last 30 days, or distance from usual shipping location. They can be learned features such as embeddings for user behavior, product text, images, or documents.

Feature freshness is a system requirement. Account age can be computed from a database record and cached. Failed logins in the last 10 minutes needs recent event data. A daily batch pipeline is not enough if the model must react to attacks in seconds. This is where batch vs stream processing becomes a model-quality decision, not just an infrastructure preference.

## §2 Leakage explained slowly

Leakage means the model learns from information it will not have in production. The easiest example is using the label as a feature. If you train a fraud model with `chargeback_occurred` as an input, the model will look perfect and be useless. But leakage is usually subtler.

**Future leakage:** using facts updated after prediction time, such as final order status or future chargeback count.

**Target leakage:** using a field that directly encodes the answer, such as manual review outcome when the target is fraud.

**Duplicate leakage:** nearly identical examples appear in train and test, so the model memorizes users or transactions.

**Policy leakage:** labels are created by an old model or rule, and the new model simply learns that policy instead of the true outcome.

## §3 Leakage detector

Pick a candidate feature and ask whether it would be known at payment time. The verdict for each candidate from the running fraud scenario:

| Candidate feature | Verdict | Why |
|---|---|---|
| `chargebacks_in_next_30_days` | Leakage | Future information and basically the label. The model looks excellent offline and fails at serving time because this value is unknown at payment time. |
| `failed_logins_last_10_min` | Valid (with care) | Valid if computed only from login events before payment submission and served within the checkout latency budget. It likely requires streaming or a very fresh online store. |
| `account_age_at_payment_time` | Valid | Computable from account creation timestamp and prediction timestamp. Stable and easy to reproduce point-in-time. |
| `final_order_status` | Usually leakage | Final status often includes cancellation, fulfillment, refund, manual review, or fraud investigation outcomes that happen after payment time. |
| `avg_order_value_last_30_days_as_of_payment` | Valid (as-of payment) | Valid if computed as of payment time. Invalid if computed using today's full history when creating old training examples. |

!!! warning "Interview trap"
    Saying "we will use all historical data" is not enough. You must say "historical as of prediction time." That phrase is the difference between a real ML system answer and a Kaggle answer.

## Interview Questions

**Q1: What is data leakage, and why is the label-as-feature case the easy version?**
Leakage is when the model learns from information it will not have at serving time. Using `chargeback_occurred` as an input makes the model look perfect offline and useless in production because the value is unknown at payment time. The real danger is the subtler forms, where a field quietly encodes the future or the answer.

**Q2: Name the forms of leakage beyond using the label directly.**
Future leakage uses facts updated after prediction time, such as final order status. Target leakage uses a field that directly encodes the answer, such as a manual review outcome. Duplicate leakage puts near-identical rows in train and test so the model memorizes entities. Policy leakage trains on labels created by an old rule or model, so the new model just relearns that policy.

**Q3: Why is feature freshness a model-quality decision and not just infrastructure?**
Because some features only carry signal if they are recent. Failed logins in the last 10 minutes is useless if it is a day stale, so a model that must react to attacks in seconds needs a streaming pipeline or fresh online store, not a daily batch. The freshness requirement comes from the model's job, which forces the batch-versus-stream choice.

**Q4: What single phrase signals you understand point-in-time correctness?**
"Historical as of prediction time." Saying you will use all historical data is not enough, because it invites computing features from information that only exists after the decision. The as-of-prediction-time qualifier is the difference between a production ML answer and a Kaggle answer.
