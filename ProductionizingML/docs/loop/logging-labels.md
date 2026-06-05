# Logging and Labels

If you do not log what the model saw and did, you cannot debug or evaluate it later. And because the truth often arrives weeks after the prediction, you also need proxy metrics to fly in the meantime. This page covers prediction logging and the delayed-label problem together, because they are two halves of the same evaluation bridge.

!!! tip "Rapid Recall"
    Prediction logging is the bridge between serving and monitoring: at request time you rarely know if the prediction is right, so you log enough to join future labels back. A useful log has request id, entity id, timestamp, model version, feature values or references, feature timestamps, raw and calibrated score, threshold, decision, latency, fallback status, and join keys. Never forget model version, or attributing a loss rise to a model, threshold, feature, launch, or traffic shift becomes guesswork. Because labels are delayed, teams use proxy metrics (approval rate, review rate, score distribution, complaints), but proxies are early warning signals, not truth, and optimizing them blindly can block too many good users while a single slice quietly regresses.

## §1 What to Log at Prediction Time

Prediction logging is the bridge between serving and monitoring. At request time, you rarely know whether the prediction is correct. The label may arrive later. So you log enough information to join future labels back to past predictions.

A useful prediction log contains: request id, entity id, timestamp, model name and version, feature values or feature references, feature timestamps, raw score, calibrated score, threshold, final decision, latency, fallback status, and join keys for future labels. You may not store every raw feature forever because of cost and privacy, but you need enough evidence to reconstruct behavior.

Do not forget model version. If fraud loss rises, you need to know whether the rise started after a new model, a new threshold, a new feature definition, a product launch, or a traffic shift. Without versioned logs, everything becomes guesswork.

| Logged field | Why it matters |
|---|---|
| prediction_timestamp | Needed for point-in-time analysis and label windows |
| model_version | Connects behavior to a deployed artifact |
| feature_timestamp/freshness | Detects stale feature serving |
| score and decision | Separates model output from business threshold |
| request/entity id | Allows delayed labels to join later |
| latency and fallback | Separates quality failures from serving failures |

## §2 Delayed Labels and Proxy Metrics

You often do not know today's model accuracy today.

In fraud, a transaction may be reported as fraud days or weeks later. In churn, the user may churn after 30 days. In lending, default may take months. In recommendations, long-term satisfaction may not be visible from immediate clicks. This means production evaluation has a delay.

Because labels are delayed, teams use proxy metrics. For fraud, proxies might include approval rate, manual review rate, model score distribution, payment authorization failures, customer complaints, early dispute signals, and feature missingness. Proxy metrics are not truth. They are early warning signals.

The danger is optimizing proxies blindly. A model can reduce fraud loss by blocking too many good users. Approval rate can look stable while one country regresses. Click-through rate can rise while long-term retention falls. The production loop must connect proxy metrics to delayed ground truth and business metrics.

!!! warning "Interview trap"
    "Monitor accuracy" is incomplete if labels are delayed. Say what you monitor before labels arrive and how labels eventually join back.

## Interview Questions

**Q1: Why is prediction logging the bridge between serving and monitoring?**
Because at request time you usually do not know whether the prediction is correct; the label arrives later. So you log enough to join future labels back to past predictions: request and entity ids, timestamp, model version, feature values or references and their timestamps, raw and calibrated score, threshold, decision, latency, and fallback status. Without that record, you cannot evaluate or debug the model once truth arrives.

**Q2: Why is model_version one of the most important fields to log?**
Because when a business metric like fraud loss rises, you need to know whether the change started after a new model, a new threshold, a new feature definition, a product launch, or a traffic shift. Versioned logs let you attribute the regression to a cause; without them every investigation becomes guesswork.

**Q3: What are proxy metrics, and what is the danger of optimizing them?**
Proxy metrics are early-warning signals used while true labels are delayed: approval rate, manual review rate, score distribution, authorization failures, complaints, early disputes, feature missingness. They are not truth, so optimizing them blindly is dangerous, a model can cut fraud loss by blocking too many good users, or click-through can rise while long-term retention falls. The loop must connect proxies back to delayed ground truth and business metrics.

**Q4: Why is "monitor accuracy" an incomplete answer when labels are delayed?**
Because accuracy needs labels, and if those arrive days or weeks later, you cannot compute it in real time. A complete answer states what you monitor before labels arrive (proxy metrics, drift, score distribution, slices) and how labels eventually join back via the logged join keys to give true performance once they mature.
