# Batch Inference System - Part 7 - Interview Style

# Case Studies: Daily Host Junk Risk Scoring and Daily User/Ad CTR Scoring

This note belongs with:

```text
Ensemble Learning in Production
Part 5 — Batch Inference System
Part 5.14 — Case Study: Daily Host Junk Risk Scoring
Part 5.15 — Case Study: Daily User/Ad CTR Scoring
```

These notes convert the final Batch Inference case studies into interview-speaking answers.

---

# 5.14 Daily Host Junk Risk Scoring — Interview Style

## 1. How would you design a daily host junk-risk scoring system?

### Interview answer

I would design it as a batch inference pipeline because host-level quality signals are mostly historical aggregates that change over hours or days rather than milliseconds.

The pipeline would first build the eligible host population, for example active hosts seen in the last 30 days with valid canonical host IDs. Then I would join host-level features such as crawl-error ratios, crawl coverage, staleness, spam signals, DSAT metrics, redirect behavior, and engagement signals.

After validating row count, uniqueness, feature coverage, freshness, and schema, I would load an explicit approved model version, such as an XGBoost model, together with its feature schema, missing-value rules, calibration layer, and threshold policy.

The model-ready host table would then be partitioned, for example using `hash(host_id) % N`, and workers would score the partitions in parallel. Each worker would write temporary output with metadata such as host ID, junk probability, risk bucket, model version, scoring run ID, feature timestamp, and data-quality flags.

After all partitions finish, I would validate output completeness, uniqueness, score ranges, model consistency, and score distribution before atomically publishing the new host-risk snapshot. Downstream systems could then use the snapshot for crawl prioritization, manual review, quality investigation, or policy workflows.

---

## 2. Why is batch inference suitable for host junk scoring?

### Interview answer

Batch inference is suitable because most host-risk signals are historical aggregates rather than request-time context.

For example, the model may use the last 10 days of crawl errors, 30-day dissatisfaction ratios, spam signals, redirect rates, dwell time, or crawl coverage. These do not need to be recomputed on every downstream request.

So instead of serving the model in real time, I can score the active host population once per day, store the result, and reuse it across multiple systems.

This reduces real-time infrastructure complexity, allows expensive aggregate features, and makes it practical to score millions of hosts consistently.

---

## 3. How would you define the scoring population?

### Interview answer

I would start from the full host universe and create an eligibility layer.

For example, I may require that a host was active recently, has a valid canonical identifier, belongs to the supported traffic scope, and has enough evidence to score meaningfully.

The goal is to avoid wasting compute on inactive, unsupported, or permanently resolved hosts.

I would also validate the size of the daily population because a sudden large drop or spike can indicate an upstream problem.

---

## 4. What features would you use for host junk-risk scoring?

### Interview answer

I would combine multiple feature groups because host junk risk is usually not captured by one signal.

Typical groups include crawl and error signals, crawl coverage or staleness, user dissatisfaction, engagement, spam signals, and host-size or evidence-support features.

For example, I may use ratios like recent junk URLs divided by recently crawled URLs, error-related URLs divided by crawled URLs, and crawl coverage relative to total indexed URLs. I would also include DSAT-related ratios, spam ratio, average dwell time, and support counts.

The support counts are important because a ratio of 1.0 based on one observation should not be treated the same as a ratio of 1.0 based on one hundred thousand observations.

---

## 5. How would you handle sparse or low-evidence hosts?

### Interview answer

I would preserve the distinction between missing evidence, observed zero, and low-support evidence.

For example, `1 error / 1 crawl` and `100,000 errors / 100,000 crawls` both give an error ratio of 1.0, but the confidence is very different.

So I would include support features such as crawl count, click count, or judgment count, and I may also use smoothing or low-evidence indicators.

The model should learn not only the ratio but also how much evidence supports that ratio.

---

## 6. How would you partition and score the host population?

### Interview answer

I would use balanced partitioning, commonly hash partitioning on the host ID.

For example, with 20 million hosts and 200 partitions:

```text
partition_id = hash(host_id) % 200
```

Each partition would contain roughly 100,000 hosts.

Each worker would load the same model artifact once, validate the model checksum and feature schema, score rows in batches, and write its output to a temporary partition-specific location.

Hash partitioning helps avoid skew and makes retries easy because a failed partition can be recomputed independently.

---

## 7. What would the prediction output contain?

### Interview answer

I would treat the output as a versioned production data product, not just a score.

A prediction row may contain:

```text
host_id
junk_score
calibrated_probability
risk_bucket
model_version
feature_pipeline_version
threshold_policy_version
scoring_run_id
partition_id
scored_at
feature_timestamp
score_valid_until
missing_feature_count
data_quality_status
```

This gives downstream systems enough information to interpret, trace, and safely use the prediction.

---

## 8. How would you separate model score from business action?

### Interview answer

I would keep the model output and policy layer separate.

The model may produce a calibrated junk probability, while a separate policy maps the probability to actions such as no action, monitor, manual review, or stronger intervention.

This separation is useful because threshold policy can change without retraining the model.

I would version the policy separately so I can explain whether a change in behavior came from a new model or from a new threshold rule.

---

## 9. Would you use incremental scoring for host risk?

### Interview answer

Yes, especially at large scale.

Instead of rescoring every host every day, I can identify hosts whose risk-relevant information changed, such as hosts with new crawl activity, new DSAT data, new spam signals, newly discovered hosts, expired scores, or late-arriving events.

I would rescore those hosts and reuse still-valid predictions for the rest.

However, if the model version or feature definition changes materially, I would usually run a full refresh to avoid mixing incompatible predictions.

A practical pattern could be daily incremental scoring plus a periodic full rebuild.

---

## 10. How would you handle failure in the host-junk pipeline?

### Interview answer

I would use stage-level validation and never publish partial or untrusted output.

If one partition fails, I would retry only that partition. If the model checksum is wrong, I would stop scoring immediately. If output row count is significantly lower than expected, I would block publishing.

If the current run fails completely, I may keep the previous successful host-risk snapshot active for a bounded freshness window.

If a non-critical feature group fails and the model was trained to handle missing values safely, I may enter degraded mode and mark predictions accordingly, possibly restricting automated actions.

---

## 11. What would you monitor in this system?

### Interview answer

I would monitor the full pipeline.

For job health: runtime, SLA, retries, and failed partitions.

For input health: active-host count, duplicates, and source freshness.

For feature health: coverage, missing rates, freshness, and distributions for crawl, spam, DSAT, and engagement signals.

For scoring: model version, checksum, throughput, and partition runtimes.

For output: prediction count, null scores, duplicate IDs, score distribution, and risk-bucket distribution.

For business impact: number of hosts entering review, action rate, review precision, and false-positive feedback.

---

## Complete 5.14 Answer

If asked:

```text
Design a production batch system for daily host junk-risk scoring.
```

Answer:

```text
I would treat host junk scoring as a daily batch inference problem because the important signals are historical aggregates and do not require millisecond freshness.

I would first build the eligible host population, canonicalize host IDs, and join feature groups such as crawl errors, crawl coverage, staleness, spam, DSAT, redirect behavior, dwell time, and support counts. Before scoring, I would validate uniqueness, row count, feature coverage, ranges, freshness, and schema.

Then I would load an explicit approved ensemble model version together with its preprocessing, missing-value rules, class mapping, calibration, threshold policy, and feature schema. The host table would be hash-partitioned and scored in parallel, with every worker loading the same model artifact once.

The output would include host ID, calibrated junk probability, risk bucket, model version, feature version, scoring run ID, timestamps, and data-quality flags. I would write outputs to a temporary location, validate all partitions and global score distributions, and atomically publish only after the run is complete and trustworthy.

At scale, I would use incremental rescoring for hosts whose features changed or scores expired, while running full refreshes on major model or feature changes. If a run fails, I would retry isolated partitions and fall back to the previous successful snapshot for a bounded period rather than publishing incomplete output.
```

---

# 5.15 Daily User/Ad CTR Scoring — Interview Style

## 12. How would you use batch inference in a CTR system?

### Interview answer

CTR is usually a hybrid problem rather than a purely batch problem.

The final click probability may depend on live request context such as current page, current query, device, time, and recent session behavior. Those signals belong in the real-time layer.

However, many useful CTR signals change slowly, such as user long-term click propensity, campaign historical CTR, advertiser quality, publisher quality, ad historical performance, or user-category affinity.

I would compute or score those slower signals in batch and then combine them with live context during online ranking.

So batch inference reduces request-time work while real-time inference preserves freshness.

---

## 13. What are two possible batch CTR architectures?

### Interview answer

One design is entity-level batch scoring. I can precompute user-quality scores, ad-quality scores, campaign CTR priors, or publisher-quality scores and store them for online use.

The second design is candidate-pair scoring, where I precompute CTR scores for selected user-ad pairs.

The pair-scoring approach only works if candidate generation reduces the possible combinations enough to make precomputation practical.

The choice depends on candidate scale, freshness requirements, storage cost, and how much live context the final ranking needs.

---

## 14. Why can't we score every user against every ad?

### Interview answer

Because the cross-product is usually enormous.

If I have 100 million users and 10 million ads, the full user-by-ad space is far too large to score and store.

So I need candidate generation before ML scoring.

I would use eligibility rules, geography, campaign targeting, category or language constraints, retrieval systems, budget constraints, and relevance signals to reduce the candidate set first.

This is a very important production principle: scaling the candidate-generation stage often matters more than optimizing the prediction function itself.

---

## 15. How would you define the scoring entity in a CTR system?

### Interview answer

It depends on the architecture.

The entity may be a user, ad, campaign, publisher, user-category pair, or user-ad pair.

If I am precomputing candidate-level CTR, the entity is usually `(user_id, ad_id)`.

If I am producing reusable priors, I may score users, ads, or campaigns independently.

This decision affects feature joins, partitioning, storage, prediction reuse, and cost, so it should be made before designing the scoring pipeline.

---

## 16. What features would you use for CTR scoring?

### Interview answer

I would use user features, ad features, campaign features, and interaction features.

User features may include historical CTR, recent clicks, category affinity, active days, or device preference.

Ad and campaign features may include historical CTR, impression count, age, advertiser quality, budget utilization, and conversion history.

Interaction features may include user-category affinity, prior user-ad exposure, prior user-ad clicks, or publisher-ad compatibility.

The slower historical features are good batch candidates, while request-specific features stay in the online layer.

---

## 17. How would you handle sparse CTR data?

### Interview answer

CTR data is highly sparse and imbalanced because most impressions are not clicked, and many user-ad pairs have little or no interaction history.

So I would explicitly handle cold-start users, cold-start ads, missing interaction features, and low-support rates.

For example, `1 click / 1 impression` should not have the same confidence as `10,000 clicks / 10,000 impressions` even though both have a CTR of 1.0.

I would include support counts, smoothed priors, missing indicators, and robust fallback features such as category-level or campaign-level statistics.

---

## 18. How would candidate generation work before batch scoring?

### Interview answer

I would first apply cheap filters and retrieval logic before expensive ML scoring.

For example, I may filter ads by user eligibility, geography, language, campaign targeting, budget, publisher constraints, and broad relevance.

After that, a retrieval system may reduce thousands or millions of possible ads to a much smaller candidate set.

Only then would the ensemble model score the remaining candidates.

This keeps compute and output storage manageable.

---

## 19. How would you partition a large user-ad scoring job?

### Interview answer

I would often partition by user ID.

For example:

```text
partition_id = hash(user_id) % N
```

This keeps a user's candidate ads together, which is convenient for later top-K selection.

Each worker loads the same model once, scores candidate batches, and can then locally rank the candidates for each user.

This reduces shuffle and simplifies top-K output generation.

---

## 20. Why keep only top-K candidates after scoring?

### Interview answer

Because storing every scored user-ad pair can be extremely expensive.

If each user has 500 candidates but the online system only needs the best 50, I would score all 500 but persist only the top 50.

The resulting output becomes:

```text
user_id → top ads + scores
```

instead of a massive table of all candidate pairs.

This reduces storage and makes online retrieval much faster.

---

## 21. How would you combine batch CTR scores with real-time signals?

### Interview answer

I would use the batch score as a prior or historical relevance signal, not necessarily as the final decision.

At request time, the online system retrieves the precomputed candidates and scores, then adds live features such as current page, time, query, device, and recent session behavior.

The final ranking can be produced by another online model or a ranking rule.

Conceptually:

```text
final_score = f(batch_prior, live_context, business_constraints)
```

This gives us the cost benefits of precomputation while preserving request-time freshness.

---

## 22. Why is freshness more challenging for CTR than host junk scoring?

### Interview answer

Because CTR can change very quickly.

A new campaign may launch, a user's intent may shift within minutes, ad performance may change, budgets may be exhausted, or a current event may suddenly affect interest.

So a once-daily score may be too stale for the final ranking.

That is why I would normally use batch scoring only for slower-moving priors and reserve the final request-specific ranking for real time.

---

## 23. Would incremental scoring help in CTR?

### Interview answer

Yes, significantly, because the candidate space can be huge.

I would rescore only affected entities when a new ad appears, a user becomes active, campaign features change, historical interactions update, or a score expires.

For example, if a new ad is created, I may score that ad only against eligible users instead of rebuilding all existing user-ad scores.

If one user has new activity, I may refresh only that user's candidate set.

Incremental scoring can save a large amount of compute.

---

## 24. What failure strategies would you use in a CTR batch system?

### Interview answer

I would use bounded fallback because stale CTR scores can degrade quickly.

If one feature group fails, I may use fallback priors or exclude affected campaigns depending on business risk. If the full batch run fails, I may keep the previous candidate list active temporarily, but the maximum fallback age would likely be shorter than in a host-risk system.

I would also monitor campaign coverage and candidate freshness closely because missing or stale candidates can directly reduce ranking quality and revenue.

---

## 25. What would you monitor in a CTR batch system?

### Interview answer

I would monitor technical and business metrics.

Technical metrics include users scored, candidate pairs scored, mean candidates per user, top-K coverage, cold-start rate, feature missingness, campaign coverage, publisher coverage, model version, throughput, runtime, and fallback rate.

Business metrics include CTR, conversion rate, revenue per impression, ad coverage, and candidate diversity.

A technically healthy batch job can still hurt the business if ranking quality degrades, so both levels must be monitored.

---

## Complete 5.15 Answer

If asked:

```text
How would you design batch inference for CTR prediction?
```

Answer:

```text
I would use a hybrid architecture rather than treating CTR as a purely batch problem.

Slow-changing signals such as user long-term behavior, campaign CTR, advertiser quality, ad historical performance, and user-category affinity can be computed or scored in batch. Request-specific signals such as current page, device, current query, time, and recent session behavior should remain in the real-time layer.

Because the full user-by-ad space is too large, I would first perform candidate generation using eligibility rules, campaign targeting, geography, budget constraints, and retrieval logic. The reduced candidate set can then be scored by a batch ensemble model such as LightGBM or XGBoost.

I would often partition by user so that all candidates for a user stay together. After scoring, I may keep only the top-K ads per user and write those candidates to a low-latency store.

At request time, the online system retrieves the precomputed candidates and combines the batch score with live context to produce the final ranking.

I would use incremental rescoring for changed users, new ads, campaign updates, or expired scores, and I would closely monitor candidate coverage, cold-start rate, score distributions, CTR, and revenue impact.
```

---

# Comparing the Two Case Studies — Interview Style

## 26. How would you compare host-junk scoring and CTR scoring architectures?

### Interview answer

The main difference is freshness and candidate scale.

Host junk risk is dominated by slow-changing historical aggregates, so a daily batch snapshot can often be the primary prediction product.

CTR prediction depends much more on request-specific context, so batch scoring is usually only part of the system. Batch computes slow-moving priors or candidate scores, while the final ranking happens online using live context.

So host junk scoring is a strong pure-batch use case, while CTR is typically a hybrid batch-plus-real-time use case.

---

## 27. How do you decide whether to use batch or real-time inference?

### Interview answer

I would decide based on how quickly the target changes, whether request-time context is essential, whether the score can be reused, model cost, and latency requirements.

If the prediction can be computed ahead of time and reused safely for many downstream requests, batch is attractive.

If the prediction strongly depends on context that only exists at request time, real-time inference is necessary.

In many production systems, the right answer is hybrid: batch for slow-changing expensive signals and online inference for fresh request-specific signals.

---

# Complete Batch Inference System Design Answer

If the interviewer asks:

```text
How would you design a production batch inference system for an ensemble model?
```

You can now give this end-to-end answer:

```text
I would start by defining the scoring entity and the eligible population. Then I would compute or retrieve point-in-time-correct features and join them into one model-ready row per entity.

Before scoring, I would validate row count, uniqueness, feature coverage, freshness, ranges, and schema. I would then load an explicit approved model version together with its full inference contract, including feature order, preprocessing, missing-value policy, calibration, class mapping, and threshold policy.

For scale, I would partition the model-ready table and score partitions in parallel. Each worker would load the same model artifact once, score records in batches, and write versioned temporary output.

The prediction output would include entity ID, official score, score meaning, model version, feature version, scoring run ID, timestamps, freshness, and data-quality metadata.

After scoring, I would validate partition completeness, total row count, uniqueness, nulls, model consistency, and prediction distributions. Only then would I atomically publish the new snapshot.

For large populations, I would use incremental scoring to recompute only new, changed, expired, or invalid predictions, while periodically doing full refreshes for consistency.

For reliability, I would use bounded retries, idempotent outputs, quarantine for isolated bad records, fallback to previous successful snapshots, degraded mode where safe, and rollback support.

Finally, I would monitor the entire pipeline: job SLA, input population, feature quality and freshness, model version, throughput, partition runtimes, output distributions, fallback usage, and downstream business metrics.
```

---

# Crisp Final Answer

```text
A production batch inference system is much more than model.predict() on a large table.

It requires:

correct entity selection
→ point-in-time feature joins
→ feature validation
→ explicit model-contract loading
→ distributed or incremental scoring
→ traceable prediction output
→ global validation
→ atomic publishing
→ monitoring and fallback.

For a slow-changing problem like host junk risk, batch can be the primary serving architecture.

For a fast-changing problem like CTR, I would use batch for slow historical priors and candidate precomputation, then combine those with live request context in an online ranking layer.
```

---

# Most Important Memory Lines

```text
The model is only one stage; the real production product is the validated prediction snapshot consumed by downstream systems.
```

```text
Use batch scoring to precompute what changes slowly, and reserve real-time inference for what truly needs to change at request time.
```

```text
Batch inference is not model.predict() on a large dataset; it is an end-to-end production system for generating reliable reusable predictions at scale.
```

---

This completes both the detailed-study and interview-style material for:

```text
Part 5 — Batch Inference System
```

The next major topic is:

```text
Part 6 — Real-Time Inference System
```
