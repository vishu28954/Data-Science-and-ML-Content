# Batch Inference System - Part 5 - Interview Style

# Prediction Output Design and Incremental Scoring

This note belongs with:

```text
Ensemble Learning in Production
Part 5 — Batch Inference System
Part 5.10 — Prediction Output Design
Part 5.11 — Incremental Scoring
```

These notes explain Part 5.10 and Part 5.11 in interview-speaking style.

---

# 5.10 Prediction Output Design — Interview Style

## 1. What is prediction output design in production ML?

### Interview answer

Prediction output design is the process of deciding what information should be stored along with the model prediction so that downstream systems can interpret, trace, and safely use it.

In a notebook, we may only care about:

```text
prediction = 0.91
```

But in production, `0.91` by itself is ambiguous. We need to know what the score means, which model produced it, when it was generated, which features were used, whether the score is still fresh, and whether there were any data-quality issues.

So a production prediction record may contain:

```text
entity_id
score
prediction
risk_bucket
model_version
scoring_run_id
scored_at
feature_timestamp
score_valid_until
data_quality_status
```

The goal is to turn a raw model output into a reliable data product that downstream systems can consume safely.

---

## 2. Why is a score alone not enough?

### Interview answer

A score alone is not enough because the number has no meaning without context.

For example:

```text
0.91
```

could mean:

```text
raw model margin
probability of positive class
probability of negative class
calibrated probability
normalized ranking score
```

So I would always define the score semantics explicitly.

For example:

```text
score_name = junk_score
score_type = calibrated_probability
positive_class = junk
range = [0, 1]
```

This prevents different downstream systems from interpreting the same score differently.

---

## 3. What is the difference between raw score, probability, and calibrated probability?

### Interview answer

The raw score is the direct numerical output from the model before probability conversion.

For example, a boosting model may produce:

```text
raw margin = 2.31
```

That may then be converted to:

```text
probability = 0.91
```

If the model is not well calibrated, we may apply a calibration model and get:

```text
calibrated_probability = 0.84
```

These values are not interchangeable.

In production, I would clearly define which value is the official decision score and store its meaning in the output contract.

---

## 4. Why should we store both score and prediction label?

### Interview answer

I would usually store both the continuous score and the discrete label.

For example:

```text
junk_score = 0.91
prediction = junk
```

The score preserves more information.

If the business threshold changes later from:

```text
0.80
```

to:

```text
0.90
```

we may be able to reapply the new threshold to stored scores without rerunning the model.

If we store only the final label, we lose that flexibility.

---

## 5. What are risk buckets?

### Interview answer

Risk buckets convert continuous model scores into more interpretable categories.

For example:

```text
0.00–0.20 → very_low
0.20–0.50 → low
0.50–0.75 → medium
0.75–0.90 → high
0.90–1.00 → very_high
```

These buckets are useful for review queues, dashboards, prioritization, and business rules.

However, I would treat the risk bucket as a policy layer, not as the raw model output.

The model produces a score, while the business policy converts that score into a bucket.

---

## 6. Why should threshold policy version be stored?

### Interview answer

Because model version and decision policy version can change independently.

For example:

```text
model_version = v17
```

may stay unchanged while the threshold changes from:

```text
0.80
```

to:

```text
0.85
```

So the same model score may produce a different business decision.

That is why I would store something like:

```text
threshold_policy_version = risk_policy_v4
```

This helps explain later why a prediction led to a particular action.

---

## 7. Why should model metadata be stored with predictions?

### Interview answer

Model metadata gives traceability.

If someone asks:

```text
Why did abc.com get a junk score of 0.91?
```

we should be able to identify:

```text
model_name
model_version
feature_pipeline_version
scoring_run_id
```

Without this information, debugging and auditing become difficult.

So I would always store enough model metadata to reproduce or investigate the prediction later.

---

## 8. What is a scoring run ID?

### Interview answer

A scoring run ID identifies one execution of the batch inference pipeline.

For example:

```text
host_junk_2026_09_16_0200
```

All predictions generated in that execution share the same run ID.

This is useful for:

```text
auditing
rollback
debugging
monitoring
reprocessing
```

If we later discover a bad feature pipeline for one run, we can identify exactly which predictions were affected.

---

## 9. What is the difference between scoring timestamp and feature timestamp?

### Interview answer

The scoring timestamp tells us:

```text
when the prediction was generated
```

The feature timestamp tells us:

```text
how recent the data used for the prediction was
```

For example:

```text
feature_timestamp = 02:00
scored_at = 03:15
```

The score was generated at 3:15 using data available up to 2:00.

This distinction is important because a prediction can be newly generated but still rely on stale feature data.

---

## 10. Why should prediction expiry be stored?

### Interview answer

Batch predictions are not timeless.

A score may only be valid for a certain period.

For example:

```text
score_valid_until = 2026-09-17 03:15
```

This lets downstream systems know when the prediction becomes stale.

I may also store:

```text
score_age_hours
is_score_stale
```

This is especially important when downstream systems reuse batch predictions for many hours.

---

## 11. Why should data-quality flags be part of prediction output?

### Interview answer

Because the model may successfully produce a score even when some input features are missing, stale, or low quality.

For example:

```text
junk_score = 0.92
missing_feature_count = 4
data_quality_status = degraded
```

A downstream system may treat that differently from:

```text
junk_score = 0.92
data_quality_status = healthy
```

For example:

```text
high score + healthy data
→ automated action

high score + degraded data
→ human review
```

So data-quality metadata helps downstream systems use model outputs more safely.

---

## 12. Is model probability the same as confidence?

### Interview answer

Not necessarily.

A model may output:

```text
junk_probability = 0.95
```

but the input may have:

```text
missing features
stale data
low support
out-of-distribution values
```

So I would avoid treating model probability as a complete measure of system confidence.

If needed, I would maintain a separate concept such as:

```text
evidence_quality
prediction_quality
```

For example:

```text
junk_probability = 0.95
evidence_quality = low
```

---

## 13. What is a consumer contract?

### Interview answer

A consumer contract defines how downstream systems should interpret the prediction output.

It may specify:

```text
score meaning
score range
positive class
model version
refresh frequency
freshness SLA
expiry rule
fallback behavior
null behavior
threshold semantics
```

For example:

```text
junk_score = calibrated probability of junk
range = [0,1]
refresh = daily
validity = 24 hours
fallback = previous successful score for up to 48 hours
```

The consumer contract prevents downstream teams from interpreting the prediction differently.

---

# Complete 5.10 Answer

Use this when asked:

```text
How would you design prediction output for a production batch inference system?
```

Answer:

```text
I would treat prediction output as a production data product rather than just storing a model score.

Each prediction should preserve the entity ID and include the official decision score, score type, prediction label or risk bucket, model version, threshold policy version, scoring run ID, scoring timestamp, feature timestamp, score expiry, and important data-quality flags.

I would explicitly define whether the score is a raw margin, probability, or calibrated probability, and I would document the positive class so downstream systems interpret it correctly.

I would also separate model output from business policy. For example, the model may produce a calibrated junk probability, while a separate threshold policy converts that probability into a high-risk bucket.

Finally, I would publish the output only after validation and expose it through a clear consumer contract defining score meaning, freshness, expiry, and fallback behavior.
```

---

# 5.11 Incremental Scoring — Interview Style

## 14. What is incremental scoring?

### Interview answer

Incremental scoring means rescoring only the entities whose predictions may have changed instead of rescoring the entire population every time.

For example, if we have:

```text
20 million hosts
```

but only:

```text
2 million hosts
```

had meaningful feature changes today, then we may score only those 2 million and reuse the still-valid predictions for the remaining 18 million.

This reduces compute, IO, runtime, and cost.

---

## 15. Why do we need incremental scoring?

### Interview answer

Full rescoring is simple, but it can be expensive at large scale.

Many entities may not change between batch runs.

For example, a host may have:

```text
no new crawl activity
no new click signals
no metadata changes
```

If the model and features are unchanged, rescoring that host may produce exactly the same prediction.

Incremental scoring tries to avoid unnecessary work by identifying only entities whose previous scores are no longer valid.

---

## 16. What events can trigger rescoring?

### Interview answer

Common triggers include:

```text
new entity
feature values changed
score expired
late-arriving data changed features
previous score was invalid
data quality recovered
model version changed
feature pipeline version changed
```

The important point is that rescoring should happen because something relevant changed, not simply because another batch run started.

---

## 17. How do you detect changed features?

### Interview answer

One approach is to compare the current feature vector with the previous feature vector.

For example:

```text
yesterday:
error_ratio = 0.10

today:
error_ratio = 0.60
```

That entity should likely be rescored.

Another approach is feature fingerprinting.

We compute a hash over the feature vector:

```text
hash(error_ratio, crawl_ratio, dsat_ratio, static_rank, ...)
```

If the hash changes, the features changed.

This can make change detection more efficient.

---

## 18. Can we ignore very small feature changes?

### Interview answer

Sometimes, but we have to be careful.

For example:

```text
0.3001 → 0.3002
```

looks like a tiny change.

But a tree may have a split at:

```text
0.30015
```

so that tiny change sends the row down a different branch and can change the prediction.

Therefore, small numerical changes do not always imply small prediction changes for tree-based models.

That is one reason incremental scoring adds complexity compared with full rescoring.

---

## 19. Why can score expiry trigger rescoring?

### Interview answer

Even if features have not changed, a business or freshness policy may require scores to be refreshed periodically.

For example:

```text
score_valid_until = 24 hours
```

When the score expires, the entity becomes a candidate for rescoring.

So incremental candidate generation should consider both feature changes and prediction freshness.

---

## 20. What happens when model version changes?

### Interview answer

A model version change often requires broad or full rescoring.

For example:

```text
old score = model_v17(features)
new score = model_v18(features)
```

Even if the features are unchanged, the outputs may differ.

So if the production model changes from v17 to v18, I would usually perform a full refresh unless the system explicitly supports mixed model versions.

This prevents the current prediction table from containing some scores from v17 and others from v18.

---

## 21. What happens when feature pipeline version changes?

### Interview answer

A feature pipeline change may also invalidate previous predictions.

For example, the feature name:

```text
error_ratio_10d
```

may stay the same, but its computation logic may change.

Then an old prediction was generated using a different feature meaning.

So if a feature definition changes materially, I would rescore the affected population and record the new feature pipeline version.

---

## 22. How do you generate incremental scoring candidates?

### Interview answer

A typical candidate set may be:

```text
new_entities
UNION
changed_feature_entities
UNION
expired_score_entities
UNION
invalid_previous_score_entities
UNION
late_data_affected_entities
```

After deduplication, this becomes the set of entities that should be rescored.

Then the pipeline runs:

```text
incremental candidates
→ feature joins
→ model scoring
→ new scores
→ merge with reusable old scores
```

---

## 23. How do you merge new and old scores?

### Interview answer

Suppose yesterday we had:

```text
A → 0.20
B → 0.60
C → 0.80
D → 0.10
```

Today only:

```text
B
D
E
```

need scoring.

New scores:

```text
B → 0.75
D → 0.15
E → 0.90
```

The current snapshot becomes:

```text
A → 0.20 reused
B → 0.75 rescored
C → 0.80 reused
D → 0.15 rescored
E → 0.90 new
```

I would normally perform an upsert or deterministic merge rather than a blind append.

---

## 24. What is the difference between incremental features and incremental scoring?

### Interview answer

Incremental feature computation means:

```text
recompute only changed features
```

Incremental scoring means:

```text
recompute only affected predictions
```

They are separate optimizations.

A system may use:

```text
incremental features + full scoring
```

or:

```text
full features + incremental scoring
```

or both.

So I would treat them as separate architectural decisions.

---

## 25. How do late-arriving events affect incremental scoring?

### Interview answer

Late-arriving events can change historical feature values after the original scoring job has finished.

For example:

```text
late click logs arrive
→ dsat_ratio changes
→ previous host score becomes stale
→ host should be rescored
```

So incremental candidate generation should include entities affected by late-arriving data.

Otherwise, the system may keep reusing a prediction based on incomplete features.

---

## 26. How do you handle inactive or deleted entities?

### Interview answer

Incremental systems can accidentally keep old scores forever if they only merge new predictions.

So the system must also handle:

```text
deleted entities
inactive entities
expired entities
```

Possible approaches include removing them from the current snapshot or storing:

```text
entity_status = inactive
```

Without this, stale entities accumulate over time.

---

## 27. Why is idempotency important in incremental scoring?

### Interview answer

Because incremental jobs may retry.

If a retry blindly appends predictions, duplicate rows can appear.

So I would design the merge as an upsert using a deterministic key such as:

```text
entity_id
```

or:

```text
entity_id + scoring_snapshot
```

Then rerunning the same scoring task produces the same final state.

This makes retries safe.

---

## 28. Why are mixed model versions dangerous in incremental scoring?

### Interview answer

Suppose:

```text
A → model_v17
B → model_v18
C → model_v17
```

Now one current-state table contains scores from different models.

Those scores may not be directly comparable.

That is why a major model change often triggers full rescoring.

At minimum, model version must be stored per prediction so mixed-version states can be detected.

---

## 29. Why do systems still perform periodic full refreshes?

### Interview answer

Incremental systems are efficient, but they are more complex.

Over time, errors can accumulate because of:

```text
missed change events
dependency bugs
late data
incorrect merge logic
stale entities
```

So a practical pattern is:

```text
daily incremental scoring
weekly full refresh
```

The incremental runs save cost, while the periodic full rebuild restores consistency and catches missed updates.

---

## 30. What is the tradeoff between full and incremental scoring?

### Interview answer

Full scoring is:

```text
simpler
easier to validate
consistent
more expensive
```

Incremental scoring is:

```text
cheaper
faster
more scalable
more complex
harder to guarantee correctness
```

So I would choose incremental scoring only when the compute or latency savings justify the extra system complexity.

For 100,000 entities, full scoring may be perfectly reasonable.

For 1 billion entities, incremental scoring can be extremely valuable.

---

# Complete 5.11 Answer

Use this when asked:

```text
How would you design incremental scoring in a batch inference system?
```

Answer:

```text
Incremental scoring means reusing previous predictions that are still valid and rescoring only entities whose predictions may have changed.

I would first build an incremental candidate set from new entities, entities whose features changed, entities with expired or invalid scores, entities affected by late-arriving data, and any entities impacted by pipeline changes.

For feature changes, I could compare feature vectors directly or use feature fingerprints. The selected candidates would go through the normal feature validation and model scoring pipeline.

The new predictions would then be upserted into the previous prediction snapshot, while unchanged entities would keep their valid scores.

I would also store metadata such as last_scored_at, model_version, and score_source_run so that reused and rescored predictions remain traceable.

If the model version or feature definition changes materially, I would usually perform a full refresh to avoid mixing incompatible predictions.

Finally, I would periodically run a full rebuild even in an incremental system because incremental pipelines can accumulate errors over time.
```

---

# Crisp Combined Answer

```text
For Prediction Output Design, I would treat the prediction as a data product rather than just a score. The output should include entity ID, official score, score meaning, prediction or risk bucket, model version, policy version, timestamps, freshness, and data-quality metadata so downstream systems can interpret and trace it correctly.

For Incremental Scoring, I would avoid rescoring the full population when only a subset has changed. I would generate candidates from new entities, changed features, expired scores, invalid predictions, and late-arriving data, rescore only those entities, and merge the new scores with previous valid scores. I would still trigger full refreshes when model or feature versions change and periodically rebuild the full snapshot to maintain consistency.
```

---

# Most Important Memory Lines

```text
A production prediction is a data product, not just a number.
```

```text
Incremental scoring is about reusing valid predictions without sacrificing correctness.
```
