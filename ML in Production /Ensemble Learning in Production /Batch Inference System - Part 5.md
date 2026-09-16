# Batch Inference System - Part 5

# Ensemble Learning in Production

# Part 5.10 — Prediction Output Design

# Part 5.11 — Incremental Scoring

This note continues:

```text
Part 5 — Batch Inference System
```

Previously covered:

```text
5.6 Input Data Preparation
5.7 Feature Joins
5.8 Loading the Ensemble Model
5.9 Distributed Scoring
```

Now we study:

```text
5.10 Prediction Output Design
5.11 Incremental Scoring
```

At this point, the model has already produced predictions.

The next questions are:

```text
What exactly should we save as the prediction output?

How should downstream systems consume that output?

Do we really need to rescore every entity every day?
```

That is what 5.10 and 5.11 solve.

---

# 5.10 Prediction Output Design

## 5.10.1 Why Prediction Output Design Matters

A beginner may think the output of a model is simply:

```text
0.91
```

But in production, that is almost useless by itself.

What does `0.91` mean?

It could mean:

```text
91% probability of junk
raw XGBoost margin
calibrated probability
probability of not-junk
ranking score
normalized risk score
```

Without context, downstream systems cannot safely use it.

So production output design is about creating a prediction record with enough context to understand, validate, trace, and act on the prediction.

A proper production output might look like:

```text
host_id = abc.com
junk_score = 0.91
risk_bucket = high
model_version = v17
scoring_run_id = 2026_09_16_daily
scored_at = 2026-09-16 03:15
feature_timestamp = 2026-09-16 02:00
score_valid_until = 2026-09-17 03:15
data_quality_flag = healthy
```

That is much more useful than:

```text
abc.com → 0.91
```

---

## 5.10.2 Start With the Entity Identifier

Every prediction must remain connected to the entity that was scored.

For host junk detection:

```text
host_id
```

For CTR prediction:

```text
user_id
ad_id
campaign_id
```

For churn prediction:

```text
customer_id
```

For fraud:

```text
transaction_id
```

The model itself may not use the ID as a feature.

But the prediction output absolutely needs it.

Otherwise:

```text
score = 0.91
```

cannot be mapped back to anything.

This connects directly to Part 5.6, where we discussed preserving entity IDs throughout the scoring pipeline.

A production principle:

```text
Never separate a prediction from the entity identity required to interpret it.
```

---

## 5.10.3 Raw Score vs Probability vs Calibrated Probability

This is one of the most important output-design concepts.

A model may produce several different types of output.

For example, an XGBoost classifier may internally produce a raw margin:

```text
raw_score = 2.31
```

After sigmoid:

```text
probability = 0.91
```

After calibration:

```text
calibrated_probability = 0.84
```

These are not interchangeable.

A good production output may therefore contain:

```text
raw_score
model_probability
calibrated_probability
```

But downstream systems should know which one is the official decision score.

Example:

```text
decision_score = calibrated_probability
```

Otherwise one team may use:

```text
0.91
```

while another uses:

```text
0.84
```

and both think they are using “the model score.”

That creates inconsistency.

---

## 5.10.4 Why Calibration Matters to Output Meaning

Suppose the model predicts:

```text
0.90
```

If the model is perfectly calibrated, then among many examples receiving approximately `0.90`, roughly 90% should belong to the positive class.

But many ensemble models can produce probabilities that are not perfectly calibrated.

So the production system may apply a calibration layer.

Then:

```text
raw model probability = 0.90
calibrated probability = 0.77
```

If your downstream business rule is:

```text
auto-block if probability > 0.80
```

then using `0.90` versus `0.77` changes the action completely.

This is why the output should explicitly define:

```text
score_type
```

Example:

```text
score_type = calibrated_probability
```

The important idea is:

```text
The prediction output must tell consumers what the score actually means.
```

---

## 5.10.5 Prediction Label

Some systems also want a discrete prediction.

Example:

```text
junk_score = 0.91
prediction = junk
```

For binary classification:

```text
score >= threshold
    → positive

score < threshold
    → negative
```

Example:

```text
threshold = 0.80

0.91 → junk
0.32 → not_junk
```

But storing only:

```text
prediction = junk
```

is usually less useful than storing both:

```text
junk_score = 0.91
prediction = junk
```

Why?

Because the continuous score preserves more information.

Suppose later the threshold changes from:

```text
0.80
```

to:

```text
0.90
```

If you saved only labels, you may need to rescore everything.

If you saved probabilities, you may simply reapply the new threshold.

---

## 5.10.6 Risk Buckets

Sometimes downstream systems do not want a raw probability.

They want understandable categories.

Example:

```text
0.00 – 0.20 → very_low
0.20 – 0.50 → low
0.50 – 0.75 → medium
0.75 – 0.90 → high
0.90 – 1.00 → very_high
```

Then output becomes:

```text
host_id = abc.com
junk_score = 0.91
risk_bucket = very_high
```

Risk buckets are useful for:

```text
review queues
dashboards
policy rules
manual triage
alert prioritization
business workflows
```

But remember:

```text
risk bucket is derived from score + policy
```

It is not the raw model output.

So the system should distinguish:

```text
model output
```

from:

```text
business interpretation
```

---

## 5.10.7 Threshold Policy Version

Suppose today:

```text
score >= 0.80 → high_risk
```

Next month:

```text
score >= 0.85 → high_risk
```

The model did not change.

But the decision policy did.

If the output only stores:

```text
risk_bucket = high
```

you may not know which threshold policy produced it.

So production output should include something like:

```text
threshold_policy_version = risk_policy_v4
```

Then you can distinguish:

```text
model_version = v17
threshold_policy_version = v4
```

This is important because model behavior and business policy are two different things.

---

## 5.10.8 Model Metadata in Prediction Output

Every prediction should usually be traceable to the model that created it.

Useful metadata:

```text
model_name
model_version
model_checksum
```

Example:

```text
model_name = host_junk_xgboost
model_version = 17
```

Why is this useful?

Imagine:

```text
abc.com score = 0.91
```

A week later, someone asks:

```text
Why was abc.com blocked?
```

You need to know:

```text
which model produced the score
which feature version was used
which scoring run produced it
```

Without that, debugging becomes difficult.

---

## 5.10.9 Scoring Run ID

A scoring run ID identifies one execution of the batch inference pipeline.

Example:

```text
scoring_run_id = host_junk_2026_09_16_0200
```

All rows produced in that execution share the same run ID.

Why is this useful?

Suppose:

```text
20 million predictions
```

were generated.

Later, you discover an upstream feature bug.

You can identify:

```text
all predictions from scoring_run_id X
```

and invalidate or roll them back.

The run ID is useful for:

```text
auditing
debugging
rollback
monitoring
lineage
reprocessing
```

---

## 5.10.10 Scoring Timestamp

Every prediction should indicate when it was produced.

Example:

```text
scored_at = 2026-09-16 03:15
```

This tells consumers:

```text
when did this prediction become available?
```

This is different from:

```text
feature_timestamp
```

The feature timestamp might be:

```text
2026-09-16 02:00
```

meaning the feature data reflects state up to 2 AM.

The score may be produced at:

```text
3:15 AM
```

These timestamps have different meanings.

---

## 5.10.11 Feature Timestamp

A very useful output field is:

```text
feature_timestamp
```

This means:

```text
How recent was the data used to generate the prediction?
```

Example:

```text
scored_at = 03:15
feature_timestamp = 02:00
```

Fine.

But imagine:

```text
scored_at = 03:15
feature_timestamp = three days ago
```

The prediction is technically new, but the underlying information is stale.

That is an important distinction.

A prediction can be:

```text
freshly computed
```

while using:

```text
stale features
```

So:

```text
prediction freshness ≠ feature freshness
```

---

## 5.10.12 Score Validity and Expiry

Batch predictions are not timeless.

A prediction produced today may become stale tomorrow.

So output should define:

```text
score_valid_until
```

Example:

```text
scored_at = 2026-09-16 03:15
score_valid_until = 2026-09-17 03:15
```

Then downstream consumers know:

```text
After this time, do not treat this score as current.
```

Another representation could be:

```text
score_age_hours
is_score_stale
```

This is especially important in batch systems because predictions are reused over time.

---

## 5.10.13 Data Quality Flags

Suppose a prediction was produced, but some input features were missing.

Example:

```text
missing_feature_count = 3
```

The model may still score successfully.

Should consumers treat that score exactly the same as one computed with complete data?

Maybe not.

So output can include:

```text
missing_feature_count
is_low_evidence
has_stale_features
data_quality_status
```

Example:

```text
data_quality_status = degraded
```

or:

```text
data_quality_flags = [missing_dsat, stale_crawl]
```

This lets downstream systems choose different behavior.

Example:

```text
high score + healthy data
    → automated action

high score + degraded data
    → manual review
```

This is a powerful design pattern.

---

## 5.10.14 Confidence Is Not Always the Same as Probability

Be careful with this term.

If:

```text
junk_probability = 0.95
```

that does not automatically mean:

```text
95% confidence that the system is reliable
```

The model may have:

```text
missing features
low-support ratios
out-of-distribution input
stale signals
```

So you may maintain a separate concept like:

```text
prediction_quality
```

or:

```text
evidence_quality
```

Example:

```text
junk_probability = 0.95
evidence_quality = low
```

That tells us:

```text
model strongly predicts junk
but input evidence is weak
```

This distinction becomes useful in human-review systems.

---

## 5.10.15 Explanation Metadata

For some applications, you may also store explanation information.

Examples:

```text
top_features
SHAP values
top contributing signals
reason codes
```

Example:

```text
top_reason_1 = high_error_ratio
top_reason_2 = high_dsat_ratio
top_reason_3 = spam_signal_count
```

This can help:

```text
manual reviewers
debugging
customer support
model audits
policy teams
```

But storing full SHAP vectors for millions of predictions can be expensive.

So sometimes production systems store:

```text
top 3 reason codes
```

rather than full explanations.

---

## 5.10.16 Output Schema Example

A mature output schema might look like:

```text
host_id
junk_score
raw_score
calibrated_score
prediction
risk_bucket

model_name
model_version
threshold_policy_version

scoring_run_id
scored_at
feature_timestamp
score_valid_until

missing_feature_count
is_low_evidence
has_stale_features
data_quality_status

top_reason_1
top_reason_2
```

Not every project needs all of these.

The key principle is:

```text
Output should contain enough information
for downstream systems to interpret and trust the score.
```

---

## 5.10.17 Output Format

Predictions may be written to:

```text
Parquet files
database tables
data warehouse tables
key-value stores
feature stores
object storage
```

The format depends on consumption pattern.

Example:

For analytics:

```text
Parquet / warehouse table
```

For low-latency lookup later:

```text
Redis / online feature store / key-value store
```

For dashboards:

```text
warehouse table
```

For review queue:

```text
ranked table or queue
```

The general design lesson is:

```text
The prediction output should preserve the mapping between input entity and inference result.
```

---

## 5.10.18 Consumer Contract

A downstream consumer should know exactly what the score means.

A good consumer contract defines:

```text
score name
score meaning
range
model version
update frequency
freshness SLA
expiry rule
fallback behavior
threshold semantics
null behavior
```

Example:

```text
Field: junk_score
Meaning: calibrated probability that host is junk
Range: [0, 1]
Refresh: daily
Validity: 24 hours
Positive class: junk
Fallback if missing: use last successful score up to 48 hours
```

This prevents different systems from interpreting the same output differently.

---

## 5.10.19 Publishing Output Safely

Remember Part 5.9.

Workers should not publish final scores directly while the run is incomplete.

Safe flow:

```text
worker outputs
    ↓
temporary prediction table
    ↓
validation
    ↓
atomic publish
    ↓
downstream consumption
```

The broader production principle:

```text
Only validated predictions should become consumable predictions.
```

---

# 5.10 Final Mental Model

Prediction output is not:

```text
entity → score
```

It is closer to:

```text
entity
+
score
+
score meaning
+
model identity
+
time context
+
data quality
+
decision context
```

Memory line:

```text
A production prediction is a data product, not just a number.
```

---

# 5.11 Incremental Scoring

Now we ask a different question.

Suppose yesterday we scored:

```text
20 million hosts
```

Today, do we really need to score all 20 million again?

Maybe only:

```text
2 million hosts
```

actually changed.

If so, rescoring everything wastes:

```text
compute
IO
time
money
```

This motivates incremental scoring.

---

## 5.11.1 What Is Incremental Scoring?

Incremental scoring means:

```text
score only entities whose prediction may need to change
```

instead of rescoring the full population every time.

Full batch scoring:

```text
20 million hosts
↓
score all 20 million daily
```

Incremental scoring:

```text
20 million hosts
↓
identify changed / new / expired entities
↓
score only 2 million
↓
merge with previous valid scores
```

This can dramatically reduce compute.

---

## 5.11.2 Why Incremental Scoring Exists

Suppose a host risk model runs every day.

But many hosts may have:

```text
no new crawl events
no new clicks
no new spam signals
no metadata changes
```

If their features are unchanged, scoring them again with the same model may produce the same score.

So instead of:

```text
recompute everything
```

we can ask:

```text
Which entities actually need rescoring?
```

This is similar to incremental processing in data engineering.

Instead of rebuilding everything:

```text
process only changed data
```

---

## 5.11.3 What Can Trigger Rescoring?

An entity may need rescoring when:

```text
new entity appears
feature values change
feature freshness expires
model version changes
feature pipeline changes
business policy requires refresh
previous score is invalid
data quality recovers
```

---

## 5.11.4 New Entities

Simple case.

Yesterday:

```text
host abc.com existed
host xyz.net existed
```

Today:

```text
newsite.com appears
```

`newsite.com` has never been scored.

So:

```text
new entity → score it
```

This is a clear incremental candidate.

---

## 5.11.5 Changed Features

Suppose:

```text
abc.com yesterday:
error_ratio = 0.10
dsat_ratio = 0.05

today:
error_ratio = 0.60
dsat_ratio = 0.25
```

Its risk likely changed.

So it should be rescored.

But:

```text
xyz.net:
all features unchanged
```

Maybe no need to rescore.

Conceptually:

```text
current_feature_vector
        vs
previous_feature_vector
```

If changed:

```text
rescore
```

If unchanged:

```text
reuse previous score
```

---

## 5.11.6 Feature Fingerprinting

Comparing every feature individually can be cumbersome.

A useful technique is:

```text
feature fingerprint
```

For example:

```text
hash(
    error_ratio,
    crawl_ratio,
    dsat_ratio,
    static_rank,
    ...
)
```

Yesterday:

```text
feature_hash = A72F...
```

Today:

```text
feature_hash = B89C...
```

Different hash:

```text
features changed
```

Same hash:

```text
features unchanged
```

Then you can decide whether to rescore.

This technique can simplify incremental detection.

---

## 5.11.7 But Tiny Changes May Not Matter

Suppose:

```text
error_ratio yesterday = 0.3001
error_ratio today = 0.3002
```

Technically changed.

But should we rescore?

Maybe.

Maybe not.

It depends on system requirements.

You may define meaningful change thresholds:

```text
absolute delta
relative delta
bucket change
feature-specific tolerance
```

Example:

```text
rescore only if:
abs(new_error_ratio - old_error_ratio) > 0.01
```

But be cautious.

Tree models can be sensitive near split thresholds.

Suppose a tree split is:

```text
error_ratio > 0.30015
```

Then:

```text
0.3001
```

and:

```text
0.3002
```

go down different branches.

So “small feature change” does not always mean “small prediction change.”

This is one reason full rescoring is operationally simpler.

Incremental scoring trades simplicity for efficiency.

---

## 5.11.8 Score Expiry Can Trigger Rescoring

Even if features appear unchanged, scores may have an expiry policy.

Example:

```text
score_valid_until = 24 hours
```

Then every day the score must be refreshed.

Why?

Because downstream consumers may require:

```text
fresh prediction
```

So incremental scoring may include:

```text
entities whose score has expired
```

---

## 5.11.9 Model Version Change Forces Broad Rescoring

Suppose yesterday:

```text
model_v17
```

Today production switches to:

```text
model_v18
```

Even if features are unchanged, previous scores may no longer be comparable.

Then:

```text
all entities may need rescoring
```

because:

```text
old score = model_v17(features)
new score = model_v18(features)
```

They are not necessarily equal.

So incremental scoring logic must consider:

```text
model version
```

not only feature changes.

This is very important.

---

## 5.11.10 Feature Pipeline Version Change

Similarly, suppose feature definition changes.

Old:

```text
error_ratio_10d
```

computed from one source.

New:

```text
error_ratio_10d
```

computed from corrected logic.

Even though the feature name is unchanged, its meaning changed.

Then old predictions may be invalid.

So:

```text
feature_pipeline_version changes
→ likely rescore affected population
```

This connects back to Part 3 Feature Pipeline Design.

---

## 5.11.11 Candidate Generation for Incremental Scoring

Incremental scoring needs its own candidate-generation step.

Example:

```text
incremental_candidates =
    new_entities
    UNION
    changed_feature_entities
    UNION
    expired_score_entities
    UNION
    invalid_previous_score_entities
```

Then:

```text
incremental_candidates
        ↓
feature joins
        ↓
model scoring
        ↓
new scores
        ↓
merge with previous valid scores
```

This is the basic architecture.

---

## 5.11.12 Merge New Scores With Old Scores

Suppose yesterday's prediction table:

```text
host_id      score
A            0.20
B            0.60
C            0.80
D            0.10
```

Today only:

```text
B
D
E
```

need scoring.

New outputs:

```text
B → 0.75
D → 0.15
E → 0.90
```

Final table becomes:

```text
A → 0.20  reused
B → 0.75  rescored
C → 0.80  reused
D → 0.15  rescored
E → 0.90  new
```

This is incremental merge.

The final table still presents a complete view.

Downstream systems do not necessarily need to know that some rows were reused and others recomputed.

But it is useful to store:

```text
last_scored_at
score_source_run
was_reused
```

for traceability.

---

## 5.11.13 Incremental Scoring vs Incremental Feature Computation

These are related but different.

Incremental feature computation means:

```text
only recompute changed features
```

Incremental scoring means:

```text
only recompute predictions for affected entities
```

You can have:

```text
full feature recomputation
+
incremental scoring
```

or:

```text
incremental feature recomputation
+
full scoring
```

or:

```text
incremental features
+
incremental scoring
```

They are separate optimization decisions.

---

## 5.11.14 Dependency Tracking

Suppose:

```text
host_error_ratio
```

changes.

Which predictions depend on it?

If the model uses that feature:

```text
host score should be recomputed
```

Suppose a publisher-level feature changes:

```text
publisher_quality_score
```

and millions of URLs depend on that publisher.

Then one upstream change may require rescoring many entities.

This is why incremental pipelines benefit from dependency awareness.

Conceptually:

```text
changed source entity
    ↓
affected feature entities
    ↓
affected prediction entities
```

This can become complex.

---

## 5.11.15 Late-Arriving Data

Suppose yesterday's job scored a host using incomplete logs.

Later, additional events arrive.

Now historical features for that host change.

Should we rescore?

Often yes.

Incremental scoring candidates may include:

```text
entities affected by late-arriving events
```

Example:

```text
late click logs arrive
    ↓
dsat_ratio changes
    ↓
host score becomes stale
    ↓
rescore host
```

This is an important production case.

---

## 5.11.16 Deleted or Inactive Entities

Incremental systems must also handle entities that disappear.

Suppose:

```text
host abc.com
```

is no longer active.

If you keep merging new scores with yesterday's table, its old score may remain forever.

So incremental output maintenance must support:

```text
deletion
expiration
inactive-state marking
```

Example:

```text
entity_status = inactive
```

or remove it from the active snapshot.

Otherwise you accumulate stale entities.

---

## 5.11.17 Idempotency in Incremental Scoring

Incremental jobs may retry.

Suppose:

```text
B
D
E
```

were rescored.

If the job retries, you should not create duplicate rows.

The merge should be deterministic.

Example key:

```text
entity_id + scoring_snapshot
```

or:

```text
entity_id
```

for current-state tables.

The operation should behave like:

```text
upsert
```

rather than:

```text
blind append
```

---

## 5.11.18 Incremental Scoring Can Become Inconsistent

This is an important failure mode.

Suppose:

```text
A score was generated by model_v17
B score was regenerated by model_v18
C score remains from model_v17
```

Now one table contains mixed-model predictions.

That may be unacceptable.

So when model version changes, you must decide:

```text
full refresh
```

or explicitly support mixed versions.

For most risk-scoring systems, a full refresh on major model change is cleaner.

This is why output metadata should always include:

```text
model_version
```

---

## 5.11.19 Periodic Full Refresh

Even with incremental scoring, many systems periodically perform a full batch refresh.

Example:

```text
daily → incremental
weekly → full refresh
```

Why?

Because incremental pipelines can accumulate errors.

Examples:

```text
missed change event
dependency bug
stale entity
incorrect merge
late data not propagated
```

A full refresh resets the system to a clean state.

This is similar to:

```text
incremental updates for efficiency
+
periodic rebuild for correctness
```

That is often a very practical design.

---

## 5.11.20 Incremental Scoring Tradeoff

Full scoring:

```text
simple
easy to reason about
consistent snapshot
expensive
```

Incremental scoring:

```text
cheaper
faster
less compute
more complex
harder to guarantee correctness
```

So the decision is not:

```text
incremental is always better
```

It is:

```text
Does saved compute justify added system complexity?
```

For:

```text
100,000 entities
```

maybe full scoring is easier.

For:

```text
1 billion entities
```

incremental scoring may be very valuable.

---

## 5.11.21 Host Junk Scoring Example

Suppose:

```text
20 million active hosts
```

Yesterday all were scored.

Today only:

```text
1.2M hosts had new crawl activity
300K hosts had new DSAT signals
100K hosts are newly discovered
200K previous scores expired or were degraded
```

After deduplication:

```text
incremental candidate set = 1.55M hosts
```

Instead of scoring:

```text
20M
```

you score:

```text
1.55M
```

Then:

```text
new scores
    +
previous still-valid scores
    ↓
complete current snapshot
```

That can save substantial compute.

But if:

```text
model version changes from v17 to v18
```

you may choose:

```text
full 20M rescore
```

to ensure consistency.

---

## 5.11.22 A Robust Incremental Architecture

A mature pipeline could look like:

```text
Previous Prediction Snapshot
        +
Current Feature Snapshot
        +
Change Events
        +
New Entities
        +
Expired Scores
        ↓
Incremental Candidate Generator
        ↓
Feature Validation
        ↓
Model Scoring
        ↓
New Prediction Delta
        ↓
Upsert / Merge
        ↓
Current Complete Prediction Snapshot
        ↓
Global Validation
        ↓
Publish
```

This is much more sophisticated than:

```text
model.predict(all_rows)
```

But it can be worth it at large scale.

---

# 5.11 Final Mental Model

Incremental scoring is not just:

```text
score fewer rows
```

It is:

```text
correctly determine which predictions have become invalid
and recompute only those predictions.
```

That word **correctly** is the hard part.

Memory line:

```text
Incremental scoring saves compute by reusing valid predictions, but correctness depends on accurately identifying every entity whose prediction may have changed.
```

---

# Final Summary

For **5.10 Prediction Output Design**, think:

```text
The model produces a number.
Production must turn that number into a trustworthy data product.
```

That means output should carry:

```text
entity identity
score
score meaning
prediction/risk bucket
model version
policy version
timestamps
freshness
data-quality metadata
lineage
```

For **5.11 Incremental Scoring**, think:

```text
Do not rescore an entity merely because the batch job runs again.
Rescore it because something relevant changed or its old score is no longer valid.
```

The core flow becomes:

```text
detect change
→ identify affected entities
→ rescore
→ merge with reusable scores
→ validate
→ publish complete snapshot
```

The two memory lines to keep:

```text
A production prediction is a data product, not just a number.
```

and:

```text
Incremental scoring is about reusing valid predictions without sacrificing correctness.
```

---

Next sections:

```text
5.12 Failure Handling
5.13 Monitoring Batch Inference Jobs
5.14 Case Study: Daily Host Junk Risk Scoring
5.15 Case Study: Daily User/Ad CTR Scoring
```
