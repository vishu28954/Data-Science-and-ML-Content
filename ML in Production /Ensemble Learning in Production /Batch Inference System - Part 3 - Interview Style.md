# Batch Inference System - Part 3 - Interview Style

# Input Data Preparation and Feature Joins — Interview Style

This note belongs with:

```text
Ensemble Learning in Production
Part 5 — Batch Inference System
Part 5.6 — Input Data Preparation
Part 5.7 — Feature Joins
```

These notes explain Part 5.6 and Part 5.7 in an interview-speaking style.

---

# 5.6 Input Data Preparation — Interview Style

## 1. What is input data preparation in batch inference?

In batch inference, input data preparation means creating the clean and valid dataset that will be given to the model for scoring.

In a notebook, we usually already have a clean dataframe like `X_test`, and we directly call:

```python
model.predict(X_test)
```

But in production, we do not start with a clean dataframe. We start with raw logs, event tables, snapshots, metadata tables, click logs, crawl logs, impression logs, labels, and many upstream systems.

So before calling the model, we need to prepare the input carefully.

The goal is:

```text
Create one clean, valid, scoreable row per entity.
```

For example, if we are doing host junk-risk scoring, then one row may represent:

```text
host_id = abc.com
scoring_date = 2026-09-10
```

If we are doing customer churn prediction, one row may represent:

```text
customer_id = C123
scoring_date = 2026-09-10
```

So input data preparation starts with deciding:

```text
Who should be scored?
```

This is called **entity selection** or **candidate generation**.

### Interview answer

Input data preparation in batch inference is the process of creating a clean, valid, model-ready dataset before scoring. In production, we do not usually receive a clean dataframe directly. We receive raw logs, snapshots, event tables, and metadata from many upstream systems. So the pipeline must decide which entities should be scored, remove invalid or duplicate records, assign scoring timestamps, validate schema, and prepare the exact feature format expected by the model.

For example, in a host junk-scoring system, the input preparation step decides whether to score all hosts, only active hosts, only recently crawled hosts, or only hosts with enough historical data. The output of this step should be one clean row per host per scoring time.

The most important point is that model inference should not begin until the input data satisfies the expected contract.

---

## 2. Why is input data preparation important?

Input preparation is important because a model can only make reliable predictions if the input is reliable.

A production model does not understand the business meaning of the data. It only receives numbers.

For example:

```text
host_error_ratio_10d = 0.12
```

The model does not know whether this was computed correctly or whether the upstream pipeline failed. It simply uses the number.

If the input data is wrong, the model may still return a prediction, but the prediction will be meaningless.

This is especially dangerous in tree ensembles like Random Forest, XGBoost, and LightGBM because they can produce confident scores even when the input is incorrect.

For example, suppose the model expects:

```text
host_error_ratio_10d = 0.12
```

But because of a pipeline bug, it receives:

```text
host_error_ratio_10d = 12
```

The model may treat this as an extreme value and follow a completely different path in the trees.

So a successful `model.predict()` call does not guarantee a correct prediction.

### Interview answer

Input data preparation is critical because most production ML failures are data failures before they are model failures. If the input population is wrong, if IDs are duplicated, if timestamps are incorrect, if feature values are stale, or if the schema does not match the model expectation, then the model may still produce predictions, but those predictions may be unreliable.

This is especially important for ensemble models because tree-based models can learn sharp threshold-based behavior. A wrong value or wrong feature order can send a sample down a completely different path in the trees. So before inference, I would validate row counts, duplicate IDs, schema, data types, feature ranges, missing rates, and scoring timestamps.

---

## 3. What is entity selection?

Entity selection means deciding which entities should be included in the batch scoring job.

For example, in a host-level junk model, the possible universe may be:

```text
all known hosts
```

But the actual scoring population may be:

```text
hosts seen in the last 30 days
hosts with valid host IDs
hosts with enough crawl activity
hosts not already permanently blocked
hosts not in an allowlist
```

This matters because scoring everything may be expensive and noisy.

Suppose:

```text
all known hosts = 1 billion
active hosts = 80 million
hosts with enough useful signals = 20 million
```

Scoring all 1 billion hosts may waste compute and storage. But scoring only 20 million may miss long-tail risky hosts.

So entity selection is a tradeoff between:

```text
coverage
cost
noise
actionability
```

### Interview answer

Entity selection is the step where we decide which entities should be scored in the batch run. For example, in a host junk-scoring system, we may choose to score only hosts that were active in the last 30 days, have valid IDs, have enough crawl data, and are not already permanently blocked.

This is important because the scoring population should match the population on which the model was trained and intended to operate. If we score entities with very little evidence or entities outside the model's training distribution, the predictions may be unreliable.

So entity selection is not just a filtering step. It is a system-design decision that affects coverage, cost, model reliability, and downstream actionability.

---

## 4. What is eligibility filtering?

Eligibility filtering means applying business rules and data-quality rules before scoring.

Example:

```text
Score a host only if:
- host_id is valid
- host was active in the last 30 days
- host has at least 10 crawled URLs
- host is not already blocked
- host is not in permanent allowlist
```

Why is this needed?

Because the model may have been trained on a certain population.

If the model was trained on hosts with at least 50 crawled URLs, but in production we score hosts with only 1 crawled URL, the model may overreact.

Example:

```text
1 error / 1 crawled URL = error_ratio 1.0
```

That looks extremely bad, but it is based on very weak evidence.

So eligibility rules prevent the model from being used outside its reliable operating range.

### Interview answer

Eligibility filtering ensures that we score only entities for which the model can produce meaningful predictions. For example, if the model was trained on hosts with sufficient crawl history, then scoring a new host with only one crawled URL may be risky because its ratio features can be unstable.

Eligibility rules can include valid IDs, minimum activity, sufficient historical data, business allowlists/blocklists, and freshness requirements. These rules protect the model from being applied to entities outside its intended scoring domain.

---

## 5. What is deduplication and why is it important?

Batch input should usually have:

```text
one row per entity per scoring time
```

Bad input:

```text
host_id      scoring_date
abc.com      2026-09-10
abc.com      2026-09-10
xyz.net      2026-09-10
```

If duplicate rows are scored, downstream systems may receive multiple scores for the same entity.

Example:

```text
abc.com → 0.91
abc.com → 0.76
```

Now the downstream system does not know which one is correct.

Duplicates can happen because of:

```text
bad joins
multiple snapshots
late-arriving records
case sensitivity
URL normalization issues
multiple raw sources
```

For URL and host systems, normalization is very important.

Example:

```text
ABC.com
abc.com
www.abc.com
https://abc.com/
http://abc.com
```

Depending on entity definition, some of these may represent the same logical entity.

### Interview answer

Deduplication is important because the batch scoring input should usually contain one row per entity per scoring timestamp. If duplicate rows are present, the model may generate multiple predictions for the same entity, and downstream systems may not know which score to use.

Duplicates can come from bad joins, multiple data sources, late-arriving records, or inconsistent entity normalization. For URL or host-level systems, canonicalization is especially important because `ABC.com`, `abc.com`, and `www.abc.com` may need to be mapped to the same logical entity depending on the design.

So before scoring, I would validate uniqueness of entity IDs and ensure that the input row count is expected.

---

## 6. Why is entity ID preservation important?

The model may not use the entity ID for prediction.

For example, the model may only use:

```text
error_ratio_10d
crawl_ratio_10d
dsat_ratio_30d
static_rank
```

But after prediction, the output must say:

```text
abc.com → junk_score = 0.91
```

not:

```text
row_184923 → junk_score = 0.91
```

So the pipeline must preserve IDs for traceability.

This creates two groups of columns:

```text
model input columns
traceability columns
```

Model input columns go into the model.

Traceability columns are carried along for output and debugging.

### Interview answer

Entity ID preservation is critical because the model may not use IDs as features, but the production system must still map every prediction back to the correct entity. For example, the model may score only numerical features, but the final output must tell us which host, user, URL, publisher, or campaign received that score.

So I would separate model feature columns from traceability columns. The traceability columns are not passed to the model, but they are preserved throughout the pipeline and joined back with the prediction output. This is important for debugging, auditing, downstream consumption, and rollback analysis.

---

## 7. Why do we assign a scoring timestamp?

Every batch scoring row should have a timestamp.

Example:

```text
host_id = abc.com
scoring_timestamp = 2026-09-10 02:00:00
```

This is important because many features are time-window based.

Example:

```text
host_error_ratio_10d
```

This feature actually means:

```text
error ratio during the 10 days before the scoring timestamp
```

Without a scoring timestamp, “last 10 days” becomes ambiguous.

This is even more important when creating historical training data.

For each historical row, features must be computed using only data available before that row's timestamp.

This is called **point-in-time correctness**.

### Interview answer

A scoring timestamp is important because features are often defined relative to time. For example, `host_error_ratio_10d` means error ratio in the 10 days before the scoring timestamp. Without a timestamp, time-window features become ambiguous.

It is also important for point-in-time correctness. During training or backtesting, each historical row should use only features that would have been available before that row's prediction time. Otherwise, we may accidentally leak future information into the model.

---

## 8. What is schema validation?

Schema validation means checking whether the prepared input matches the model's expected format.

The model may expect:

```text
host_error_ratio_10d: float
host_crawl_ratio_10d: float
host_dsat_ratio_30d: float
host_static_rank: float
host_category_encoded: integer
```

But production may send:

```text
host_error_ratio_10d: string
host_crawl_ratio_10d: missing
host_dsat_ratio_30d: null
host_static_rank: float
host_category_encoded: unseen category
```

This can break inference or silently corrupt predictions.

Feature order is also critical.

If the model expects:

```text
[error_ratio, crawl_ratio, dsat_ratio]
```

but receives:

```text
[crawl_ratio, error_ratio, dsat_ratio]
```

the model may still output a score, but it will be wrong.

### Interview answer

Schema validation ensures that the inference input matches the exact feature schema expected by the model. I would check column presence, column order, data types, allowed ranges, null rates, valid categorical values, and timestamp freshness.

This is especially important for ensemble models because if feature order is wrong, the model may still return predictions without crashing. For example, if `crawl_ratio` and `error_ratio` are swapped, the model will treat one signal as another and produce unreliable scores. So schema validation is necessary before calling the model.

---

# 5.7 Feature Joins — Interview Style

## 9. What are feature joins?

Feature joins are the process of attaching feature values to the entities we want to score.

Entity table:

```text
host_id      scoring_timestamp
abc.com      2026-09-10 02:00
xyz.net      2026-09-10 02:00
```

Feature table:

```text
host_id      feature_timestamp      error_ratio_10d      crawl_ratio_10d
abc.com      2026-09-10 01:00       0.42                 0.18
xyz.net      2026-09-10 01:00       0.03                 0.91
```

After join:

```text
host_id      scoring_timestamp      error_ratio_10d      crawl_ratio_10d
abc.com      2026-09-10 02:00       0.42                 0.18
xyz.net      2026-09-10 02:00       0.03                 0.91
```

This looks simple, but in production this is one of the most dangerous steps.

Feature joins can cause:

```text
missing features
duplicate rows
future leakage
row explosion
wrong granularity
stale features
wrong join keys
```

### Interview answer

Feature joins are the step where we attach feature values to the entities selected for scoring. The entity table defines who should be scored, and the feature tables provide the information needed for scoring.

In production, feature joins are risky because they can silently create incorrect model inputs. A bad join may create duplicate rows, missing values, stale features, future leakage, or one-to-many row explosion. So after feature joins, I would always validate row count, entity uniqueness, feature coverage, missing rates, timestamp correctness, and feature freshness.

---

## 10. What is feature granularity?

Feature granularity means the level at which the feature is defined.

Examples:

```text
host-level feature: host_error_ratio_10d
URL-level feature: url_http_status
user-level feature: user_click_count_7d
publisher-level feature: publisher_ctr_30d
```

If the scoring entity is host, then host-level features naturally match.

But if we directly join URL-level features to a host-level entity table, one host may join to many URL rows.

Example:

```text
host_id
abc.com
```

URL table:

```text
abc.com /page1 error=1
abc.com /page2 error=0
abc.com /page3 error=1
```

Direct join creates three rows for one host.

This is a row explosion.

Correct approach:

```text
aggregate URL-level data to host level first
```

Example:

```text
host_error_ratio = error_urls / total_urls
```

Then join one host-level feature row.

### Interview answer

Feature granularity means the entity level at which a feature is computed. For example, `host_error_ratio_10d` is host-level, while `url_http_status` is URL-level.

The feature granularity must match the scoring granularity. If I am scoring hosts, I should join host-level features. If I directly join URL-level rows to a host-level table, I may create multiple rows for the same host, causing row explosion and duplicate predictions. The correct approach is to aggregate lower-level data to the scoring entity level before joining.

---

## 11. What are one-to-one, many-to-one, and one-to-many joins?

A one-to-one join means:

```text
one entity row joins to one feature row
```

This is safest.

A many-to-one join means:

```text
many entity rows join to one shared feature row
```

Example:

```text
many URLs joining to the same host-level feature
```

This can be valid.

A one-to-many join means:

```text
one entity row joins to many feature rows
```

This is dangerous if accidental.

It can duplicate model inputs and distort outputs.

### Interview answer

In feature joins, one-to-one joins are usually safest because each entity gets exactly one feature row. Many-to-one joins can be valid, for example when multiple URLs join to the same host-level feature. But accidental one-to-many joins are dangerous because they duplicate entities and create row explosion.

So after joins, I would compare input row count with output row count and check duplicate entity IDs. If row count unexpectedly increases, it usually indicates a one-to-many join problem.

---

## 12. What is point-in-time correctness?

Point-in-time correctness means:

```text
For a scoring row at time T, use only feature values available at or before time T.
```

Example:

Scoring row:

```text
host_id = abc.com
scoring_time = 2026-09-10 02:00
```

Feature values:

```text
2026-09-09 02:00 → error_ratio = 0.30
2026-09-10 01:00 → error_ratio = 0.42
2026-09-10 03:00 → error_ratio = 0.90
```

Correct feature:

```text
2026-09-10 01:00 → 0.42
```

Wrong feature:

```text
2026-09-10 03:00 → 0.90
```

because that is from the future.

Using future values causes leakage.

### Interview answer

Point-in-time correctness means that for each prediction timestamp, the joined features must only use information that was available before that timestamp. This is critical to avoid future leakage.

For example, if I am scoring a host at 2 AM, I can use the latest feature snapshot before 2 AM, but I should not use a feature computed at 3 AM. In historical training data, this becomes even more important because each training row may have a different prediction timestamp. The feature join must be an as-of join: for each entity and timestamp, join the latest feature value available at or before that time.

---

## 13. What is an as-of join?

An as-of join is a time-aware join.

Instead of joining only on entity ID, we join on:

```text
entity_id
+
latest feature_timestamp <= scoring_timestamp
```

Example:

```text
For host abc.com at 2 AM:
choose the latest feature row before or equal to 2 AM
```

This prevents future leakage.

### Interview answer

An as-of join is a time-aware feature join. For each entity row with a scoring timestamp, we join the latest feature value whose feature timestamp is less than or equal to the scoring timestamp.

This is important because a normal join may accidentally pick the latest feature overall, which could be from the future for historical training rows. As-of joins help preserve point-in-time correctness and prevent future leakage.

---

## 14. What is join coverage?

Join coverage means the percentage of entity rows that successfully received a feature value after the join.

Example:

```text
Input entities = 10 million
Rows with host_error_ratio_10d = 9.7 million
Coverage = 97%
```

If coverage suddenly drops:

```text
Yesterday: 97%
Today: 42%
```

something likely broke.

Possible causes:

```text
feature pipeline delay
join key change
entity normalization bug
missing partition
upstream data failure
```

### Interview answer

Join coverage measures how many scoring entities successfully received feature values after the join. For example, if I have 10 million hosts and 9.7 million get a particular feature, the coverage is 97%.

This is an important production metric because a SQL join can complete successfully while still producing bad data. If feature coverage drops suddenly, it may indicate a pipeline delay, join key mismatch, missing partition, or upstream failure. So I would monitor join coverage for each important feature group before allowing the batch scoring output to be published.

---

## 15. Inner join vs left join in feature joins

An inner join keeps only rows that match in both tables.

If entity table has 10 million hosts and feature table matches 8 million:

```text
inner join output = 8 million rows
```

Problem:

```text
2 million entities silently disappear
```

A left join keeps all entity rows:

```text
left join output = 10 million rows
```

with missing features for unmatched rows.

This is usually safer for scoring because we preserve the scoring population and handle missingness explicitly.

### Interview answer

For batch scoring, I usually prefer a left join from the entity table to feature tables because it preserves the full scoring population. If we use an inner join, entities without matching features may silently disappear, which can reduce coverage and bias the scored population.

With a left join, missing feature values remain visible, and we can handle them intentionally using imputation, missing indicators, fallback logic, or rejection rules. However, if a feature is absolutely required, then we may reject those rows after explicitly measuring missingness rather than losing them silently through an inner join.

---

## 16. What is row explosion?

Row explosion happens when one entity row joins to multiple feature rows.

Example:

Entity table:

```text
host_id
abc.com
```

Feature table:

```text
host_id      url       error_flag
abc.com      /page1    1
abc.com      /page2    0
abc.com      /page3    1
```

Direct join output:

```text
abc.com /page1 1
abc.com /page2 0
abc.com /page3 1
```

Now one host became three rows.

If the model expects one row per host, this is wrong.

### Interview answer

Row explosion happens when one scoring entity joins to multiple feature rows, usually due to an accidental one-to-many join. For example, if I am scoring hosts but directly join URL-level records, one host may become many rows.

This can create duplicate predictions and distort downstream metrics. The correct approach is to aggregate lower-level data to the scoring level before joining. I would also validate row counts and duplicate entity IDs after every major join to detect row explosion early.

---

## 17. What is feature freshness in joins?

Feature freshness means how recent the joined feature value is compared to the scoring time.

Example:

```text
scoring_time = 2026-09-10 02:00
feature_time = 2026-09-07 02:00
```

Feature age:

```text
3 days
```

If freshness SLA is 24 hours, this feature is stale.

A stale feature may be worse than missing because it looks valid but represents old reality.

### Interview answer

Feature freshness means how old the joined feature value is relative to the scoring timestamp. A feature may exist, but if it is too old, it may not represent the current state of the entity.

For example, if the scoring time is today but the latest feature value is from three days ago, and the freshness SLA is 24 hours, then the feature should be marked as stale. I would track feature age, freshness status, and stale-feature rate during feature joins. If important features are stale, I may block scoring, use fallback logic, or mark the score as lower confidence.

---

## 18. How do feature joins cause leakage?

Feature joins cause leakage when they attach information that would not have been available at prediction time.

Example:

Prediction time:

```text
2026-09-10 02:00
```

Wrong feature window:

```text
full month of September
```

This includes data from September 11 to September 30, which is future information.

Another example:

```text
manual_review_result
```

If manual review happens after the prediction time, then using it as a feature leaks the label.

### Interview answer

Feature joins can cause leakage when they attach future information or label-derived information to a training or scoring row. For example, if I am creating a feature for September 10 but use the full month of September, then I include data from after the prediction time. That gives the model information it would not have in production.

Another example is joining manual review outcomes or post-action signals as features when those outcomes happen after the prediction time. Tree ensembles are especially good at exploiting such leakage, so the model may show very high offline performance but fail in production. To prevent this, feature joins must be point-in-time correct and label-derived features must be excluded.

---

## 19. How do you validate feature joins?

After feature joins, I would validate:

```text
row count
duplicate entity IDs
join coverage
missing rates
feature timestamps
feature freshness
feature ranges
feature distributions
one-to-many joins
future leakage
label-derived features
```

Validation should happen before model scoring.

### Interview answer

I validate feature joins by checking both structural correctness and statistical correctness. Structurally, I check that row count has not unexpectedly changed, entity IDs are unique, and no one-to-many join happened. I also check join coverage and missing rates for every important feature group.

Statistically, I check feature ranges, distributions, freshness, and timestamp validity. For time-window features, I ensure that feature timestamps are at or before the scoring timestamp. I also check that no label-derived or future information has been joined. Only after these checks pass should the feature table be sent to the model.

---

# 20. Complete interview answer: explain 5.6 and 5.7 together

Use this when interviewer asks:

```text
How would you prepare data for batch inference in a production ensemble model?
```

Answer:

```text
For batch inference, I would start with input data preparation. The first step is to define the scoring population: which entities should be scored and at what timestamp. For example, in a host-level junk scoring system, the entity may be a host, and the scoring timestamp may be the daily batch run time.

Then I would apply eligibility filtering to ensure we score only valid and meaningful entities. This may include checking valid IDs, recent activity, minimum historical data, allowlists or blocklists, and whether the entity falls within the model's intended scoring domain.

After that, I would deduplicate entities so that we have one row per entity per scoring timestamp. I would preserve traceability columns like host_id, user_id, URL, or campaign_id even if these are not passed to the model, because the final prediction must be mapped back to the correct entity.

Then I would join features from different feature tables. This is one of the most sensitive parts of the pipeline. I would make sure that the feature granularity matches the scoring granularity. For example, if I am scoring hosts, URL-level data should first be aggregated to host level before joining.

I would also make the joins point-in-time correct. For every scoring row at time T, I would join only feature values available at or before T. This avoids future leakage. In historical training data, this usually requires an as-of join.

After joining, I would validate row counts, duplicate IDs, join coverage, missing rates, feature freshness, feature timestamps, feature ranges, and schema compatibility with the model. I would also check for row explosion and label leakage.

Only after the input table is clean, complete, point-in-time safe, and schema-compatible would I call the ensemble model for scoring.
```

---

# Crisp version

```text
In batch inference, 5.6 is about preparing the correct scoring population, and 5.7 is about attaching the correct features to that population.

For input preparation, I would select eligible entities, deduplicate them, preserve IDs, assign scoring timestamps, and validate schema.

For feature joins, I would ensure correct granularity, use left joins where appropriate, monitor coverage, avoid one-to-many row explosion, enforce point-in-time correctness, check freshness, and prevent leakage.

The key principle is that before model.predict, the system must produce one clean, valid, point-in-time-safe feature row per entity.
```

---

# Most Important Memory Line

```text
In production batch inference, the hard part is often not model.predict. The hard part is creating the correct feature table before model.predict.
```
