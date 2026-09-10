# Ensemble Learning in Production

# Part 5.6 — Input Data Preparation

# Part 5.7 — Feature Joins

We are inside:

```text
Part 5 — Batch Inference System
```

Already covered:

```text
5.1 What is batch inference?
5.2 Why batch inference exists
5.3 Batch inference vs real-time inference
5.4 When ensemble models are suitable for batch scoring
5.5 End-to-end batch scoring architecture
```

Now we study:

```text
5.6 Input Data Preparation
5.7 Feature Joins
```

These two topics are extremely important because before the model predicts anything, the system must answer:

```text
Which records should be scored?
Are those records valid?
Do we have the right features for those records?
Are feature joins correct?
Are we avoiding duplicate rows?
Are we avoiding future leakage?
Are we preserving entity IDs so predictions can be traced back?
```

Many production ML failures happen before `model.predict()` is even called.

---

# 5.6 Input Data Preparation

## 5.6.1 What Does Input Data Mean in Batch Inference?

In batch inference, input data is the set of records that the model will score.

A record usually represents:

```text
one entity
+
one scoring time
```

For example:

```text
host_id = abc.com
scoring_date = 2026-09-10
```

or:

```text
user_id = U123
scoring_date = 2026-09-10
```

or:

```text
url = https://example.com/page1
scoring_date = 2026-09-10
```

So input data is not just a collection of features.

It begins with the list of things we want to score.

Before model scoring, the system must first decide:

```text
What is the scoring population?
```

For example:

```text
All hosts?

Only active hosts?

Only hosts crawled in the last 30 days?

Only hosts with enough historical data?

Only hosts not already blocked?

Only hosts with valid IDs?
```

This step is called:

```text
Candidate Generation
```

or:

```text
Entity Selection
```

---

# 5.6.2 Why Input Data Preparation Exists

In a notebook, you may already have a clean DataFrame:

```python
X_test
```

Then you can simply call:

```python
model.predict(X_test)
```

Production systems are very different.

Production does not usually start with a clean DataFrame.

It starts with messy raw systems such as:

```text
logs
tables
events
metadata stores
crawl outputs
click logs
impression logs
manual labels
daily snapshots
partial updates
late-arriving data
```

So input preparation exists to convert messy production data into a:

```text
clean
valid
consistent
scoreable
```

dataset.

### Goal

```text
Create one clean row per entity that should be scored.
```

Batch inference platforms may execute scoring over prepared input datasets, but the production pipeline is still responsible for ensuring that the records, IDs, and features are correct.

---

# 5.6.3 The Most Basic Batch Input Table

For a host junk-risk model, the initial input may contain only:

```text
host_id        scoring_date
abc.com        2026-09-10
xyz.net        2026-09-10
news123.com    2026-09-10
```

At this point, we may not yet have model features.

This is just the **entity list**.

Later, features are joined onto these entities.

After feature joins:

```text
host_id        scoring_date    error_ratio_10d    crawl_ratio_10d    dsat_ratio_30d
abc.com        2026-09-10      0.42               0.18               0.25
xyz.net        2026-09-10      0.03               0.91               0.01
news123.com    2026-09-10      0.11               0.64               0.08
```

Now the rows become model-ready.

---

# 5.6.4 Entity Table vs Feature Table

This distinction is extremely important.

The **entity table** answers:

```text
Who should be scored?
```

The **feature table** answers:

```text
What information do we know about them?
```

Example:

```text
Entity Table:

host_id
scoring_date
```

Feature table:

```text
host_id
feature_timestamp
error_ratio_10d
crawl_ratio_10d
dsat_ratio_30d
```

The model-ready input is created by joining the feature values onto the entity rows.

Conceptually:

```text
Entity Table
     +
Feature Tables
     ↓
Model Input Table
```

---

# 5.6.5 Input Data Preparation Is Not Just Cleaning

Many beginners think input preparation means:

```text
remove nulls
remove duplicates
fix data types
```

Those are important steps, but production input preparation is much broader.

It includes:

```text
1. Entity selection
2. Eligibility filtering
3. Deduplication
4. Snapshot selection
5. Timestamp assignment
6. Schema validation
7. ID validation
8. Feature availability checks
9. Handling missing records
10. Preparing data in the exact model input format
```

Let us go through these deeply.

---

# 5.6.6 Entity Selection

Entity selection means deciding:

```text
Which entities should be scored in this batch run?
```

For host junk scoring:

```text
Possible universe:
all known hosts
```

But the actual scoreable population may be:

```text
hosts seen in the last 30 days
hosts with valid host_id
hosts with at least some crawl activity
hosts not already permanently blocked
```

Why not score everything?

Because scoring everything may be expensive and unnecessary.

Suppose:

```text
All known hosts = 1 billion

Active hosts = 80 million

Hosts with enough useful signals = 20 million
```

Scoring all `1 billion` hosts would create massive:

```text
compute cost
storage cost
processing time
noise
```

But scoring only `20 million` may miss long-tail risky hosts.

So entity selection creates a tradeoff:

```text
More coverage
    ↓
More cost + more noise

Less coverage
    ↓
Cheaper + cleaner
but potentially misses important cases
```

This is therefore a **system-design decision**, not just a data-processing step.

---

# 5.6.7 Eligibility Filtering

Eligibility filtering means applying business and data-quality rules before scoring.

Example:

```text
Score a host only if:

- host_id is valid
- host was active in the last 30 days
- host has at least 10 crawled URLs
- host is not in the permanent allowlist
- host is not already blocked
```

Why do this?

Because the model may have been trained on a particular population.

If production scoring uses a very different population, predictions may become unreliable.

Suppose the model was trained only on hosts with:

```text
at least 50 crawled URLs
```

Now imagine production scores a host with:

```text
1 crawled URL
```

If that one URL has an error:

```text
1 error / 1 crawled URL
```

then:

```text
error_ratio = 1.0
```

But this ratio is based on extremely weak evidence.

The model may interpret it as a strong signal and overreact.

Therefore:

```text
Eligibility rules help prevent the model
from being used outside its intended scoring domain.
```

---

# 5.6.8 Deduplication

Batch input should usually contain:

```text
one row
per entity
per scoring time
```

Bad input:

```text
host_id      scoring_date
abc.com      2026-09-10
abc.com      2026-09-10
xyz.net      2026-09-10
```

This creates duplicate predictions.

For example:

```text
abc.com → 0.91
abc.com → 0.76
```

Now the downstream system must ask:

```text
Which score should I use?
```

Duplicates can occur because of:

```text
multiple data sources
bad joins
late-arriving records
case sensitivity
URL normalization problems
multiple snapshots
```

For hosts and URLs, normalization is particularly important.

Examples:

```text
ABC.com
abc.com
http://abc.com
https://abc.com/
www.abc.com
```

Depending on your entity definition, some of these may represent the same logical entity.

Therefore, input preparation may require:

```text
Entity Normalization
        ↓
Deduplication
        ↓
One Canonical Entity Row
```

---

# 5.6.9 Entity ID Preservation

This is a critical production detail.

The model itself may not need the entity ID.

For example, the model may only receive:

```text
error_ratio_10d
crawl_ratio_10d
dsat_ratio_30d
static_rank
```

But the prediction must still be mapped back to the correct entity.

Therefore, the pipeline must preserve identifiers such as:

```text
host_id
url
user_id
customer_id
publisher_id
campaign_id
```

This creates an important distinction:

```text
Columns used by the model
```

versus:

```text
Columns used for traceability
```

For example:

```text
Input Row:

host_id = abc.com
error_ratio = 0.42
crawl_ratio = 0.18
```

The model may receive only:

```text
0.42
0.18
```

But the output must still be:

```text
abc.com → junk_score = 0.91
```

not:

```text
row_184923 → junk_score = 0.91
```

### Production Principle

```text
Never lose the mapping between entity and prediction.
```

---

# 5.6.10 Timestamp Assignment

Every batch scoring row should have a scoring timestamp.

Example:

```text
host_id = abc.com

scoring_timestamp = 2026-09-10 02:00:00
```

Why?

Because time-window features are defined relative to that timestamp.

For example:

```text
host_error_ratio_10d
```

actually means:

```text
Error ratio during the 10 days
before the scoring timestamp.
```

Without a scoring timestamp, the meaning of:

```text
last 10 days
```

is ambiguous.

This becomes even more important when constructing historical training or backtesting data.

Suppose we create examples for:

```text
2026-08-01
2026-08-02
2026-08-03
```

Each row must use features available before its own timestamp.

This is the foundation of:

```text
Point-in-Time Correctness
```

---

# 5.6.11 Snapshot Selection

Batch systems often consume snapshot-based tables.

Example:

```text
host_metadata_snapshot_2026_09_08
host_metadata_snapshot_2026_09_09
host_metadata_snapshot_2026_09_10
```

When scoring on September 10, we need to decide:

```text
Which snapshot is safe to use?
```

Usually:

```text
The latest successfully completed snapshot.
```

A dangerous pattern is:

```text
Read today's snapshot
while the upstream pipeline is still writing it.
```

This may result in:

```text
partial rows
missing partitions
inconsistent values
```

A safer pattern is:

```text
Upstream Job Runs
        ↓
Snapshot Written
        ↓
Snapshot Validated
        ↓
Marked Successful
        ↓
Scoring Pipeline Reads It
```

### Production Question

```text
Which upstream data version is safe to consume?
```

This must be explicit.

---

# 5.6.12 Schema Validation

Before scoring, input data must match the schema expected by the model.

The model may expect:

```text
host_error_ratio_10d: float
host_crawl_ratio_10d: float
host_dsat_ratio_30d: float
host_static_rank: float
host_category_encoded: integer
```

But production input may contain:

```text
host_error_ratio_10d: string
host_crawl_ratio_10d: missing
host_dsat_ratio_30d: null
host_static_rank: float
host_category_encoded: unseen value
```

Without strong schema validation, the model may:

```text
crash
```

or worse:

```text
silently produce incorrect predictions
```

Silent errors are especially dangerous.

A good batch input validation process checks:

```text
column presence
column order
data types
allowed ranges
null percentage
valid categories
duplicate entity IDs
row counts
timestamp freshness
```

---

## Feature Order Mismatch

Tree ensemble models can be particularly vulnerable when the scoring code passes features as arrays.

Suppose the model expects:

```text
[
    error_ratio,
    crawl_ratio,
    dsat_ratio
]
```

But it receives:

```text
[
    crawl_ratio,
    error_ratio,
    dsat_ratio
]
```

The model may still return a score.

There may be no error.

But the prediction is meaningless.

### Important Principle

```text
A successful model call does not guarantee a correct prediction.
```

---

# 5.6.13 Missing Input Records

Sometimes the problem is not a missing feature.

The entire entity may be missing from the scoring input.

Example:

```text
Yesterday:
80 million active hosts

Today:
10 million active hosts
```

This could represent a real event.

But more likely something broke.

Possible causes:

```text
upstream crawl logs delayed
entity selection query bug
partition missing
date filter wrong
join key changed
permission issue
storage path issue
```

Therefore, input preparation should include row-count monitoring.

Example rule:

```text
If today's input row count differs from
the 7-day average by more than 30%:

    alert
    investigate
    potentially block publishing
```

This is not model theory.

This is **production ML reliability**.

---

# 5.6.14 Input Data Preparation for Ensemble Models Specifically

Tree ensembles are powerful, but they can produce confident predictions even when the input data is wrong.

Example:

```text
Expected:

host_error_ratio_10d = 0.12
```

Buggy input:

```text
host_error_ratio_10d = 12
```

The model may treat `12` as an extreme value and traverse a completely different set of branches.

Another example:

```text
Missing value encoded as 0
```

Suppose:

```text
error_ratio = 0
```

normally means:

```text
No observed errors.
```

But if `0` is being used to mean:

```text
Feature unavailable.
```

then the model receives a false signal.

These states are not equivalent:

```text
Missing value ≠ Zero

Stale value ≠ Fresh value

Default value ≠ Observed value
```

### Production Memory Hook

```text
The model only sees numbers,
but the system must preserve what those numbers mean.
```

---

# 5.6.15 Final Mental Model for Input Preparation

Think of input preparation as creating a valid **contract** before the model is allowed to score.

The contract says:

```text
These are the entities to score.

Each entity appears once.

Each entity has a valid ID.

Each entity has a scoring timestamp.

Each required feature column exists.

Each feature has the expected type and range.

Missing values are handled consistently.

The input snapshot is complete.

The scoring population matches
the population for which the model was designed.
```

Only after these conditions are satisfied should the system call:

```python
model.predict(...)
```

---

# 5.7 Feature Joins

Feature joins are where we attach feature values to the entities that need to be scored.

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

Joined model input:

```text
host_id      scoring_timestamp      error_ratio_10d      crawl_ratio_10d
abc.com      2026-09-10 02:00       0.42                 0.18
xyz.net      2026-09-10 02:00       0.03                 0.91
```

This looks simple.

But feature joins are one of the most dangerous parts of production ML.

---

# 5.7.1 Why Feature Joins Are Dangerous

A feature join can silently create incorrect training or scoring data.

Common failures include:

```text
one-to-many join explosion
missing join keys
wrong timestamp joins
joining future feature values
joining stale features
duplicate feature rows
wrong entity granularity
different join logic between training and serving
```

The model will not know that any of this happened.

It will simply score the table it receives.

Therefore:

```text
Feature join correctness is part of model correctness.
```

---

# 5.7.2 Entity Granularity Mismatch

Every feature has an entity level.

Examples:

```text
Host-level feature:
host_error_ratio_10d

URL-level feature:
url_http_status

User-level feature:
user_click_count_7d

Publisher-level feature:
publisher_ctr_30d
```

If the scoring entity is:

```text
host
```

then joining host-level features is straightforward.

But joining URL-level rows directly onto a host-level entity table can create problems.

Entity table:

```text
host_id
abc.com
```

URL-level feature table:

```text
host_id      url      url_error_flag
abc.com      /page1   1
abc.com      /page2   0
abc.com      /page3   1
```

A direct join produces:

```text
abc.com      /page1      1
abc.com      /page2      0
abc.com      /page3      1
```

Now one host has become three rows.

This is a:

```text
One-to-Many Join Explosion
```

The correct approach is usually:

```text
Aggregate URL-level features
to host level first.
```

For example:

```text
host_error_ratio =
error_urls / total_urls
```

For `abc.com`:

```text
error_urls = 2
total_urls = 3

host_error_ratio = 2 / 3 = 0.67
```

Then join:

```text
host_id      host_error_ratio
abc.com      0.67
```

### Production Rule

```text
Feature granularity must match scoring granularity.
```

---

# 5.7.3 One-to-One, Many-to-One, and One-to-Many Joins

Feature joins can be classified by their cardinality.

## One-to-One Join

```text
One entity row
→
One feature row
```

Example:

```text
host_id abc.com
→
one host metadata row
```

This is usually the safest structure.

---

## Many-to-One Join

```text
Many entity rows
→
One shared feature row
```

Suppose the scoring entity is a URL.

Multiple URLs may belong to the same host:

```text
url1 → abc.com
url2 → abc.com
url3 → abc.com
```

All of them may legitimately receive the same host-level feature.

Example:

```text
url1 → host_quality_score = 0.82
url2 → host_quality_score = 0.82
url3 → host_quality_score = 0.82
```

This can be valid.

---

## One-to-Many Join

```text
One entity row
→
Many feature rows
```

Example:

```text
one host
→
many URLs
```

This is dangerous before model scoring unless the multiplication is intentional.

### Production Rule

```text
Never allow accidental one-to-many joins
before model scoring.
```

---

# 5.7.4 Time-Aware Feature Joins

This is one of the most important feature-join concepts.

A feature value has a timestamp.

A scoring row also has a timestamp.

The model must use only feature values that were available before or at the scoring time.

Suppose:

```text
host_id = abc.com

scoring_time = 2026-09-10 02:00
```

Available feature values:

```text
2026-09-09 02:00 → error_ratio = 0.30

2026-09-10 01:00 → error_ratio = 0.42

2026-09-10 03:00 → error_ratio = 0.90
```

The correct feature value is:

```text
2026-09-10 01:00 → 0.42
```

because it is the latest value available before scoring time.

Using:

```text
2026-09-10 03:00 → 0.90
```

would use future information.

That is:

```text
Future Leakage
```

---

# 5.7.5 Point-in-Time Correctness

Point-in-time correctness means:

```text
For every scoring row at time T,
all joined feature values must have been
available at or before T.
```

It is easy to violate this accidentally.

For example:

```sql
SELECT
    host_id,
    MAX(feature_timestamp),
    error_ratio
FROM host_features
GROUP BY host_id;
```

This retrieves the latest feature value overall.

That may be fine for current-day inference.

But it is incorrect for historical training examples if the latest feature value occurred after the historical prediction time.

The correct logic is closer to:

```text
For each entity row:

Find the latest feature_timestamp
such that:

feature_timestamp <= scoring_timestamp
```

This is commonly called an:

```text
As-Of Join
```

### Memory Hook

```text
Feature joins must time-travel correctly.
```

---

# 5.7.6 Training Feature Joins vs Batch Inference Feature Joins

For current-day batch scoring, we may have:

```text
scoring_time = today at 2 AM
```

The system may simply need:

```text
latest completed feature snapshot
before 2 AM
```

Historical training data is more complicated.

Suppose:

```text
host_id      prediction_time
abc.com      2026-07-01
abc.com      2026-08-01
abc.com      2026-09-01
```

Each row needs features from before its own prediction timestamp.

Therefore:

```text
July row → July historical features

August row → August historical features

September row → September historical features
```

The principle is the same in both training and inference:

```text
Use only what would have been known
at prediction time.
```

If training joins leak future information, the model learns unrealistically strong patterns and may fail in production.

---

# 5.7.7 Join Keys

A join key is the identifier used to connect an entity row to its feature row.

Examples:

```text
host_id
url_id
user_id
publisher_id
campaign_id
customer_id
```

Join-key problems are common.

Example:

Entity table:

```text
host = www.example.com
```

Feature table:

```text
host = example.com
```

The join may fail.

Another example:

```text
Entity:
ABC.com

Feature:
abc.com
```

A case mismatch can cause missing feature values.

Another example:

```text
https://abc.com/page/

vs

https://abc.com/page
```

Depending on the system's canonicalization rules, these may represent the same or different entities.

For URL and host systems:

```text
Normalization is not optional.

It is part of feature correctness.
```

---

# 5.7.8 Join Coverage

Join coverage measures how many entity rows successfully receive a feature value.

Suppose:

```text
Input entities = 10 million

Rows with host_error_ratio_10d = 9.7 million
```

Then:

```text
Coverage = 9.7M / 10M
         = 97%
```

If coverage suddenly changes:

```text
Yesterday = 97%

Today = 42%
```

something probably broke.

Possible causes:

```text
feature table delayed
join key changed
entity normalization changed
partition missing
feature pipeline failed
```

Therefore, every important feature or feature group should have coverage monitoring.

### Production Rule

```text
A feature join is not successful
just because the SQL query completed.

It is successful only when
coverage and quality are acceptable.
```

---

# 5.7.9 Joining Multiple Feature Tables

In real production systems, features often come from many different pipelines.

For a host junk model:

```text
host_crawl_features

host_spam_features

host_click_features

host_metadata_features

host_index_features
```

The final model table is built by joining all of them.

Conceptually:

```text
Entity Table
    ↓
Join Crawl Features
    ↓
Join Spam Features
    ↓
Join Click Features
    ↓
Join Metadata Features
    ↓
Join Index Features
    ↓
Final Model Table
```

Each join can fail independently.

Therefore, the pipeline should monitor feature coverage by group.

Example:

```text
crawl feature coverage:     99.2%

spam feature coverage:      94.1%

click feature coverage:     78.3%

metadata feature coverage:  99.9%
```

This makes debugging much easier.

If model predictions suddenly shift, we can inspect:

```text
Which feature group changed?
```

---

# 5.7.10 Join Strategy: Inner Join vs Left Join

This is an important production decision.

## Inner Join

An inner join keeps only entities that have a matching feature row.

Example:

```text
Entity table = 10 million hosts

Matching feature rows = 8 million
```

After an inner join:

```text
Output = 8 million hosts
```

Problem:

```text
2 million entities silently disappeared.
```

That may be dangerous.

---

## Left Join

A left join preserves the entire scoring population.

Example:

```text
Entity rows = 10 million
Feature matches = 8 million
```

Output:

```text
10 million rows

8 million → have feature values

2 million → feature is null
```

This is often safer for scoring because the missingness becomes explicit.

The system can then decide whether to:

```text
impute
use default
use missing indicator
reject row
use fallback
```

### Common Production Pattern

```text
Entity Table
    LEFT JOIN
Feature Tables
    ↓
Preserve Scoring Population
    ↓
Validate Missingness
```

However, this depends on the application.

If a feature is absolutely required, entities missing that feature may need to be rejected.

---

# 5.7.11 Feature Freshness in Joins

A feature may exist but still be unusable because it is stale.

Suppose:

```text
scoring_time = 2026-09-10 02:00

feature_time = 2026-09-07 02:00
```

Then:

```text
feature_age = 3 days
```

If the feature's freshness SLA is:

```text
24 hours
```

the feature should be considered stale.

So a feature join should not only ask:

```text
Did we find a feature row?
```

It should also ask:

```text
Is the feature row recent enough?
```

The joined table may include metadata such as:

```text
feature_age_hours
is_stale_feature
```

This allows the scoring system to distinguish:

```text
feature unavailable
```

from:

```text
feature available but stale
```

---

# 5.7.12 Feature Joins and Leakage Through Aggregation Windows

Time-window features can accidentally leak future information.

Suppose the feature is:

```text
host_dsat_ratio_30d
```

and the scoring time is:

```text
2026-09-10 02:00
```

A valid window would use data from approximately:

```text
2026-08-11 02:00
to
2026-09-10 02:00
```

A dangerous implementation might use:

```text
full month of September
```

But if we are scoring on September 10, the full September aggregate contains data from:

```text
September 11–30
```

which is future information.

That creates leakage.

This is especially common when historical training tables are built from convenient calendar-based aggregates.

The offline model then appears unusually strong because it has access to future behavior.

Production performance drops because that information is unavailable at actual prediction time.

---

# 5.7.13 Feature Joins and Label Leakage

Sometimes the feature table contains information derived from the eventual label.

Suppose the label is:

```text
host_is_junk
```

A dangerous feature might be:

```text
manual_review_result
```

If manual review happens after the prediction time, that feature should not be available to the model.

Another leaky feature:

```text
number_of_times_host_was_blocked_after_scoring_date
```

This directly includes future behavior.

Tree ensembles are particularly good at exploiting such shortcuts.

For example:

```python
if manual_review_result == "junk":
    predict_junk()
```

Offline performance becomes excellent.

But production performance can collapse.

### Important Principle

```text
A powerful model will exploit leakage
rather than politely ignore it.
```

---

# 5.7.14 Feature Joins and Row Explosion

Consider this entity table:

```text
host_id
abc.com
```

Feature table:

```text
host_id      url      error_flag
abc.com      u1       1
abc.com      u2       0
abc.com      u3       1
```

A direct join produces:

```text
abc.com      u1      1
abc.com      u2      0
abc.com      u3      1
```

Instead of:

```text
1 host row
```

we now have:

```text
3 host rows
```

If the model expects host-level predictions, this is wrong.

The correct approach is to aggregate first.

```text
Errors = 2
URLs = 3

error_ratio = 2 / 3 = 0.67
```

Then join:

```text
abc.com      0.67
```

Row explosion can also distort model evaluation.

If one entity appears multiple times, it may receive greater effective weight than other entities.

Therefore, after joins, always compare:

```text
Input entity count

Output row count

Unique entity count

Duplicate entity count
```

---

# 5.7.15 Joining Predictions Back to Input Records

After model scoring, predictions must be mapped back to their entity IDs.

The model input may contain:

```text
error_ratio_10d
crawl_ratio_10d
dsat_ratio_30d
```

But the final output needs:

```text
host_id
junk_score
```

Conceptually:

```text
Input Row:

host_id + features
```

The model receives:

```text
features only
```

The final output becomes:

```text
host_id + prediction
```

This sounds trivial, but it requires care at scale.

If row order changes during:

```text
partitioning
distributed processing
filtering
joining
sorting
```

then a prediction can accidentally be attached to the wrong entity.

Therefore, robust systems use:

```text
stable IDs
deterministic joins
explicit row mapping
```

rather than assuming row order will always remain unchanged.

---

# 5.7.16 Feature Join Validation Checklist

After all feature joins, validate the resulting table.

```text
1. Row count did not unexpectedly change.

2. Entity IDs are unique where uniqueness is expected.

3. No accidental one-to-many joins occurred.

4. Feature coverage is acceptable.

5. Missing rates are within expected limits.

6. Feature timestamps are <= scoring timestamp.

7. Feature age is within freshness SLA.

8. Feature distributions look reasonable.

9. Feature schema matches model expectations.

10. Categorical values are valid.

11. No future data was joined.

12. No label-derived features are present.
```

This is an extremely practical production checklist.

---

# 5.7.17 Production Example: Host Junk Scoring Input Preparation and Feature Joins

Let us design the complete flow.

## Step 1 — Build Entity Table

```text
Select hosts seen in crawl logs
during the last 30 days.

Normalize host names.

Remove invalid hosts.

Remove duplicates.

Assign:
scoring_time = current batch run time
```

Output:

```text
host_id      scoring_time
abc.com      2026-09-10 02:00
xyz.net      2026-09-10 02:00
```

---

## Step 2 — Join Crawl Features

Features:

```text
host_crawled_urls_10d
host_error_ratio_10d
host_crawl_ratio_10d
```

Validation:

```text
coverage >= expected threshold

ratios between 0 and 1

feature_timestamp <= scoring_time
```

---

## Step 3 — Join Spam Features

Features:

```text
host_spam_ratio_10d
host_crushlist_flag
host_demoted_url_ratio
```

Validation:

```text
no future labels

no post-action signals

feature freshness valid
```

---

## Step 4 — Join Click Dissatisfaction Features

Features:

```text
host_dsat_ratio_30d
avg_dwell_time_30d
short_click_ratio_30d
```

Validation:

```text
missingness acceptable

low-volume hosts handled carefully

ratios smoothed if needed
```

---

## Step 5 — Join Metadata Features

Features:

```text
host_age
static_rank
language
region
category
```

Validation:

```text
category values valid

static_rank range valid

metadata snapshot complete
```

---

## Step 6 — Create Final Model Table

Final schema:

```text
host_id
scoring_time
host_crawled_urls_10d
host_error_ratio_10d
host_crawl_ratio_10d
host_spam_ratio_10d
host_dsat_ratio_30d
avg_dwell_time_30d
static_rank
missing_feature_count
feature_version
```

This table is now ready for model scoring.

---

# 5.7.18 Production Example: CTR Batch Features for Real-Time Serving

Sometimes batch processing does not generate final predictions.

Instead, it generates expensive historical features that are later consumed by a real-time model.

For example, the batch job may compute:

```text
publisher_ctr_7d
campaign_ctr_7d
advertiser_quality_score
domain_safety_score
```

These features can then be materialized into an online feature store.

The architecture becomes:

```text
Historical Logs
        ↓
Batch Aggregation Job
        ↓
Feature Table
        ↓
Materialize Features
        ↓
Online Feature Store
        ↓
Real-Time Ad Request
        ↓
Fetch Precomputed Features
        ↓
Combine with Live Request Features
        ↓
Online Model
        ↓
Ad Score
```

This is a **hybrid production architecture**.

The batch system performs expensive historical aggregation.

The online system performs latency-sensitive scoring.

---

# 5.6–5.7 Final Summary

Input data preparation answers:

```text
Who should be scored?

Are the entities valid?

Are there duplicates?

Are IDs normalized?

Is the scoring timestamp correct?

Does the scoring population match
the population for which the model was designed?
```

Feature joins answer:

```text
What information should be attached to each entity?

Are features joined at the correct entity level?

Are timestamps correct?

Are we avoiding future leakage?

Is feature coverage acceptable?

Are missing values handled correctly?

Did row count change unexpectedly?
```

---

# Production Mental Model

```text
Raw Production Data
        ↓
Entity Selection
        ↓
Eligibility Filtering
        ↓
Normalization
        ↓
Deduplication
        ↓
Assign Scoring Timestamp
        ↓
Select Valid Data Snapshot
        ↓
Join Feature Group 1
        ↓
Validate
        ↓
Join Feature Group 2
        ↓
Validate
        ↓
Join Feature Group 3
        ↓
Validate
        ↓
Final Schema Validation
        ↓
Point-in-Time Correctness Check
        ↓
Final Model-Ready Feature Table
        ↓
model.predict(...)
```

---

# Most Important Memory Hook

```text
Before model.predict(),
the system must create a correct,
complete,
point-in-time-safe
feature table.
```

And the second important memory line:

```text
Most batch inference failures
are data failures
before they are model failures.
```

---

# Join Memory Hook

```text
Entity Table
    +
Feature Tables
    ↓
Correct Granularity
    +
Correct Join Key
    +
Correct Timestamp
    +
Correct Coverage
    ↓
Model-Ready Table
```

---

# Critical Production Checks

```text
Entity count before joins
        ≈
Entity count after joins

Feature timestamp
        <=
Scoring timestamp

Feature granularity
        =
Scoring granularity

Missing
        ≠
Zero

Stale
        ≠
Fresh

Future information
        =
Never allowed
```

---

# Next Topics

```text
5.8 Loading the Ensemble Model

5.9 Distributed Scoring
```

This is where we move from **data preparation** into the actual **model execution system**.
````
