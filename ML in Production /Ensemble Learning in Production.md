# Ensemble Learning in Production

# Part 3 — Feature Pipeline Design

This is one of the most important parts of production Machine Learning.

Many people think production ML is mainly about choosing the best model:

```text
Random Forest
XGBoost
LightGBM
CatBoost
Neural Network
```

But in real production systems, the model is often not the biggest problem.

The bigger problem is usually:

```text
Are the right features available?
Are they fresh?
Are they correct?
Are they consistent between training and serving?
Are they leaking future information?
Can they be computed at production scale?
Can they be monitored?
```

So before we talk about serving XGBoost or Random Forest, we need to understand the **feature pipeline**.

---

# 1. What Is a Feature Pipeline?

A **feature pipeline** is the system that converts raw data into model-ready inputs.

Raw data may look like this:

```text
user clicked ad
user viewed page
URL was crawled
host had 404 errors
publisher category = sports
device = mobile
country = India
timestamp = 2026-09-08 18:00
```

But the model usually cannot use raw logs directly.

The model needs structured feature values like:

```text
historical_ctr_7d = 0.034
user_click_count_30d = 12
publisher_avg_revenue_7d = 43.5
host_error_ratio_10d = 0.18
device_type_mobile = 1
country_india = 1
```

So the feature pipeline does this:

```text
Raw data
   ↓
Cleaning
   ↓
Joining
   ↓
Aggregation
   ↓
Transformation / Encoding
   ↓
Validation
   ↓
Model-ready feature vector
```

### Memory Hook

```text
Feature pipeline = raw data → useful model inputs
```

---

# 2. Why Feature Pipeline Matters More in Production

During offline experiments, you may have a clean CSV file.

Example:

```text
features.csv
labels.csv
```

You train the model and get good validation performance.

But production is different.

In production:

```text
Data arrives continuously.
Some features may be delayed.
Some values may be missing.
Schemas may change.
Logs may break.
Feature distributions may drift.
Real-time features may not match offline features.
```

So a production feature pipeline must be:

- Reliable
- Repeatable
- Consistent
- Scalable
- Monitorable

A model trained on bad or inconsistent features will fail even if the algorithm is strong.

### Important Line

```text
A production ML system is only as good as its feature pipeline.
```

---

# 3. Feature Pipeline in Ensemble Learning

Tree ensembles such as **Random Forest**, **XGBoost**, and **LightGBM** are usually trained on tabular features.

Examples:

```text
numerical features
categorical features
count features
ratio features
time-window features
historical aggregates
behavioral features
risk scores
```

Tree ensembles are very powerful on tabular data because they can learn non-linear rules.

Example:

```text
If host_error_ratio_10d > 0.4
and host_crawl_ratio_10d < 0.1
and serp_dsat_ratio > 0.2
then junk probability is high.
```

But the model can learn this only if the features are correctly built.

So for ensemble learning in production, the feature pipeline is the backbone.

---

# 4. Raw Data Sources

The first part of feature pipeline design is identifying raw data sources.

For an ad-tech CTR prediction system, raw data sources may include:

```text
impression logs
click logs
conversion logs
ad metadata
publisher metadata
user/device context
geography
time of day
auction data
bid price
ad category
```

For a junk URL or host-risk scoring system, raw data sources may include:

```text
crawl logs
index status
HTTP status codes
redirect signals
spam labels
content quality signals
host-level aggregates
click dissatisfaction signals
dwell time
SERP interaction logs
manual judgments
```

For fraud detection, raw sources may include:

```text
transaction logs
user account history
merchant history
device fingerprint
location history
payment metadata
chargeback labels
```

The first production question is:

```text
Where does every feature come from?
```

Every feature should have:

- An owner
- A source
- A refresh schedule
- A quality check

---

# 5. Feature Types

Production features usually fall into multiple categories.

## 5.1 Static Features

Static features do not change often.

Examples:

```text
country
device type
publisher category
account age bucket
domain registration age
ad category
```

These are usually easy to serve.

But they can still become stale.

Example:

```text
publisher category changed from news to sports
but the feature table still says news
```

---

## 5.2 Real-Time Features

Real-time features are computed at request time or near request time.

Examples:

```text
current session click count
current page context
current device
current location
current ad auction features
current query
current URL
```

These are powerful but harder to manage because they must be available with low latency.

### Important Production Question

```text
Can this feature be computed fast enough during inference?
```

---

## 5.3 Batch Aggregated Features

Batch features are computed periodically.

Examples:

```text
user_click_count_7d
publisher_ctr_30d
host_error_ratio_10d
avg_dwell_time_7d
conversion_rate_14d
```

These may be computed:

```text
hourly
daily
weekly
monthly
```

They are common in ensemble systems because tree models work very well with aggregate features.

Example:

```text
host_error_ratio_10d =
number_of_error_urls_in_last_10_days
/
number_of_crawled_urls_in_last_10_days
```

---

## 5.4 Time-Window Features

Time-window features are extremely important in production ML.

Examples:

```text
clicks_last_1h
clicks_last_24h
clicks_last_7d
clicks_last_30d
revenue_last_7d
errors_last_10d
```

Time-window features capture recent behavior.

But they must be designed carefully.

Example problem:

```text
If you train using clicks from after the prediction time,
you are leaking future information.
```

For a prediction at time `T`:

### Correct

```text
Use data from T - 7 days to T
```

### Wrong

```text
Use data from T to T + 7 days
```

The second case introduces **future leakage**.

---

# 6. Feature Freshness

Feature freshness means:

```text
How recent is the feature value?
```

Example:

```text
publisher_ctr_7d was last updated 5 minutes ago
```

versus:

```text
publisher_ctr_7d was last updated 3 days ago
```

For some systems, 3-day-old data may be acceptable.

For others, it may be useless.

Example:

```text
Ad auction scoring may need very fresh features.

Daily host junk scoring may tolerate daily features.

Credit risk scoring may tolerate slower updates.
```

Feature freshness depends on the use case.

Important production questions:

```text
How often is this feature updated?

What is the maximum acceptable staleness?

What happens if the feature is stale?

Should we fallback, block prediction, or use a default value?
```

---

# 7. Feature Availability

A feature may exist during training but not during production.

This is one of the biggest ML production mistakes.

Example:

```text
Training feature:
final_conversion_status

Problem:
At prediction time, we do not know whether conversion will happen.
```

Another example:

```text
Training feature:
total_revenue_after_ad_was_shown

Problem:
Revenue after the ad is shown is future information.
```

Another example:

```text
Training feature:
manual_review_label

Problem:
Manual review happens after model prediction.
```

For every feature, ask:

```text
Is this feature available at prediction time?
```

If not, it should not be used.

This is the foundation of **leakage prevention**.

---

# 8. Training-Serving Skew

Training-serving skew happens when the feature values used during training differ from the feature values used during production.

This is one of the most important production ML concepts.

## Simple Definition

```text
Training-serving skew =
model sees one type of feature during training
and another type during serving
```

Example:

During training:

```text
feature_user_ctr_7d =
computed from clean offline warehouse
```

During production:

```text
feature_user_ctr_7d =
computed from real-time store with delayed clicks
```

The numbers may not match.

The model was trained on one distribution but served on another.

This can cause performance drops.

---

# 9. Example of Training-Serving Skew

Suppose we train a model with this feature:

```text
host_error_ratio_10d
```

Offline training code computes:

```text
hard_errors / total_urls
```

But production serving code computes:

```text
(hard_errors + redirects) / total_urls
```

Now the same feature name means different things.

Training saw one feature definition.

Production uses another feature definition.

The model behavior becomes unreliable.

This is why feature definitions must be shared and versioned.

### Memory Hook

```text
Same feature name does not guarantee same feature meaning.
```

---

# 10. Feature Definition

A production feature should have a precise definition.

Bad feature definition:

```text
host_error_ratio
```

Good feature definition:

```text
host_error_ratio_10d =
(number of URLs on the host with hard error status in last 10 days)
/
(number of URLs crawled for the host in last 10 days)
```

Even better:

```text
Name:
host_error_ratio_10d

Entity:
host

Time window:
last 10 days before prediction timestamp

Numerator:
count of URLs with hard error status

Denominator:
count of crawled URLs

Default:
-1 if denominator is 0

Refresh:
daily at 2 AM

Owner:
crawl quality pipeline

Version:
v1
```

This level of clarity prevents bugs.

---

# 11. Feature Schema

A feature schema defines what features the model expects.

Example:

```yaml
feature_name: host_error_ratio_10d
type: float
allowed_range: 0 to 1
default_value: -1
nullable: false
freshness_requirement: 24 hours
```

Another example:

```yaml
feature_name: device_type
type: categorical
allowed_values:
  - mobile
  - desktop
  - tablet
  - unknown
default_value: unknown
```

A model should not silently accept broken inputs.

Example bad production behavior:

```text
feature expected: float
received: string
```

or:

```text
feature expected range: 0 to 1
received: 183
```

A good production system catches this early.

---

# 12. Feature Store

A **feature store** is a centralized system for storing, computing, and serving features.

It helps maintain consistency between:

```text
offline training
online serving
batch inference
real-time inference
```

A feature store usually provides:

```text
feature definitions
historical feature values
online low-latency feature lookup
offline training feature retrieval
versioning
freshness metadata
ownership
monitoring
```

The main purpose is:

```text
Use the same feature logic in training and serving.
```

This reduces training-serving skew.

---

# 13. Offline Store vs Online Store

Production feature systems often have two stores.

## Offline Feature Store

Used mainly for training.

Usually built on:

```text
data warehouse
data lake
Spark tables
Hive tables
Parquet files
BigQuery / Snowflake / Redshift-like systems
```

It stores historical feature values.

Example:

```text
host_id | timestamp | host_error_ratio_10d | avg_dwell_time_7d
```

This allows historical training data construction.

---

## Online Feature Store

Used for real-time inference.

Usually optimized for low-latency lookup.

Example systems:

```text
Redis-like key-value store
DynamoDB-like store
Cassandra-like store
custom in-memory serving layer
```

It stores the latest feature values.

Example:

```text
key = host_id

value = latest feature vector
```

### Important Question

```text
Are offline and online features computed using the same logic?
```

---

# 14. Point-in-Time Correctness

This is extremely important.

Point-in-time correctness means:

```text
When building training data for prediction time T,
only use feature values that were available before T.
```

Example:

Suppose the model predicts whether a user will click an ad at:

```text
2026-09-08 10:00 AM
```

Then the training row should use only data available before:

```text
2026-09-08 10:00 AM
```

### Wrong

```text
Use clicks from 10:30 AM or later.
```

That leaks future information.

### Correct

```text
Use user behavior until 10:00 AM only.
```

Point-in-time correctness prevents accidental leakage.

### Memory Hook

```text
Training rows must time-travel correctly.
```

---

# 15. Feature Leakage

Feature leakage happens when the model uses information that would not be available at prediction time.

Leakage makes offline performance look amazing while production performance becomes poor.

Examples:

```text
Using future clicks to predict click

Using post-conversion revenue to predict conversion

Using manual review result to predict review outcome

Using label-derived features

Using aggregate features computed from the entire dataset,
including future validation or test rows
```

For ensemble models, leakage can be especially dangerous because trees are very good at exploiting shortcuts.

A tree may quickly learn:

```text
if feature_leaked_label_flag = 1:
    predict positive
```

This gives high offline accuracy but fails in production.

---

# 16. Why Trees Are Sensitive to Leakage

Decision Trees and tree ensembles are very powerful at finding sharp splits.

Example feature:

```text
review_completed = 1
```

If almost all reviewed samples are positive, the tree may split on this feature immediately.

This makes the model look excellent offline.

But in production, this feature may not exist or may be meaningless at prediction time.

So tree ensembles require strict leakage checks.

### Important Line

```text
Tree models are shortcut learners when leakage exists.
```

---

# 17. Missing Features

In production, some features will inevitably be missing.

Reasons include:

```text
new user
new publisher
new host
pipeline delay
lookup failure
logging issue
schema change
network timeout
```

Missing features should be handled intentionally.

Bad approach:

```text
Let missing value crash the prediction service.
```

Better options:

```text
use default value
use unknown category
use model-native missing handling
use fallback model
route to human review
skip prediction
```

XGBoost and LightGBM can handle missing values internally by learning default split directions.

But you still need to monitor missing rates.

For example, if the missing rate suddenly jumps from:

```text
2% → 40%
```

the model may degrade substantially.

---

# 18. Default Values

Default values must be chosen carefully.

Example:

```text
host_error_ratio_10d is missing
```

Possible defaults:

```text
0
-1
mean value
median value
unknown bucket
```

Each has a different meaning.

If you use:

```text
0
```

the model may interpret missing as:

```text
No errors.
```

That may be wrong.

Using:

```text
-1
```

can explicitly tell the tree:

```text
This value is missing.
```

Trees can then learn a separate path for missing values.

But the default must be consistent between training and serving.

---

# 19. Categorical Features

Production systems often contain categorical features.

Examples:

```text
country
device_type
publisher_category
ad_category
browser
domain
host
city
campaign_id
```

Tree models do not always directly understand raw strings.

So categorical features must often be encoded.

Common methods:

```text
one-hot encoding
ordinal encoding
target encoding
frequency encoding
hashing trick
native categorical handling
```

Each has different production tradeoffs.

---

# 20. One-Hot Encoding

One-hot encoding creates one binary feature per category.

Example:

```text
device_type = mobile
```

becomes:

```text
device_mobile = 1
device_desktop = 0
device_tablet = 0
```

This is good for low-cardinality features.

Examples:

```text
device type
browser family
country group
weekday
```

It is usually unsuitable for very high-cardinality features.

Examples:

```text
host_id
user_id
campaign_id
```

because it may create too many columns.

---

# 21. Target Encoding

Target encoding replaces a category with an aggregate target statistic.

Example:

```text
publisher_category = sports
```

becomes:

```text
average CTR for sports category = 0.042
```

This can be powerful.

But it can leak labels if implemented incorrectly.

### Wrong

```text
Use the full dataset target average,
including the current row.
```

### Better

```text
Use out-of-fold target encoding.

For time-based problems,
use only historical data available before the prediction timestamp.
```

Target encoding must therefore be handled carefully in production.

---

# 22. Frequency Encoding

Frequency encoding replaces a category with how often it appears.

Example:

```text
host = example.com
```

becomes:

```text
host_frequency = 120000
```

This can help the model understand:

```text
popularity
volume
rarity
```

But frequency values can drift over time.

Therefore, they require regular refresh and monitoring.

---

# 23. Hashing Trick

Hashing maps categories into a fixed number of buckets.

Example:

```text
host = abc.com
```

may become:

```text
bucket_438
```

Benefits:

```text
fixed feature size
handles unseen categories
useful for high-cardinality features
```

Problems:

```text
hash collisions
reduced interpretability
bucket distribution drift
```

Hashing is common in large-scale ML systems.

---

# 24. Numerical Feature Transformations

Numerical features may need transformations.

Examples:

```text
log transformation
clipping
bucketing
winsorization
normalization
standardization
ratio creation
missing indicators
```

Tree models usually do **not** require feature scaling in the same way as linear models or neural networks.

But transformations can still help.

Example:

```text
total_clicks
```

may range from:

```text
0 to 10,000,000
```

A log transformation can reduce extreme skew:

```text
log(1 + total_clicks)
```

---

# 25. Ratio Features

Tree ensembles often benefit from ratio features.

Examples:

```text
clicks / impressions
errors / crawled_urls
conversions / clicks
revenue / impression
junk_urls / total_urls
```

Ratios are useful because they normalize raw counts by volume.

But ratios require denominator handling.

Example:

```text
ctr = clicks / impressions
```

If:

```text
impressions = 0
```

what should happen?

Possible options:

```text
set CTR to 0
set CTR to -1
set missing flag
use smoothing
```

The choice should be explicit and consistent.

---

# 26. Smoothing Ratio Features

Raw ratios can be noisy when the denominator is small.

Example:

```text
Publisher A:
1 click / 1 impression = CTR 1.0

Publisher B:
100 clicks / 1000 impressions = CTR 0.1
```

Publisher A has a CTR of `1.0`, but it is based on only one impression.

That estimate is unreliable.

Smoothing helps.

A simple smoothed CTR is:

```text
smoothed_ctr =
(clicks + alpha * global_ctr)
/
(impressions + alpha)
```

Where:

```text
alpha = smoothing strength
global_ctr = average CTR across all data
```

This pulls low-volume estimates toward the global average.

### Important Idea

```text
Small denominator ratios should not be trusted too much.
```

---

# 27. Feature Validation

Before features reach the model, they should be validated.

Checks include:

```text
type check
range check
null check
missing rate check
freshness check
distribution check
allowed category check
schema compatibility check
```

Example:

```text
host_error_ratio_10d should be between 0 and 1
```

If the pipeline suddenly produces:

```text
host_error_ratio_10d = 27
```

something is clearly wrong.

Feature validation catches pipeline bugs before they damage predictions.

---

# 28. Feature Monitoring

After deployment, monitor features continuously.

Important metrics include:

```text
mean
median
min
max
standard deviation
percentiles
missing rate
zero rate
cardinality
top categories
freshness
distribution drift
```

Example alert:

```text
Feature avg_dwell_time_7d missing rate increased from 3% to 45%.
```

This could indicate:

```text
logging broke
join failed
upstream pipeline delayed
schema changed
```

Without monitoring, the model may silently fail.

---

# 29. Feature Ownership

Every production feature should have an owner.

Why?

Because when a feature breaks, someone must understand and fix it.

A feature should have:

```text
owner
definition
source table
refresh frequency
quality checks
business meaning
deprecation plan
```

Without ownership, old features become dangerous.

Example:

```text
A feature was created two years ago.

Nobody knows what it means.

The upstream table changed.

The model still uses it.
```

This creates production risk.

---

# 30. Feature Backfilling

Backfilling means recomputing historical feature values.

Example:

```text
We created a new feature:

host_error_ratio_10d

We need historical values for the past 6 months
to train the model.
```

So we run the feature logic over historical data.

But backfilling must remain point-in-time correct.

### Wrong

```text
Use future information while recreating historical rows.
```

### Correct

```text
For each historical timestamp T,
compute features only using data available before T.
```

Backfilling is one of the common places where leakage enters.

---

# 31. Feature Versioning

Feature definitions change over time.

Example `v1`:

```text
host_error_ratio_10d =
hard_errors / crawled_urls
```

Example `v2`:

```text
host_error_ratio_10d =
(hard_errors + soft_errors + redirects) / crawled_urls
```

These are different features.

If the definition changes while the name remains the same, models become difficult to debug.

Better options:

```text
host_error_ratio_10d_v1
host_error_ratio_10d_v2
```

or maintain version metadata in a feature registry.

### Important Line

```text
Changing feature logic is changing the model input distribution.
```

---

# 32. Feature Pipeline for Batch Scoring

For batch scoring, the pipeline usually looks like this:

```text
Raw logs
   ↓
Daily/hourly aggregation jobs
   ↓
Feature table
   ↓
Batch model scoring job
   ↓
Prediction output table
   ↓
Downstream system / dashboard / review queue
```

Example for host junk scoring:

```text
Crawl logs + spam labels + click signals
   ↓
Compute host-level features
   ↓
Load XGBoost model
   ↓
Generate junk risk score per host
   ↓
Rank hosts by risk
   ↓
Send top hosts to review/action system
```

Batch pipelines can tolerate more latency than real-time systems, but correctness is still critical.

---

# 33. Feature Pipeline for Real-Time Scoring

For real-time scoring, the pipeline may look like this:

```text
Request arrives
   ↓
Extract request features
   ↓
Fetch online features
   ↓
Apply transformations
   ↓
Build feature vector
   ↓
Call model
   ↓
Return prediction
```

Example for CTR prediction:

```text
Ad request arrives
   ↓
Get user/device/context features
   ↓
Fetch publisher/ad historical features
   ↓
Score candidate ads
   ↓
Rank ads by expected value
   ↓
Return selected ad
```

Real-time pipelines must care deeply about latency.

---

# 34. Latency in Feature Pipelines

In real-time systems, feature retrieval can dominate latency.

Example latency budget:

```text
Total allowed latency = 50 ms

Feature lookup = 20 ms
Model scoring = 10 ms
Business logic = 10 ms
Network overhead = 10 ms
```

If feature lookup suddenly becomes:

```text
80 ms
```

the whole latency budget is violated.

Production systems therefore use techniques such as:

```text
caching
online feature stores
precomputed features
fallback features
parallel lookups
timeout handling
```

### Important Line

```text
A powerful feature is useless if it cannot be served within the latency budget.
```

---

# 35. Fallback Logic

Production systems need fallback mechanisms.

Example failure:

```text
Feature store timeout
```

Possible responses:

```text
use cached value
use default value
use simpler fallback model
skip model and use rule-based logic
route to safe default action
```

The correct fallback depends on the application risk.

Examples:

### Ad Ranking

```text
Fallback to previous stable model
or heuristic ranking.
```

### Fraud Detection

```text
Fallback to manual review
for high-risk transactions.
```

### Junk Detection

```text
Avoid automatic destructive action
and route uncertain cases to review.
```

Fallback logic is part of system design, not an afterthought.

---

# 36. Feature Pipeline Failure Modes

Common failures include:

```text
upstream logs delayed
join keys changed
schema changed
feature missing rate increased
categorical values changed
timestamp bug
timezone bug
duplicate records
backfill leakage
training-serving skew
online feature store timeout
stale features
wrong default value
```

These failures can silently damage model performance.

---

# 37. Designing a Good Feature Pipeline

A good production feature pipeline should satisfy:

```text
correctness
freshness
consistency
scalability
monitorability
explainability
versioning
fault tolerance
low latency if real-time
point-in-time correctness
```

For every feature, ask:

```text
What does it mean?

Where does it come from?

When is it updated?

Is it available at prediction time?

Can it leak future information?

How is it computed offline?

How is it served online?

What is its default value?

How do we monitor it?

Who owns it?
```

---

# 38. Case Study: Host Junk Risk Feature Pipeline

Suppose we want to build a host-level ensemble model that predicts:

```text
Probability that a host is junk-heavy
```

Possible raw sources:

```text
crawl logs
index status
spam/crushlist signals
SERP dissatisfaction clicks
dwell time
manual judgments
host metadata
redirect signals
HTTP error signals
```

Possible features:

```text
host_total_urls
host_crawled_urls_10d
host_crawl_ratio_10d
host_error_ratio_10d
host_redirect_ratio_10d
host_junk_ratio_10d
host_dsat_click_ratio_30d
host_serp_dsat_ratio_30d
host_avg_dwell_time_30d
host_static_rank
```

Feature vector:

```text
[
    host_total_urls,
    host_crawl_ratio_10d,
    host_error_ratio_10d,
    host_redirect_ratio_10d,
    host_junk_ratio_10d,
    host_dsat_click_ratio_30d,
    host_serp_dsat_ratio_30d,
    host_avg_dwell_time_30d,
    host_static_rank
]
```

Possible model:

```text
XGBoost / Random Forest
```

Output:

```text
junk_probability_score
```

Possible downstream use:

```text
rank hosts
send top hosts to review
trigger deeper crawling/analysis
monitor junk ecosystem
```

---

# 39. Case Study: CTR Prediction Feature Pipeline

For ad-tech CTR prediction:

Prediction target:

```text
Will the user click this ad impression?
```

Raw sources:

```text
impression logs
click logs
user context
publisher metadata
ad metadata
auction metadata
device/browser/geography
time features
```

Possible features:

```text
user_ctr_7d
ad_ctr_7d
publisher_ctr_7d
campaign_ctr_7d
device_type
country
hour_of_day
day_of_week
ad_category
publisher_category
bid_price
historical_conversion_rate
```

Possible model:

```text
Gradient Boosted Trees / XGBoost / LightGBM
```

Output:

```text
click_probability
```

Business use:

```text
expected_value = click_probability × bid_value
```

Ads can then be ranked by expected value.

Important production issues:

```text
real-time latency
delayed click labels
feedback loops
cold-start users/ads
calibration
A/B testing
```

---

# 40. Final Mental Model

For ensemble learning in production, do not think:

```text
I trained XGBoost and deployed it.
```

Think:

```text
I designed a system that continuously transforms raw production data
into reliable features,
feeds those features into an ensemble model,
produces predictions,
monitors input and output quality,
and handles failures safely.
```

That is the difference between **ML modeling** and **production ML system design**.

---

# Part 3 Summary

```text
Feature Pipeline Design

Goal:
Convert raw production data into reliable model-ready features.

Key concerns:
feature correctness
feature freshness
feature availability
training-serving consistency
point-in-time correctness
leakage prevention
missing values
categorical encoding
feature validation
feature monitoring
feature versioning
feature ownership

Most important risks:
data leakage
training-serving skew
stale features
missing features
schema changes
broken joins

Production memory line:
A model is only as reliable as the features it receives.
```

---

# Next Topic

```text
Part 5 — Batch Inference System
```

Batch scoring is usually easier to understand before moving to real-time inference.
