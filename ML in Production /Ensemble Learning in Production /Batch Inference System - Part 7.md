# Batch Inference System - Part 7

# Ensemble Learning in Production

# Part 5.14 — Case Study: Daily Host Junk Risk Scoring

# Part 5.15 — Case Study: Daily User/Ad CTR Scoring

This is the final detailed-study block for:

```text
Part 5 — Batch Inference System
```

We now connect the entire batch inference architecture to two complete production case studies.

The goal is not to introduce completely new concepts. The goal is to combine everything we studied into realistic end-to-end systems.

---

# 5.14 Case Study — Daily Host Junk Risk Scoring

## 5.14.1 Problem Statement

Suppose we operate a web search system and want to estimate the junk risk of every active host.

Examples of junk signals may include:

```text
HTTP hard errors
soft-404 behavior
parked domains
DNS failures
spam signals
redirect-heavy behavior
user dissatisfaction
low-quality engagement
stale or poorly crawled hosts
```

We do not want to classify only individual URLs.

Instead, we want to answer:

```text
How risky is this host overall?
```

The output will be used for:

```text
crawl prioritization
host investigation
manual review
index-quality workflows
candidate suppression
quality dashboards
```

Because host-level risk usually does not need millisecond freshness, this is a strong batch inference use case.

---

## 5.14.2 Why Batch Inference Fits This Problem

Imagine we have:

```text
20 million active hosts
```

A host's risk score is based mostly on historical aggregates such as:

```text
last 10 days crawl statistics
30-day dissatisfaction signals
spam counts
error rates
redirect ratios
index coverage
```

These features change over hours or days rather than milliseconds.

So serving the model on every downstream request would create unnecessary real-time complexity.

Instead:

```text
score hosts once per day
store the result
reuse it throughout the day
```

This gives us:

```text
lower serving cost
simpler downstream systems
ability to use expensive aggregate features
ability to score the full host population
```

---

## 5.14.3 Define the Scoring Entity

The entity is:

```text
host
```

Examples:

```text
example.com
news.example.com
shop.example.org
```

The first production requirement is a canonical host identifier.

We must normalize things like:

```text
case
www prefix
subdomain policy
punycode / international domains
trailing dots
host aliases
```

If host identity is inconsistent, feature joins and prediction history become unreliable.

So the pipeline begins with a stable:

```text
host_id
```

---

## 5.14.4 Build the Daily Scoring Population

We probably do not score every host ever seen.

Instead, we construct an eligible population.

Example criteria:

```text
host active in last 30 days
host has minimum crawl evidence
host is not permanently blocked already
host belongs to supported traffic scope
host has valid canonical identifier
```

Suppose the raw universe contains:

```text
120 million known hosts
```

but only:

```text
20 million active eligible hosts
```

are scored daily.

This reduces cost and avoids scoring irrelevant entities.

---

## 5.14.5 Feature Groups

A realistic host junk model might use feature groups such as:

### Crawl / error features

```text
RecentCrushedUrls
Last_10_Days_CrawledUrls
Hard_Error_Urls
Redirected_Urls
SR_Marked_Urls
```

Derived examples:

```text
R_junk10d = RecentCrushedUrls / Last_10_Days_CrawledUrls

R_error =
(SR_Marked_Urls + Redirected_Urls + Hard_Error_Urls)
/
Last_10_Days_CrawledUrls
```

### Crawl coverage / staleness

```text
crawlRatio = Last_10_Days_CrawledUrls / TotalUrlCountIndexProbe
R_stale = 1 - crawlRatio
```

### User dissatisfaction

```text
dsatClickRatio
serpDsatClickRatio
```

### Engagement

```text
avgDwellTimeSeconds
```

### Spam / quality signals

```text
spamRatio
spam counts
host reputation signals
```

The important point is that features come from multiple systems.

So the feature join layer is critical.

---

## 5.14.6 Build the Model-Ready Feature Snapshot

Suppose the daily scoring timestamp is:

```text
2026-09-16 02:00
```

The system constructs one feature row per host using only information available up to that timestamp.

Example:

```text
host_id     error_ratio   crawl_ratio   dsat_ratio   spam_ratio   avg_dwell
A.com       0.42          0.18          0.25         0.11         8.4
B.com       0.03          0.91          0.01         0.00         48.2
```

Before scoring, validate:

```text
one row per host
expected row count
feature ranges
missing rates
feature freshness
feature schema version
join coverage
```

This is where many real-world failures are caught.

---

## 5.14.7 Sparse / Low-Evidence Hosts

Some hosts may have very little evidence.

Example:

```text
1 crawled URL
1 error
```

Then:

```text
error_ratio = 1.0
```

But that is not equivalent to:

```text
100,000 errors / 100,000 crawled URLs
```

Both ratios equal 1.0, but confidence is very different.

So the model should also receive support signals such as:

```text
crawled_url_count
total_url_count
click_count
judgment_count
```

or use smoothed features.

A useful design principle is:

```text
Ratios tell us direction.
Counts tell us evidence strength.
```

---

## 5.14.8 Load the Production Model Contract

Suppose the approved model is:

```text
host_junk_xgboost_v17
```

The worker loads:

```text
model artifact
feature schema
feature order
missing-value policy
calibration model
threshold policy
class mapping
model checksum
```

The job must not simply load:

```text
latest model
```

because reproducibility matters.

Every partition must use the same model version and checksum.

---

## 5.14.9 Distributed Scoring

Suppose:

```text
20 million hosts
200 partitions
```

A simple partition rule could be:

```text
partition_id = hash(host_id) % 200
```

Each partition contains roughly:

```text
100,000 hosts
```

Worker flow:

```text
receive partition
↓
load model bundle once
↓
validate schema
↓
score host batches
↓
write temporary partition output
↓
report metrics
```

Because each host can usually be scored independently, this parallelizes naturally.

Current Azure ML documentation similarly describes batch endpoints as appropriate when inference runs over large amounts of data, low latency is not required, and parallelization is useful.

---

## 5.14.10 Prediction Output

A host prediction record might contain:

```text
host_id
junk_score
calibrated_junk_probability
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

Example:

```text
abc.com
junk_score = 0.91
risk_bucket = very_high
model_version = v17
data_quality_status = healthy
```

The output is not just a score.

It is a reusable production data product.

---

## 5.14.11 Decision Policy

The model output should remain separate from the business action.

Example:

```text
score < 0.50
    → no action

0.50–0.80
    → monitor

0.80–0.95
    → manual review

>= 0.95
    → eligible for stronger intervention
```

These thresholds are illustrative.

The key architecture point is:

```text
model score
        ≠
business action
```

The policy layer can change without retraining the model.

---

## 5.14.12 Incremental Scoring

Not every host may need to be rescored every day.

Candidate triggers might be:

```text
new crawl activity
new error signals
new DSAT data
new spam evidence
score expiry
newly discovered host
late-arriving data
```

Suppose only:

```text
1.8 million of 20 million hosts
```

have relevant changes.

We can score those 1.8 million and reuse valid predictions for the rest.

However, if:

```text
model version changes
feature definition changes materially
```

we may run a full 20 million-host refresh.

A practical pattern could be:

```text
daily incremental scoring
weekly full refresh
```

---

## 5.14.13 Failure Handling

Consider a few failure scenarios.

### DSAT pipeline missing

If DSAT is important but not critical:

```text
score using trained missing-value behavior
mark output degraded
restrict automated actions
```

If DSAT is mandatory:

```text
block publish
```

### One partition fails

```text
retry partition
```

If retry repeatedly fails:

```text
block publication
keep previous successful score table active
```

### Model checksum mismatch

```text
stop scoring immediately
```

### Output row count = 17M instead of 20M

```text
do not publish
```

The principle is always:

```text
Partial or untrusted output should not silently become production output.
```

---

## 5.14.14 Monitoring

Monitor several layers.

### Job health

```text
runtime
SLA
failed partitions
retries
```

### Population health

```text
host count
new host count
duplicate hosts
```

### Feature health

```text
crawl coverage
DSAT coverage
spam coverage
missing rates
feature freshness
feature distributions
```

### Model health

```text
model version
checksum
score throughput
```

### Output health

```text
prediction count
null-score rate
duplicate prediction rate
mean score
p50/p95/p99 scores
risk-bucket distribution
```

### Business impact

```text
hosts entering review
hosts auto-actioned
review precision
false-positive feedback
```

---

## 5.14.15 End-to-End Architecture

The complete host junk batch pipeline now looks like:

```text
Raw Crawl / Click / Spam / Index Data
                ↓
Daily Feature Pipelines
                ↓
Eligible Host Population
                ↓
Point-in-Time Feature Joins
                ↓
Feature Validation
                ↓
Approved Model Bundle
                ↓
Distributed Batch Scoring
                ↓
Temporary Prediction Partitions
                ↓
Output Validation
                ↓
Atomic Publish
                ↓
Current Host Risk Snapshot
                ↓
Review / Crawl / Quality Systems
                ↓
Monitoring + Feedback
```

This is the full batch-inference architecture we have been building throughout Part 5.

---

# 5.14 Mental Model

For a host-risk problem, think:

```text
aggregate slow-changing evidence
→ score the host population periodically
→ publish a validated reusable risk snapshot
```

Memory line:

```text
The model is only one stage; the production product is the validated host-risk snapshot consumed by downstream systems.
```

---

# 5.15 Case Study — Daily User/Ad CTR Scoring

Now consider a very different application.

CTR means:

```text
Click-Through Rate
```

A model estimates:

```text
P(click | user, ad, context)
```

At first glance this sounds like a real-time problem.

And often it is.

But batch inference can still play an important role.

This case study teaches when batch scoring and real-time scoring should be combined.

---

## 5.15.1 Why CTR Can Use Batch Scoring

A fully real-time CTR model may use:

```text
current user
current ad
current page
current time
current device
recent session behavior
```

Those request-specific signals require online inference.

But many CTR features change slowly.

Examples:

```text
user long-term click propensity
advertiser quality
campaign historical CTR
publisher quality
user-category affinity
ad historical performance
```

These can be computed in batch.

So one production architecture is:

```text
batch model / batch features
        +
real-time request features
        ↓
online final ranking
```

This is a hybrid system.

---

## 5.15.2 Two Possible Batch CTR Designs

### Design A — Batch score entities

Daily compute:

```text
user_quality_score
ad_quality_score
campaign_ctr_prior
publisher_quality_score
```

Online serving combines these with live request context.

### Design B — Batch score user-ad candidates

For a smaller candidate universe, precompute:

```text
user_id + ad_id → CTR score
```

Then store the top candidates or scores in a low-latency store.

This is useful when:

```text
candidate set is stable enough
scores can tolerate some staleness
real-time compute is expensive
```

---

## 5.15.3 Why Full User × Ad Scoring Is Usually Impossible

Suppose:

```text
100 million users
10 million ads
```

The full cross-product is:

```text
100,000,000 × 10,000,000
```

which is enormous.

So we do not score every user against every ad.

Instead:

```text
candidate generation
→ score plausible pairs only
```

For example:

```text
user interests
campaign eligibility
geography
language
budget
category
publisher constraints
```

may reduce the space dramatically.

This is an important systems lesson:

```text
Batch scoring scale is often controlled by candidate generation before ML scoring begins.
```

---

## 5.15.4 Define the Entity

Depending on architecture, the scoring entity may be:

```text
user
ad
campaign
publisher
user-ad pair
user-category pair
```

For pair scoring:

```text
entity_id = (user_id, ad_id)
```

For user profile scoring:

```text
entity_id = user_id
```

This decision affects:

```text
feature joins
partitioning
storage
prediction reuse
freshness
cost
```

---

## 5.15.5 Feature Groups

Possible CTR feature groups include:

### User features

```text
historical CTR
recent click count
category affinity
device preference
active days
```

### Ad features

```text
historical CTR
impression count
ad age
creative category
advertiser quality
```

### Campaign features

```text
campaign CTR
budget utilization
campaign age
conversion history
```

### Interaction features

```text
user-category affinity
user-ad exposure count
user-ad historical clicks
publisher-ad compatibility
```

Batch features typically represent historical aggregates.

Live context such as:

```text
current query
current page
current hour
current session
```

may remain in the real-time layer.

---

## 5.15.6 Sparse CTR Data

CTR data is naturally sparse.

Most impressions are:

```text
not clicked
```

and many user-ad pairs have never appeared before.

So the system faces:

```text
class imbalance
cold-start users
cold-start ads
low-support rates
missing interaction history
```

A pair with:

```text
1 click / 1 impression
```

should not necessarily receive the same confidence as:

```text
10,000 clicks / 10,000 impressions
```

Again:

```text
rate + support count
```

is often more informative than rate alone.

---

## 5.15.7 Candidate Generation Before Scoring

Suppose each user has 500 eligible ad candidates after business filtering.

For:

```text
100 million users
```

that is still:

```text
50 billion user-ad pairs
```

So further candidate reduction may be needed.

Possible stages:

```text
eligibility filtering
retrieval / approximate matching
campaign constraints
historical relevance
budget filters
```

Then ML scoring is applied only to the reduced candidate set.

This keeps batch scoring tractable.

---

## 5.15.8 Model Artifact and Scoring

Suppose the approved model is:

```text
ctr_lightgbm_v42
```

The model bundle includes:

```text
feature order
categorical mappings
missing-value behavior
calibration
model version
schema version
```

The candidate table may contain billions of rows, so distributed scoring becomes important.

Partitioning might use:

```text
hash(user_id)
```

Why user-based partitioning?

Because all candidate ads for a user can stay together.

That can make downstream top-K selection easier.

---

## 5.15.9 Score Candidates Then Keep Top-K

For recommendation or ad ranking, we may not need to store every scored pair.

Suppose a user has:

```text
500 candidate ads
```

After scoring, maybe we keep only:

```text
top 50
```

Then storage changes from:

```text
all user-ad scores
```

into:

```text
user → top candidate ads + scores
```

This dramatically reduces output size.

Example:

```text
user_123
    ad_81   0.091
    ad_14   0.083
    ad_77   0.079
    ...
```

The online system can then retrieve these candidates quickly.

---

## 5.15.10 Batch Score as a Prior, Not Final Decision

A useful hybrid approach is:

```text
batch CTR score = prior
```

Then online serving adjusts using live signals.

Conceptually:

```text
final_score =
f(batch_score, live_context, business_constraints)
```

The exact function may be another model or ranking rule.

For example:

```text
batch_score = user-ad long-term relevance
```

Online model adds:

```text
current query
current page
current time
recent session behavior
```

This balances cost and freshness.

---

## 5.15.11 Freshness Is More Important Than in Host Junk Scoring

CTR behavior can change quickly.

Examples:

```text
new campaign launches
ad performance changes
user intent changes
budget gets exhausted
viral event occurs
```

So a once-daily score may be too stale for the final decision.

This is why batch CTR systems often use:

```text
batch priors + online corrections
```

or more frequent batch refreshes.

This case study shows an important principle:

```text
The acceptable batch interval depends on how quickly the target changes.
```

---

## 5.15.12 Incremental CTR Scoring

Incremental rescoring is very useful because the candidate space can be huge.

Rescore when:

```text
user activity changes
ad performance changes
new ad appears
campaign changes
score expires
new interaction history arrives
```

For example:

```text
new ad created
```

We do not need to recompute every existing user-ad score.

We may only score:

```text
eligible users × new ad
```

Similarly, if one user becomes active again:

```text
rescore that user's candidate set
```

This can save enormous compute.

---

## 5.15.13 Failure Handling in CTR Batch Scoring

Suppose one campaign-feature pipeline fails.

Possible actions:

```text
exclude affected campaigns
use fallback priors
reuse previous batch scores
mark candidates stale
```

If the whole scoring run fails:

```text
previous candidate list may remain active temporarily
```

But stale CTR scores can degrade ranking faster than stale host-risk scores.

So the maximum fallback age may be shorter.

This shows that failure policy depends on target dynamics.

---

## 5.15.14 Monitoring CTR Batch Output

Monitor:

```text
number of users scored
candidate pairs scored
mean candidates per user
top-K coverage
CTR score distribution
cold-start rate
missing feature rate
campaign coverage
publisher coverage
model version
throughput
job runtime
fallback rate
```

Business monitoring may include:

```text
click-through rate
conversion rate
revenue per impression
ad coverage
candidate diversity
```

Technical health can be normal while business metrics degrade.

So both matter.

---

## 5.15.15 End-to-End Hybrid Architecture

A realistic architecture could be:

```text
Historical Impression / Click Logs
              ↓
Batch Feature Pipelines
              ↓
User / Ad / Campaign Feature Tables
              ↓
Candidate Generation
              ↓
Batch Ensemble Scoring
              ↓
Top-K Candidates per User
              ↓
Online Key-Value Store
              ↓
User Request
              ↓
Retrieve Precomputed Candidates
              ↓
Add Real-Time Context
              ↓
Online Re-ranking
              ↓
Final Ads
```

This design uses batch inference where it is economically efficient and real-time inference where freshness matters.

---

# 5.15 Mental Model

CTR teaches us that batch and real-time inference are not mutually exclusive.

The best production architecture may be:

```text
batch for expensive slow-changing signals
+
real-time for fresh request-specific signals
```

Memory line:

```text
Use batch scoring to precompute what changes slowly, and reserve real-time inference for what truly needs to change at request time.
```

---

# Comparing the Two Case Studies

## Host Junk Risk

Typical characteristics:

```text
entity = host
features = historical aggregates
freshness requirement = hours/daily
batch suitability = very high
output reused broadly
```

Architecture:

```text
Daily feature snapshot
→ host scoring
→ validated host-risk table
→ downstream quality systems
```

## CTR

Typical characteristics:

```text
entity = user/ad/candidate pair
features = historical + live context
freshness requirement = often seconds/minutes for final ranking
batch suitability = partial / hybrid
candidate space = potentially enormous
```

Architecture:

```text
Batch historical scores / candidate generation
+
real-time context
→ final ranking
```

The important distinction is not the algorithm.

Both systems may use:

```text
XGBoost
LightGBM
Random Forest
```

The architecture changes because:

```text
target dynamics
latency requirements
candidate scale
data freshness
business action
```

are different.

---

# Final Batch Inference Mental Model

Across all of Part 5, the complete system is:

```text
Raw Data
    ↓
Entity Selection
    ↓
Feature Computation
    ↓
Point-in-Time Feature Joins
    ↓
Input Validation
    ↓
Model Contract Loading
    ↓
Distributed / Incremental Scoring
    ↓
Prediction Output Design
    ↓
Output Validation
    ↓
Atomic Publishing
    ↓
Downstream Consumption
    ↓
Monitoring + Failure Handling
```

The central lesson is:

```text
Batch inference is not model.predict() on a large dataset.
```

It is a production data-and-ML system that must guarantee:

```text
correct inputs
correct model
scalable scoring
traceable outputs
safe publishing
recoverability
monitoring
```

---

# Final Memory Lines

```text
A production batch prediction is a validated, versioned, reusable data product.
```

```text
Use batch inference when the prediction can be computed ahead of time and safely reused.
```

```text
Use real-time inference only for signals or decisions whose freshness truly requires request-time computation.
```

```text
The model is one component; the real system is everything required to make its predictions reliable at scale.
```

---

# References

- Microsoft Azure Machine Learning documentation — Batch model deployments and batch endpoints. Current documentation describes batch endpoints as suitable for large datasets, expensive inference, low-latency-insensitive workloads, and parallel execution.
- Feast documentation — historical feature retrieval and point-in-time correctness concepts.
- XGBoost documentation — production model loading/prediction behavior concepts.

---

This completes the detailed-study portion of:

```text
Part 5 — Batch Inference System
```

The next step in our pattern is the separate interview-style version for:

```text
5.14 Daily Host Junk Risk Scoring
5.15 Daily User/Ad CTR Scoring
```
