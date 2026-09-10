# Ensemble Learning in Production

# Part 5 — Batch Inference System

## Deep Version: 5.1 to 5.3

We are studying:

```text
5.1 What is batch inference?
5.2 Why batch inference exists
5.3 Batch inference vs real-time inference
```

The earlier version gave us the overview. Now we will build a deeper **production-level understanding**.

---

# 5.1 What Is Batch Inference?

Batch inference means:

```text
The model is not called for one live request at a time.

Instead, the model is run on a large collection of records together,
and the predictions are stored for later use.
```

The system does not wait for a user, request, or event to arrive and then score it immediately.

Instead, it says:

```text
I already know the set of entities I care about.

Let me compute their scores in advance.

Then downstream systems can use those scores whenever needed.
```

The word **entity** is important.

An entity can be:

```text
user
host
URL
publisher
advertiser
product
customer
transaction
campaign
document
search query
```

A batch inference system usually scores many entities at once.

Examples:

```text
Score all hosts once per day.

Score all users every night.

Score all products every 6 hours.

Score all publishers every hour.

Score all URLs after a crawl cycle.
```

The output of batch inference is usually a table or file containing predictions.

Example:

| host_id | junk_score | model_version | scoring_date |
|---|---:|---|---|
| abc.com | 0.91 | xgb_v12 | 2026-09-09 |
| xyz.net | 0.76 | xgb_v12 | 2026-09-09 |
| goodsite.com | 0.04 | xgb_v12 | 2026-09-09 |

This output is then consumed by another system.

For example:

```text
review dashboard
ranking pipeline
alerting system
human review queue
blocking pipeline
recommendation system
ad-serving feature lookup
```

### Key Idea

```text
Batch inference separates prediction generation from prediction consumption.
```

Prediction generation happens in one scheduled or data-triggered job.

Prediction consumption happens later, possibly many times.

---

# A Simple Mental Model

Think of batch inference like preparing food in advance.

Real-time inference is similar to:

```text
Customer orders food.
        ↓
Kitchen cooks immediately.
        ↓
Customer waits.
```

Batch inference is similar to:

```text
Kitchen prepares many meals in advance.
        ↓
Meals are stored.
        ↓
Customer arrives.
        ↓
Prepared food is served quickly.
```

In ML terms:

```text
Batch inference precomputes predictions
so production systems do not need to compute them from scratch every time.
```

This is why batch inference is common when predictions can be reused.

---

# Concrete Example: Host Junk Risk Scoring

Suppose we are building an ensemble model to predict whether a host is junk-heavy.

The entity is:

```text
host
```

The model output is:

```text
junk_probability
```

Input features may include:

```text
host_error_ratio_10d
host_redirect_ratio_10d
host_crawl_ratio_10d
host_dsat_click_ratio_30d
host_avg_dwell_time_30d
host_spam_ratio_10d
host_total_urls
```

Now ask:

```text
Do I need to score this host in real time
every time somebody sees a URL from the host?
```

Probably not.

Host quality usually changes over hours or days rather than milliseconds.

Therefore, we can run a daily batch job.

```text
Every day at 2 AM:

1. Get all active hosts.
2. Compute host-level features.
3. Load the trained XGBoost model.
4. Score every host.
5. Write host_id → junk_score to an output table.
6. Let downstream systems consume that table.
```

The overall architecture may look like:

```text
Crawl Logs
Spam Labels
Click Signals
Dwell Time Logs
Manual Judgments
        ↓
Feature Aggregation Job
        ↓
Host Feature Table
        ↓
Batch Scoring Job
        ↓
XGBoost / Random Forest
        ↓
Host Junk Score Table
        ↓
Review / Ranking / Action Systems
```

This is **batch inference**.

The model may score millions of hosts, but nobody is waiting for an immediate response.

The important production requirement is therefore not:

```text
Can this prediction return within 20 milliseconds?
```

Instead, the important requirement may be:

```text
Can the entire scoring job finish correctly
before the downstream system needs the scores?
```

For example:

```text
The host score table must be ready every morning by 6 AM.
```

That becomes the **batch inference SLA**.

---

# 5.1.1 Batch Inference Has Its Own SLA

In real-time inference, the SLA is often based on latency per request.

Example:

```text
Prediction must return within 50 ms.
```

In batch inference, the SLA is usually based on **job completion time**.

Example:

```text
Daily scoring job must finish before 6 AM.
```

Instead of optimizing primarily for one-request latency, we optimize for:

```text
total job runtime
data correctness
output completeness
failure recovery
cost
prediction freshness
```

Suppose:

```text
Batch job starts = 2 AM

Scores required = 6 AM
```

The system has a:

```text
4-hour processing window
```

Within this window, the pipeline may need to:

```text
read input data
join features
perform aggregations
load model
score entities
write predictions
validate outputs
publish results
```

If the job finishes at:

```text
7 AM
```

the predictions may be mathematically correct, but the production system has still violated its contract.

### Important Distinction

```text
Real-time SLA → latency per request

Batch SLA → completion of the entire job before a deadline
```

---

# 5.1.2 Batch Inference Is Usually More Data-Heavy Than Model-Heavy

In many production systems, the expensive or complicated part is not the model prediction itself.

The difficult part is often:

```text
collecting input entities
joining features
reading large tables
handling missing data
deduplicating records
computing aggregations
writing outputs
validating results
```

Suppose we want to score:

```text
100 million hosts
```

Model scoring itself may be relatively fast.

But feature preparation may involve:

```text
joining crawl logs
joining click logs
joining spam labels
aggregating 10-day windows
aggregating 30-day windows
removing duplicates
handling missing host IDs
computing ratios
validating features
```

This can be much more expensive than simply calling:

```python
predictions = model.predict(features)
```

Therefore, production batch inference is not just:

```text
Run model on CSV.
```

A better mental model is:

```text
Batch inference
=
Data Engineering
+
Feature Correctness
+
Model Scoring
+
Output Validation
+
Output Publishing
```

---

# 5.1.3 Batch Inference Produces Stored Predictions

This is another major distinction.

In real-time inference, a prediction may exist only for the duration of one request.

In batch inference, predictions are usually written somewhere.

Common destinations include:

```text
database table
Parquet files
Hive table
data lake path
key-value store
search index
feature store
dashboard table
queue for downstream processing
```

A basic output table may contain:

```text
entity_id
score
prediction_label
score_timestamp
model_version
feature_version
run_id
confidence
explanation_fields
```

A good production batch system does not store only the score.

It also stores **metadata**.

Why?

Because later someone may ask:

```text
Which model produced this score?

When was this score generated?

Which feature pipeline version was used?

Was this score generated by today's run
or yesterday's fallback run?

Were important features missing?

Which policy used this score?
```

Without metadata, debugging becomes difficult.

A more production-ready output may look like:

```text
host_id: abc.com

junk_score: 0.91

risk_bucket: high

model_version: xgb_host_junk_v12

feature_pipeline_version: host_features_v5

scoring_time: 2026-09-09 02:43:12

run_id: daily_20260909

missing_feature_count: 0

action_policy_version: policy_v3
```

This is production-level thinking.

---

# 5.2 Why Batch Inference Exists

Batch inference exists because many predictions do not need to be made at the exact moment they are consumed.

This is the core reason.

Many entities change slowly enough that precomputed scores are acceptable.

Examples:

```text
host quality
publisher quality
customer churn risk
credit risk
product demand forecast
daily fraud risk
content quality score
user lifetime value
advertiser risk score
campaign health score
```

These predictions may still be useful even if they are several minutes or hours old.

Instead of building a complex real-time model-serving architecture, we can precompute them.

---

# 5.2.1 Reason 1: The Prediction Target Changes Slowly

Some targets do not change second by second.

Consider **host junk risk**.

A host with:

```text
years of low-quality content
high error ratio
spam signals
poor user behavior
many redirects
low-quality pages
```

will probably not become a high-quality host within five seconds.

Therefore:

```text
daily scoring
```

or perhaps:

```text
hourly scoring
```

may be sufficient.

Another example is **customer churn risk**.

Churn risk may depend on long-term signals such as:

```text
usage drop
billing issues
support tickets
low engagement
missed payments
```

The score usually does not need to be recomputed after every user click.

Another example is **publisher quality**.

Publisher quality may depend on:

```text
historical CTR
invalid traffic rate
complaints
revenue consistency
content category
policy violations
```

These signals are often aggregated over days or weeks.

Therefore, batch scoring is natural.

### Production Principle

```text
If the underlying entity changes slowly,
the prediction can often be precomputed.
```

---

# 5.2.2 Reason 2: Batch Inference Reduces Real-Time Complexity

Real-time ML systems are difficult to build reliably.

A real-time ML service may require:

```text
low-latency feature lookup
model server
autoscaling
timeouts
fallback logic
high availability
monitoring
request tracing
dependency management
```

Batch inference avoids many of these problems.

Instead of doing this during every request:

```text
Request arrives
        ↓
Fetch 50 features
        ↓
Join multiple online stores
        ↓
Build feature vector
        ↓
Run large ensemble
        ↓
Return prediction within 30 ms
```

we can do the expensive work earlier:

```text
Compute features offline
        ↓
Score entity offline
        ↓
Store final prediction
```

Then the live system becomes:

```text
Request arrives
        ↓
Lookup precomputed score
        ↓
Use score in business logic
```

This is much simpler and faster.

### Memory Hook

```text
Batch inference moves expensive computation away from request time.
```

---

# 5.2.3 Reason 3: Batch Inference Is Cost Efficient

Batch jobs can often be optimized for throughput.

Instead of keeping a model-serving system running continuously, we can run a scheduled job.

Example:

```text
Run scoring job every night.

Use a temporary compute cluster.

Process all entities together.

Write predictions.

Shut down resources after completion.
```

This may be cheaper than maintaining a highly available real-time inference service.

Batch inference is especially attractive when:

```text
prediction volume is huge
predictions are not needed immediately
models are computationally expensive
features require heavy aggregation
scores can be reused many times
```

Example:

Suppose we need:

```text
1 host score per day
```

for:

```text
100 million hosts
```

If each score is reused many times throughout the day, precomputing it can be far more efficient than recomputing it on every request.

### Production Idea

```text
Compute once, reuse many times.
```

---

# 5.2.4 Reason 4: Batch Inference Handles Large Population Scoring

Sometimes the goal is not to respond to a live user request.

The goal is to scan an entire population.

Examples:

```text
Find the top 10,000 risky hosts out of 100 million.

Find users most likely to churn.

Find products likely to go out of stock.

Find advertisers showing suspicious behavior.

Find URLs that need recrawling.

Find publishers requiring manual investigation.
```

These are often **ranking or prioritization problems**.

Batch inference is ideal because we can:

```text
Score entire population
        ↓
Sort by predicted risk/value
        ↓
Select top candidates
        ↓
Send to downstream action
```

Example:

```text
Score all hosts
        ↓
Sort by junk_probability descending
        ↓
Select top 50,000
        ↓
Send to review/action pipeline
```

Real-time inference is not naturally suited to this because there may be no live request corresponding to every entity.

Batch inference lets the ML system proactively scan the population.

---

# 5.2.5 Reason 5: Batch Inference Supports Human Review Workflows

Many production ML systems do not immediately take automatic action.

Instead, they create prioritized review queues.

Examples:

```text
High-risk host
    ↓
Reviewer investigates

Suspicious transaction
    ↓
Fraud analyst reviews

Potential policy violation
    ↓
Moderation team reviews

Bad publisher traffic
    ↓
Quality team investigates
```

These workflows may operate hourly or daily.

Batch inference fits naturally:

```text
Daily Score Generation
        ↓
Rank Cases by Risk
        ↓
Create Review Queue
        ↓
Human Review
        ↓
Reviewer Decisions
        ↓
New Training Labels
        ↓
Future Model Training
```

This creates a powerful **feedback loop**.

```text
Model prioritizes cases
        ↓
Humans investigate cases
        ↓
Human decisions become labels
        ↓
Model improves
```

This pattern is extremely common in production ML.

---

# 5.2.6 Reason 6: Batch Inference Enables More Expensive Models

Real-time systems have strict latency constraints.

Suppose an XGBoost model contains:

```text
1000 trees
deep trees
hundreds of features
complex preprocessing
```

It may not fit comfortably within:

```text
20 ms
```

of online latency.

But the same model may be perfectly acceptable in a batch system.

If the entire batch job has:

```text
4 hours
```

to finish, the system can afford more computation.

Therefore:

```text
Batch inference gives more freedom in model complexity.

Real-time inference imposes stricter latency constraints.
```

---

# 5.3 Batch Inference vs Real-Time Inference

The difference is not simply:

```text
scheduled vs immediate
```

The entire system architecture changes.

---

# 5.3.1 Trigger

Batch inference is usually triggered by:

```text
schedule
or
data availability
```

Examples:

```text
Every day at 2 AM

Every hour

After crawl logs arrive

After labels are refreshed

After the feature table is updated
```

Real-time inference is triggered by a live event.

Examples:

```text
user opens an app

ad request arrives

payment transaction occurs

search query is submitted

recommendation page loads
```

Therefore:

```text
Batch
=
time-triggered or data-triggered

Real-time
=
request-triggered or event-triggered
```

This changes reliability requirements.

If a batch job fails, we may have time to retry.

If a real-time request fails, the system usually needs an immediate fallback.

---

# 5.3.2 Latency Requirement

Batch inference latency is measured at the **job level**.

Example:

```text
The entire job must finish within 3 hours.
```

Real-time latency is measured at the **request level**.

Example:

```text
The model must respond within 30 ms.
```

This changes optimization priorities.

## Batch Optimization

Batch systems focus on:

```text
throughput
parallelism
distributed processing
efficient file reads
efficient writes
job completion time
cluster resource utilization
```

## Real-Time Optimization

Real-time systems focus on:

```text
low latency
p95 latency
p99 latency
warm model loading
caching
online feature lookup speed
timeout handling
high availability
```

### Important Distinction

```text
Batch cares about finishing all work by a deadline.

Real-time cares about responding to each request quickly.
```

---

# 5.3.3 Feature Availability

Batch inference usually uses offline or batch-computed features.

Examples:

```text
aggregated logs
daily tables
historical windows
warehouse features
```

Real-time inference needs features available immediately.

Examples:

```text
current session features
request context
latest user state
online feature store values
```

The feature pipeline complexity therefore differs.

Batch systems can perform expensive operations such as:

```text
Spark joins
large aggregations
window calculations
historical backfills
```

Real-time features need low-latency retrieval from systems such as:

```text
key-value stores
caches
online feature stores
in-memory databases
precomputed aggregates
```

A feature that is easy to compute in batch may be impractical to compute directly during a live request.

Example:

```text
average dwell time over the last 30 days for this host
```

This is easy to compute through an offline aggregation.

But calculating it live from raw logs would be expensive.

Therefore, a real-time system may instead retrieve a **precomputed version** of this feature.

---

# 5.3.4 Output Storage

Batch inference usually stores predictions.

Real-time inference usually returns predictions directly to the caller.

## Batch Outputs

```text
prediction table
score file
risk queue
ranked list
feature store entry
dashboard table
```

## Real-Time Outputs

```text
API response
ranking decision
fraud approve/deny decision
ad selection
recommendation result
```

Because batch predictions are stored, their lifecycle must be managed.

Important questions include:

```text
How long are scores valid?

Should old scores be overwritten?

Should score history be retained?

Can downstream systems read partial outputs?

How do we prevent consumers from seeing an incomplete run?

When should a new run become visible?
```

This leads to an important production concept:

## Atomic Publishing

We do not want downstream systems to read a half-written output table.

A safer pattern is:

```text
Run scoring job
        ↓
Write to temporary output location
        ↓
Validate row counts and schema
        ↓
Validate score distributions
        ↓
Mark run as successful
        ↓
Publish / swap pointer
        ↓
Downstream systems see new scores
```

In other words:

```text
write
↓
validate
↓
publish
```

rather than:

```text
write directly into production table while consumers are reading it
```

This prevents downstream systems from consuming incomplete results.

---

# 5.3.5 Failure Handling

Batch failures and real-time failures are handled differently.

## Batch Failure

Suppose the daily scoring job fails.

Possible responses include:

```text
retry job
use yesterday's scores
run partial scoring
alert the owning team
block downstream publishing
fallback to previous model version
```

If scores are not extremely time-sensitive, yesterday's scores may still be useful.

Example:

```text
Yesterday's host quality score
may be better than having no score.
```

But this should be explicit.

Metadata could indicate:

```text
score_date = 2026-09-08
freshness_status = stale
```

---

## Real-Time Failure

Suppose online model scoring fails.

Possible responses include:

```text
return default score
use cached score
fallback to rules
fallback to simpler model
skip personalization
send case to manual review
allow or deny based on risk policy
```

Real-time failures require an immediate decision.

There is no opportunity to debug the issue while the request waits.

---

# 5.3.6 Freshness and Staleness

Batch predictions can become stale.

Example:

```text
Score generated = 2 AM
Score consumed = 8 PM

Score age = 18 hours
```

Whether this is acceptable depends on the task.

For slow-moving problems:

```text
daily churn score
daily host quality score
weekly credit risk
```

some staleness may be acceptable.

For fast-moving problems:

```text
fraud attack detection
breaking news ranking
real-time bidding
current user session intent
```

stale predictions may be dangerous.

Therefore, every batch prediction should ideally have:

```text
scoring timestamp
validity window
freshness SLA
fallback policy
```

Example policy:

```python
if score_age <= 24_hours:
    use_score()
else:
    use_fallback()
```

### Memory Hook

```text
A batch prediction is not timeless.

It has an expiry.
```

---

# 5.3.7 Cost Model

Batch and real-time inference have different cost structures.

## Batch Costs

```text
cluster compute during scheduled jobs
large table reads
large table writes
storage for predictions
job orchestration
```

## Real-Time Costs

```text
always-on serving infrastructure
online feature store
low-latency compute
autoscaling
high availability
real-time monitoring
```

Batch inference can often be cheaper when:

```text
scores are reused many times
model does not require immediate updates
jobs can run periodically
features are expensive to compute
```

Real-time inference is justified when:

```text
live context strongly affects prediction
decision cannot be delayed
freshness is critical
request-specific information matters
```

---

# 5.3.8 Ensemble Model Implication

Now let us connect this specifically to ensemble learning.

Tree ensembles such as:

```text
Random Forest
XGBoost
LightGBM
```

may contain:

```text
hundreds of trees
thousands of trees
trees with depth 4, 6, 8, or more
many input features
```

Prediction requires traversing many trees.

Conceptually:

```text
Input row
   ↓
Tree 1 → score
Tree 2 → score
Tree 3 → score
...
Tree N → score
   ↓
Combine tree outputs
   ↓
Final prediction
```

For batch inference, this works well because scoring can be parallelized across rows.

Example:

```text
Worker 1:
rows 1 → 1,000,000

Worker 2:
rows 1,000,001 → 2,000,000

Worker 3:
rows 2,000,001 → 3,000,000

Worker 4:
rows 3,000,001 → 4,000,000
```

Each worker can independently score its partition.

For real-time inference, however, each request may need to traverse all relevant trees immediately.

This creates stricter latency constraints.

### Production Implication

```text
Batch inference is generally more tolerant
of large and computationally expensive ensembles.
```

---

# 5.3.9 Summary Table

| Dimension | Batch Inference | Real-Time Inference |
|---|---|---|
| **Trigger** | Schedule or data arrival | Live request or event |
| **Latency** | Job completion deadline | Per-request latency |
| **Feature Source** | Offline/batch tables | Online features/cache |
| **Output** | Stored predictions | Immediate response |
| **Failure Handling** | Retry or use previous scores | Immediate fallback required |
| **Freshness Risk** | Scores can become stale | Usually fresher |
| **Cost Pattern** | Periodic compute | Always-on serving |
| **Best For** | Slow-changing entities | Fast-changing decisions |
| **Model Complexity** | Can tolerate larger models | Usually requires latency optimization |
| **Primary Optimization** | Throughput | Latency |
| **Typical Scale Unit** | Millions of entities per job | Individual requests |
| **Prediction Reuse** | Often reused many times | Often request-specific |

---

# Deep Example: Choosing Batch vs Real-Time

Suppose we are building an ad-tech system.

We want to predict:

```text
Probability that a user clicks an ad.
```

Some features are slow-changing:

```text
publisher_ctr_7d
campaign_ctr_7d
advertiser_quality_score
domain_safety_score
```

These can be batch-computed.

Other features are request-specific:

```text
current page
current ad candidate
current time
current device
current user session
auction price
```

These must be available at request time.

Therefore, the final architecture may be **hybrid**.

```text
Historical Logs
        ↓
Batch Feature Computation
        ↓
Precomputed Historical Features
        ↓
Online Feature Store
        ↓
                         Live Request
                              ↓
                    Request-Time Features
                              ↓
Precomputed Features ───────→ Join
                              ↓
                         Online Model
                              ↓
                    Click Probability
                              ↓
                         Ad Ranking
```

This pattern is very common.

Production ML is often neither purely batch nor purely real-time.

It may use:

```text
Batch feature computation
+
Real-time model scoring
```

or:

```text
Batch model scoring
+
Real-time score lookup
```

For example:

```text
Batch:
Compute publisher quality score daily.

Real-time:
Use publisher quality score during ad ranking.
```

This hybrid design is extremely important in production ML.

---

# Deep Example: Host Junk Risk Batch System

Let us design a daily host junk-risk scoring system.

## Step 1: Define the Entity

```text
Entity = host
```

Examples:

```text
example.com
abc.net
news-site.org
```

---

## Step 2: Define the Prediction

```text
Prediction =
Probability that the host is junk-heavy
```

---

## Step 3: Define the Scoring Frequency

```text
Daily
```

Why daily?

Because host-level quality generally does not need millisecond-level updates.

---

## Step 4: Prepare Features

Possible features:

```text
host_error_ratio_10d
host_redirect_ratio_10d
host_crawl_ratio_10d
host_spam_ratio_10d
host_dsat_ratio_30d
host_avg_dwell_time_30d
host_total_url_count
```

---

## Step 5: Run the Model

```text
Load XGBoost model
        ↓
Score all eligible hosts
        ↓
Generate junk_probability
```

---

## Step 6: Store the Output

Possible schema:

```text
host_id
junk_probability
risk_bucket
model_version
feature_version
scoring_date
run_id
```

---

## Step 7: Use the Output

Example policy:

```python
if score >= 0.90:
    action = "High risk → urgent review / safe automated action"

elif 0.70 <= score < 0.90:
    action = "Medium risk → human review"

else:
    action = "No action"
```

Notice that the model produces a **score**.

A separate policy layer decides what action to take.

Conceptually:

```text
Model
   ↓
junk_probability
   ↓
Policy / Threshold Layer
   ↓
Business Action
```

This separation is important because thresholds and business policies may change without retraining the model.

---

## Step 8: Monitor the Job

Important metrics include:

```text
number of hosts expected
number of hosts scored
percentage successfully scored
missing feature rate
score distribution
top-risk hosts
job runtime
failed partitions
input data freshness
output data freshness
```

Example checks:

```text
Expected hosts = 100M
Scored hosts = 99.9M

Missing host_error_ratio_10d = 1.3%

Previous average junk score = 0.21
Today's average junk score = 0.52
```

A sudden jump in average score may indicate:

```text
real ecosystem change
feature pipeline bug
schema change
upstream data issue
model issue
```

---

## Step 9: Define the Fallback

If today's job fails:

```text
Use yesterday's scores
        ↓
Mark scores as stale
        ↓
Alert owner
        ↓
Disable risky automated actions if score age exceeds threshold
```

Example:

```python
if score_age <= 24_hours:
    use_score()

elif score_age <= 48_hours:
    use_score_but_disable_high_risk_auto_actions()

else:
    use_safe_fallback()
```

This creates a more resilient batch inference system.

---

# End-to-End Host Junk Batch Architecture

```text
                RAW DATA SOURCES
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Crawl Logs      Spam Signals    Click Signals
        │              │              │
        └──────────────┼──────────────┘
                       ↓
             Feature Aggregation
                       ↓
               Feature Validation
                       ↓
                Host Feature Table
                       ↓
                  Batch Scorer
                       ↓
                  XGBoost Model
                       ↓
              Host Junk Probability
                       ↓
               Output Validation
                       ↓
                Atomic Publishing
                       ↓
                Production Score Table
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
 Human Review      Ranking System    Action System
```

---

# The Most Important Concept

Batch inference is not simply:

```text
offline prediction
```

It is a production system with explicit contracts.

---

# Batch Inference Contracts

## 1. Input Contract

```text
Which entities should be scored?
```

Example:

```text
All active hosts with at least one crawl in the last 30 days.
```

---

## 2. Feature Contract

```text
Which features are required?

What are their definitions?

How fresh must they be?

What happens when they are missing?
```

---

## 3. Model Contract

```text
Which model version should be used?
```

Example:

```text
model_version = xgb_host_junk_v12
```

---

## 4. Output Contract

```text
Where are predictions written?

What schema do they follow?

What metadata must be present?
```

---

## 5. Freshness Contract

```text
How old can the predictions become
before they should no longer be trusted?
```

---

## 6. SLA Contract

```text
When must the scoring job finish?
```

Example:

```text
Start: 2 AM
Deadline: 6 AM
```

---

## 7. Failure Contract

```text
What happens if:

feature generation fails?
model scoring fails?
some partitions fail?
output validation fails?
the entire run fails?
```

---

## 8. Consumer Contract

```text
Which downstream systems consume these scores?

What do they expect?

Can they use stale scores?

What happens when today's score is unavailable?
```

---

# Batch Inference Production Mental Model

Instead of thinking:

```text
Load model
↓
Run predict()
↓
Done
```

Think:

```text
Identify Entities
        ↓
Load / Compute Features
        ↓
Validate Features
        ↓
Load Correct Model Version
        ↓
Score Entities
        ↓
Validate Predictions
        ↓
Attach Metadata
        ↓
Write Output
        ↓
Validate Output
        ↓
Atomically Publish
        ↓
Downstream Consumption
        ↓
Monitor
        ↓
Fallback if Required
```

That is a production batch inference system.

---

# Final Summary

Batch inference means computing predictions for many entities together and storing the results for later use.

It exists because many predictions do not require immediate request-time computation.

It is especially useful when:

```text
the underlying entity changes slowly
scores can be reused
the population is very large
features require expensive aggregation
the model is expensive
human review queues need to be generated
predictions only need periodic refreshes
```

The main production challenge is not merely running the model.

It is ensuring that:

```text
the correct entities are scored
features are correct
features are fresh
the correct model version is used
all expected entities receive predictions
outputs are validated
results are published safely
failures have defined fallbacks
downstream systems understand score freshness
```

### Strongest Memory Hook

```text
Batch inference precomputes predictions
so downstream systems can use them later
without running the model live.
```

### Another Useful Memory Hook

```text
Real-time inference optimizes for latency.

Batch inference optimizes for throughput,
correctness, freshness, and deadline completion.
```

### Production Mental Model

```text
Batch Inference
=
Large-Scale Data Pipeline
+
Feature Pipeline
+
Model Scoring
+
Prediction Storage
+
Validation
+
Publishing
+
Monitoring
+
Failure Recovery
```

---

# Next Topics

```text
5.4 When ensemble models are suitable for batch scoring

5.5 End-to-end batch scoring architecture
```
````
