# Batch Inference System - Part 4

# Ensemble Learning in Production

# Part 5.8 — Loading the Ensemble Model

# Part 5.9 — Distributed Scoring

This note continues:

```text
Part 5 — Batch Inference System
```

Previously covered:

```text
5.1 What is batch inference?
5.2 Why batch inference exists
5.3 Batch inference vs real-time inference
5.4 When ensemble models are suitable for batch scoring
5.5 End-to-end batch scoring architecture
5.6 Input Data Preparation
5.7 Feature Joins
```

Now we study:

```text
5.8 Loading the Ensemble Model
5.9 Distributed Scoring
```

At this point, the system has already created a model-ready feature table.

Example:

```text
host_id      scoring_time          error_ratio_10d    crawl_ratio_10d    dsat_ratio_30d    static_rank
abc.com      2026-09-10 02:00      0.42               0.18               0.25              4.7
xyz.net      2026-09-10 02:00      0.03               0.91               0.01              8.9
pqr.org      2026-09-10 02:00      0.76               0.10               0.44              3.1
```

Now the next two questions are:

```text
Which model should score this table?

How do we score this table at production scale?
```

That is what Part 5.8 and Part 5.9 are about.

---

# 5.8 Loading the Ensemble Model

## 5.8.1 Why Model Loading Is a Separate Production Step

In a notebook, model loading looks simple:

```python
import pickle

model = pickle.load(open("model.pkl", "rb"))
predictions = model.predict(X_test)
```

But production batch inference is not that simple.

In production, before scoring, the system must answer:

```text
Which exact model version should be used?

Where is the model stored?

Is the model artifact complete?

Does the model match the feature table?

Does the model require preprocessing?

Does the model output raw score, probability, calibrated probability, or class label?

Are all workers loading the same model?

Can we reproduce this score later?
```

This is why model loading is not just:

```text
read model file from disk
```

It is more like:

```text
load the full inference contract
```

This is the most important idea.

---

## 5.8.2 What Is a Model Artifact?

A model artifact is the saved object produced after training.

For a basic experiment, this could be:

```text
random_forest.pkl
xgboost_model.json
lightgbm_model.txt
catboost_model.cbm
```

But in production, the model artifact should not only mean the raw model.

A production model artifact should contain, or be linked to, the complete information needed to run inference correctly.

That includes:

```text
trained model
feature names
feature order
feature data types
preprocessing logic
categorical encoders
missing-value handling
calibration model
threshold policy
model version
training data version
feature pipeline version
library/runtime version
evaluation metrics
owner/team metadata
creation timestamp
checksum/hash
```

Why is this necessary?

Because the model alone does not fully define prediction behavior.

Suppose you trained an XGBoost model using this feature order:

```text
[
    host_error_ratio_10d,
    host_crawl_ratio_10d,
    host_dsat_ratio_30d,
    host_static_rank
]
```

But during batch inference, the scoring job passes:

```text
[
    host_crawl_ratio_10d,
    host_error_ratio_10d,
    host_dsat_ratio_30d,
    host_static_rank
]
```

The model may still produce a score.

But it will be wrong.

The model does not understand column names unless the serving system enforces them. It mostly receives numbers in positions. If position 1 used to mean `error_ratio`, and now position 1 means `crawl_ratio`, the model's logic becomes corrupted.

So the model artifact must capture the contract between:

```text
training pipeline
        and
inference pipeline
```

---

## 5.8.3 The Ensemble Model Is Not Only Many Trees

For ensemble learning, we often say:

```text
Random Forest = many decision trees
XGBoost = many boosted trees
LightGBM = many boosted trees
```

That is true algorithmically.

But in production, the inference system is usually bigger than the trees.

A more realistic prediction flow is:

```text
raw features
    ↓
feature validation
    ↓
missing-value handling
    ↓
categorical encoding
    ↓
feature ordering
    ↓
ensemble model
    ↓
raw model score
    ↓
probability conversion
    ↓
probability calibration
    ↓
thresholding / business rule
    ↓
final prediction output
```

Example:

```text
Model raw margin = 2.31
Sigmoid probability = 0.91
Calibrated probability = 0.84
Threshold policy:
    if calibrated_probability >= 0.80:
        risk_bucket = high_risk
```

Now imagine we accidentally skip the calibration step.

Yesterday, downstream systems received:

```text
score = calibrated probability
```

Today, they receive:

```text
score = raw model probability
```

Both may look like normal numbers between 0 and 1.

But their meaning is different.

This is dangerous.

A downstream rule like:

```text
if score > 0.80:
    send to review
```

may behave differently depending on whether `score` means raw probability or calibrated probability.

So production model loading must load not only the model but also the score meaning.

---

## 5.8.4 Model Versioning

In production, this is a bad pattern:

```text
load latest model
```

Why?

Because “latest” is unstable.

Today:

```text
latest = model_v17
```

Tomorrow, someone registers:

```text
latest = model_v18
```

Now the same batch inference job may produce different scores without any code change.

That makes debugging difficult.

A safer pattern is:

```text
model_name = host_junk_xgboost
model_version = 17
```

or:

```text
model_alias = production
resolved_model_version = 17
```

The important part is that the scoring job should log the resolved model version.

Every output should be traceable.

Example output:

```text
host_id      junk_score      model_name          model_version      scoring_run_id
abc.com      0.91            host_junk_xgboost   v17                2026_09_10_daily
xyz.net      0.08            host_junk_xgboost   v17                2026_09_10_daily
```

This helps answer later:

```text
Why did abc.com get score 0.91 on September 10?
```

The answer should not be:

```text
I think it was the latest model.
```

The answer should be:

```text
It was scored by host_junk_xgboost version 17,
using feature pipeline version 9,
during scoring run 2026_09_10_daily.
```

That is production traceability.

---

## 5.8.5 Feature Schema Compatibility

Before the model scores anything, the system must verify:

```text
Does the input table match what this model expects?
```

This is called schema compatibility.

The expected schema may include:

```text
feature name
feature order
data type
allowed range
missing-value representation
categorical encoding
feature version
```

Example expected schema:

```text
feature_name              type       valid_range
host_error_ratio_10d      float      0 to 1
host_crawl_ratio_10d      float      0 to 1
host_dsat_ratio_30d       float      0 to 1
host_static_rank          float      positive
```

Now suppose production sends:

```text
host_error_ratio_10d = "0.42"
```

as a string instead of a float.

Or sends:

```text
host_error_ratio_10d = 42
```

instead of:

```text
host_error_ratio_10d = 0.42
```

Or sends:

```text
host_error_ratio_10d = null
```

without applying the expected missing-value policy.

All of these can break or corrupt inference.

The dangerous case is not always a crash.

The dangerous case is:

```text
the model scores successfully,
but the score is wrong.
```

---

## 5.8.6 Feature Order Mismatch

Feature order mismatch deserves special attention.

Suppose the model expects:

```text
[
    error_ratio,
    crawl_ratio,
    dsat_ratio
]
```

But the inference job passes:

```text
[
    crawl_ratio,
    error_ratio,
    dsat_ratio
]
```

The model may not know this happened.

For a tree ensemble, this is especially harmful.

Imagine one tree learned:

```text
if error_ratio > 0.35:
    go right
else:
    go left
```

But now the model receives `crawl_ratio` in the `error_ratio` position.

So the tree is effectively doing:

```text
if crawl_ratio > 0.35:
    go right
else:
    go left
```

That is not the learned logic.

This is why a production inference job should not just rely on column position casually.

It should validate:

```text
input feature names match expected feature names
input feature order matches expected feature order
no extra feature is passed accidentally
no required feature is missing
```

A useful production rule:

```text
Never let the model silently infer feature meaning from position unless the position has been validated.
```

---

## 5.8.7 Preprocessing Must Be Loaded With the Model

Many models are not trained directly on raw feature columns.

They may require preprocessing.

Examples:

```text
median imputation
missing indicator creation
one-hot encoding
target encoding
frequency encoding
log transformation
standardization
winsorization
categorical mapping
feature bucketing
```

Suppose training used:

```text
missing host_error_ratio_10d = -1
host_error_ratio_10d_missing = 1
```

But inference uses:

```text
missing host_error_ratio_10d = 0
```

Now the model receives a different meaning.

This is training-serving skew.

Similarly, suppose training encoded categories like this:

```text
news = 1
shopping = 2
adult = 3
unknown = 0
```

But inference encodes them as:

```text
news = 2
shopping = 3
adult = 1
unknown = 0
```

The model will misunderstand the category.

So model loading should also load the preprocessing logic.

The correct production artifact is usually not:

```text
model.pkl
```

It is more like:

```text
model_bundle/
    model
    feature_schema.json
    preprocessing_config.json
    categorical_encoder.json
    imputation_values.json
    calibration_model
    threshold_policy.json
    metadata.json
```

---

## 5.8.8 Missing-Value Policy Must Be Part of the Artifact

From Part 5.6, we discussed:

```text
missing is not the same as zero
```

Now connect that to model loading.

The model artifact should define how missing values were handled during training.

Example metadata:

```text
host_error_ratio_10d:
    valid_range: 0 to 1
    missing_strategy: sentinel
    missing_value: -1
    missing_indicator: host_error_ratio_10d_missing

host_static_rank:
    missing_strategy: median_imputation
    imputation_value: 4.73
```

Why store this?

Because inference must reuse the exact same policy.

Bad:

```text
Training used median from training data.
Inference recomputes median from today's scoring batch.
```

Why bad?

Because today's median changes every day.

Then the model input meaning changes every day.

Correct:

```text
Training computes median = 4.73.
Artifact stores median = 4.73.
Inference always uses 4.73 for that model version.
```

If you train model version 18 later, it may have a new median. That is fine because model v18 carries its own artifact contract.

---

## 5.8.9 Class-Label Mapping

This is a subtle but important production issue.

For classification models, the model may output probabilities in a fixed class order.

Example:

```python
model.classes_
```

could be:

```text
[0, 1]
```

Meaning:

```text
probability column 0 = not_junk
probability column 1 = junk
```

But in another model, labels may be ordered differently:

```text
["junk", "not_junk"]
```

Now probability column 0 means junk.

If the scoring system assumes the wrong column, it may invert the meaning.

Example:

```text
model output = [0.92, 0.08]
```

If class order is `[not_junk, junk]`, then:

```text
junk_probability = 0.08
```

If class order is `[junk, not_junk]`, then:

```text
junk_probability = 0.92
```

Huge difference.

So the artifact must clearly define:

```text
positive_class
class_order
score_column_used
```

For junk scoring, it should explicitly say:

```text
positive_class = junk
score = probability_of_junk
```

Never assume the second probability column is always the positive class unless the artifact confirms it.

---

## 5.8.10 XGBoost Model Loading

For XGBoost, the model may be saved in formats like JSON or UBJSON.

A simplified flow:

```python
import xgboost as xgb

booster = xgb.Booster()
booster.load_model("model.json")
```

But again, this only loads the booster.

The production system still needs:

```text
feature schema
feature order
preprocessing
missing-value rules
calibration
threshold policy
metadata
```

XGBoost can handle missing values, but you still need to ensure that the inference representation matches training.

For example:

```text
NaN means missing
0 means observed zero
```

If production accidentally converts missing values to 0, the model receives a different signal.

---

## 5.8.11 Random Forest Model Loading

For Random Forest, a saved artifact usually stores many fitted decision trees.

Each tree contains:

```text
split feature index
split threshold
left child
right child
leaf prediction
class counts or regression value
```

The forest prediction is then:

```text
classification:
    aggregate votes or average class probabilities

regression:
    average tree outputs
```

But Random Forest models are often saved using framework-specific serialization.

Example:

```text
scikit-learn pickle/joblib artifact
```

The risk is runtime compatibility.

If the training environment used one library version and production uses another, loading may fail or behave unexpectedly.

So production systems often pin:

```text
Python version
library version
model framework version
dependency environment
```

This is why the model artifact should record runtime metadata.

---

## 5.8.12 LightGBM Model Loading

LightGBM models may be saved as booster files.

Conceptually, the production concerns are the same:

```text
load the trained booster
validate feature names/order
apply the same preprocessing
preserve missing-value behavior
apply calibration if required
log the version
```

LightGBM is often used for large tabular data because it is efficient, but efficiency does not remove production risks.

A fast model can still produce wrong scores if:

```text
features are misordered
categorical encoding changes
missing values are represented differently
wrong model version is loaded
```

---

## 5.8.13 Loading Model Once Per Worker

A bad batch inference implementation is:

```python
for row in rows:
    model = load_model("model.pkl")
    score = model.predict(row)
```

This is inefficient.

Model loading may involve:

```text
reading from remote storage
deserializing trees
loading preprocessing objects
allocating memory
initializing runtime
```

Doing this per row wastes huge time.

A better pattern:

```python
model_bundle = load_model_bundle("model_v17")

for batch in batches:
    scores = model_bundle.predict(batch)
```

In distributed scoring:

```text
Worker starts
    ↓
loads model once
    ↓
scores many rows
    ↓
writes output
```

So model loading happens:

```text
once per worker
```

or:

```text
once per executor process
```

not once per row.

---

## 5.8.14 Same Model on Every Worker

In distributed scoring, many workers score different partitions.

A critical requirement:

```text
Every worker must load the same model version.
```

Bad situation:

```text
Worker 1 loads model_v17
Worker 2 loads model_v17
Worker 3 loads model_v18
Worker 4 loads model_v17
```

Now the output table is inconsistent.

Some rows were scored by one model, others by another.

This may be hard to notice unless we log model version per partition.

So each worker should write metadata:

```text
partition_id
model_name
model_version
model_checksum
feature_schema_version
scoring_run_id
```

Then validation can check:

```text
all partitions used same model_version
all partitions used same model_checksum
```

The checksum/hash is useful because even if two files have the same version name, the checksum confirms that the actual bytes are identical.

---

## 5.8.15 Model-Loading Failure Modes

Model-loading failures can be divided into two types:

```text
hard failures
silent failures
```

### Hard failures

These are obvious.

Examples:

```text
model file not found
permission denied
artifact store unavailable
corrupt model file
incompatible runtime
missing dependency
insufficient memory
```

The job usually crashes.

These are annoying, but easier to detect.

### Silent failures

These are more dangerous.

Examples:

```text
wrong model version loaded
feature order mismatch
preprocessing skipped
calibration skipped
wrong class-label mapping
wrong missing-value handling
wrong categorical encoder
different workers load different models
threshold policy mismatch
```

The job may still finish successfully.

But the predictions are wrong.

Production memory line:

```text
A successful scoring job is not the same as a correct scoring job.
```

---

# 5.8 Final Mental Model

Think of model loading as loading a sealed contract.

The contract says:

```text
This exact model version
expects these exact features
in this exact order
with these exact preprocessing rules
and these exact missing-value rules
and produces this exact kind of score.
```

So the production system should not ask only:

```text
Did the model file load?
```

It should ask:

```text
Did we load the correct inference contract?
```

---

# 5.9 Distributed Scoring

Now assume the model has been loaded correctly.

The next problem is scale.

In a notebook, we may score:

```text
10,000 rows
```

But in production, batch inference may need to score:

```text
10 million hosts
100 million URLs
1 billion user-item pairs
billions of ad candidates
```

A single machine may be too slow.

A single machine may not have enough memory.

So we use distributed scoring.

---

## 5.9.1 What Is Distributed Scoring?

Distributed scoring means:

```text
split the model-ready input data into partitions
score those partitions in parallel
combine the outputs safely
```

Conceptually:

```text
Model-ready feature table
        ↓
Partition input data
        ↓
Worker 1 scores partition 1
Worker 2 scores partition 2
Worker 3 scores partition 3
...
Worker N scores partition N
        ↓
Write partitioned outputs
        ↓
Validate all outputs
        ↓
Publish final prediction table
```

Each worker uses the same trained model but different data.

---

## 5.9.2 Why Distributed Scoring Is Different From Distributed Training

This distinction is important.

### Distributed training

In distributed training:

```text
the model is still learning
```

Workers may need to communicate:

```text
gradients
parameters
tree split statistics
model updates
```

The model changes during the job.

### Distributed scoring

In distributed scoring:

```text
the model is already trained
```

Workers do not update the model.

They only apply it.

Each worker needs:

```text
same model
same feature schema
different input rows
```

So distributed scoring is usually simpler than distributed training.

The main challenge is not learning.

The main challenge is:

```text
data partitioning
model consistency
throughput
fault tolerance
output correctness
```

---

## 5.9.3 Why Ensemble Scoring Is Naturally Parallel

For most ensemble inference, rows are independent.

Example:

```text
score(host_A) does not depend on score(host_B)
```

The model is fixed.

Each row goes through the trees and gets a score.

So we can split rows across workers.

This is often called:

```text
embarrassingly parallel
```

Meaning:

```text
the task can be parallelized with very little communication between workers
```

Example:

```text
100 million hosts
100 workers
≈ 1 million hosts per worker
```

Each worker can score its assigned hosts independently.

This is why batch scoring scales well when the input table is partitioned properly.

---

## 5.9.4 What Exactly Happens When a Tree Ensemble Scores One Row?

Let us make the internal process clear.

Suppose you have an XGBoost model with 500 trees.

For one row:

```text
host_error_ratio_10d = 0.42
host_crawl_ratio_10d = 0.18
host_dsat_ratio_30d = 0.25
host_static_rank = 4.7
```

The row goes through tree 1:

```text
tree_1 output = 0.08
```

Then tree 2:

```text
tree_2 output = 0.03
```

Then tree 3:

```text
tree_3 output = -0.02
```

And so on.

For gradient boosting:

```text
final_raw_score = base_score + sum(tree_outputs)
```

Then we may convert it to probability:

```text
probability = sigmoid(final_raw_score)
```

For Random Forest classification:

```text
tree_1 votes junk
tree_2 votes not_junk
tree_3 votes junk
...
final probability = fraction of trees voting junk
```

For Random Forest regression:

```text
final prediction = average of tree predictions
```

This scoring process is repeated for every row.

The important production point:

```text
Each row can be scored independently.
```

That is why distributed scoring works.

---

## 5.9.5 Partitioning the Input Data

To distribute scoring, we need to split the input table.

This is called partitioning.

Common partitioning strategies:

```text
partition by date
partition by entity hash
partition by region
partition by customer segment
partition by file block
```

### Hash partitioning

A common approach:

```text
partition_id = hash(entity_id) % num_partitions
```

Example:

```text
partition_id = hash(host_id) % 100
```

If we have:

```text
100 million hosts
100 partitions
```

then each partition may have around:

```text
1 million hosts
```

This is good because work is relatively balanced.

### Date partitioning

If we score daily data, we may partition by date:

```text
scoring_date = 2026-09-10
```

But within a single day, we may still need more partitions.

### Region partitioning

Sometimes data is naturally regional:

```text
country = India
country = USA
country = Japan
```

But this can create skew.

Example:

```text
India partition = 30 million rows
Iceland partition = 20 thousand rows
```

Now one worker gets a huge partition while others finish quickly.

This is called a straggler problem.

---

## 5.9.6 Data Skew and Stragglers

A straggler is a worker or partition that takes much longer than others.

Example:

```text
Worker 1 finished in 10 minutes
Worker 2 finished in 11 minutes
Worker 3 finished in 9 minutes
Worker 4 still running after 90 minutes
```

Why does this happen?

Possible reasons:

```text
one partition has too many rows
one partition has very large entities
data is skewed by region/category/source
one worker has slower hardware
one input file is corrupt or slow to read
model loading is slow on one worker
```

Stragglers matter because the whole batch job cannot complete until the slowest required partition finishes.

If the SLA says:

```text
scores must be ready by 6 AM
```

then one slow partition can miss the SLA.

Ways to reduce stragglers:

```text
use hash partitioning
increase number of smaller partitions
avoid partitioning only by skewed fields
split very large partitions
use dynamic task scheduling
monitor partition runtime
```

The goal is not just parallelism.

The goal is balanced parallelism.

---

## 5.9.7 Worker-Based Scoring Lifecycle

Each worker should follow a consistent lifecycle.

```text
1. Start worker
2. Receive assigned partition
3. Load model bundle
4. Validate model version/checksum
5. Read partition input
6. Validate feature schema
7. Apply preprocessing
8. Score rows in batches
9. Attach output metadata
10. Write output partition
11. Report success/failure
```

Example output partition:

```text
partition_id = 004

host_id      junk_score      model_version      scoring_run_id      scored_at
abc.com      0.91            v17                2026_09_10          2026-09-10 02:31
xyz.net      0.08            v17                2026_09_10          2026-09-10 02:31
```

The worker should not only write scores.

It should write enough metadata to validate and debug the run.

---

## 5.9.8 Batch Scoring Should Be Vectorized

A bad scoring pattern:

```python
for row in rows:
    score = model.predict(row)
```

This causes huge overhead.

Better:

```python
scores = model.predict(batch_of_rows)
```

Why?

Because every function call has overhead.

Also, ML libraries are often optimized to score many rows at once.

So a worker may process input like:

```text
read 100,000 rows
convert to model input format
predict all 100,000 rows
write output
read next 100,000 rows
...
```

This is called batch-wise or vectorized prediction.

It improves throughput.

---

## 5.9.9 Memory Considerations

Distributed scoring is not only about CPU.

It is also about memory.

Each worker must hold:

```text
model artifact
input batch
intermediate transformed features
prediction output
runtime overhead
```

For large tree ensembles, the model itself may be large.

Example:

```text
1000 trees
depth 8
many split nodes
large feature metadata
```

If the worker batch size is too large, memory can blow up.

Bad:

```text
worker reads entire 10 million row partition into memory
```

Better:

```text
worker reads partition in chunks
scores chunk
writes chunk
releases memory
```

So batch size is a tuning parameter.

Too small:

```text
too much overhead
```

Too large:

```text
memory pressure
worker crashes
slow garbage collection
```

A good production system finds a stable chunk size.

---

## 5.9.10 Output Writing

Workers should not write directly to the final production table while the job is still running.

Bad pattern:

```text
worker outputs directly into /predictions/latest/
```

Why bad?

Because downstream systems may read partial output.

Example:

```text
100 partitions expected
only 63 partitions written so far
downstream system reads latest
uses incomplete scores
```

Better pattern:

```text
write to temporary run-specific path
validate
then publish atomically
```

Example:

```text
/scoring_runs/2026_09_10/tmp/partition_id=001
/scoring_runs/2026_09_10/tmp/partition_id=002
...
```

After validation:

```text
/predictions/host_junk/latest → /scoring_runs/2026_09_10/final/
```

The idea is:

```text
Downstream systems should only see complete, validated outputs.
```

---

## 5.9.11 Idempotency

Idempotency means:

```text
rerunning the same task should not create duplicate or inconsistent results
```

This matters because distributed jobs often retry failed tasks.

Suppose partition 003 fails halfway.

The system retries partition 003.

Bad output design:

```text
append rows to production table
```

This may create duplicates:

```text
abc.com score written once
abc.com score written again after retry
```

Better design:

```text
write partition output to deterministic path
overwrite partition output for same run_id and partition_id
```

Example:

```text
/scoring_runs/2026_09_10/partition_id=003/
```

If partition 003 is retried, it writes to the same path.

Then final validation checks:

```text
one output per expected partition
one score per entity
no duplicates
```

Idempotency makes retries safe.

---

## 5.9.12 Fault Tolerance

Distributed scoring must expect failure.

Failures are normal.

Examples:

```text
worker crashes
model artifact download fails
input partition missing
bad record causes parsing error
output write fails
cluster node is preempted
timeout occurs
memory limit exceeded
```

A robust system supports:

```text
partition retry
bad-record handling
temporary output cleanup
missing-partition detection
global validation
alerting
fallback to previous successful run
```

The important point:

```text
A partial scoring run should not become the production scoring run.
```

If today's scoring run fails, downstream systems may continue using:

```text
yesterday's successful score table
```

with a warning like:

```text
score_freshness = stale
```

This is better than publishing incomplete or corrupted output.

---

## 5.9.13 Partition-Level Validation

Each partition should be validated.

Partition-level checks include:

```text
partition exists
row count is expected
no null scores
score range is valid
model version is correct
feature schema version is correct
output columns exist
no duplicate entity IDs inside partition
runtime was within expected range
```

Example:

```text
partition_id = 004
expected_rows = 1,000,000
actual_rows = 999,982
null_scores = 0
model_version = v17
score_min = 0.001
score_max = 0.998
status = pass
```

If one partition fails validation, the full run should usually not be published.

---

## 5.9.14 Global Output Validation

After all partitions finish, validate the complete output.

Global checks include:

```text
all expected partitions are present
total row count matches input row count
unique entity count matches expected count
score distribution is reasonable
missing score count is zero or acceptable
all partitions used same model version
all partitions used same feature schema
comparison with previous run is reasonable
```

Example:

```text
Yesterday:
mean junk_score = 0.12

Today:
mean junk_score = 0.67
```

This does not automatically mean the output is wrong.

Maybe the web changed drastically.

But usually this kind of shift should trigger investigation.

The system should ask:

```text
Did input population change?
Did feature joins fail?
Did missing rates increase?
Did model version change?
Did preprocessing change?
```

This connects distributed scoring back to Parts 5.6 and 5.7.

---

## 5.9.15 Score Distribution by Partition

A subtle check is score distribution by partition.

Suppose global mean looks normal:

```text
global mean score = 0.12
```

But partition-level means are:

```text
partition 001 mean = 0.11
partition 002 mean = 0.12
partition 003 mean = 0.92
partition 004 mean = 0.10
```

Partition 003 is suspicious.

Possible causes:

```text
bad input data in partition 003
wrong feature order in partition 003
corrupt feature values
wrong model loaded by one worker
region-specific real shift
data source issue
```

So we do not only validate globally.

We validate by partition also.

---

## 5.9.16 Metadata in Distributed Scoring Output

Each prediction row should include metadata.

Minimum output:

```text
entity_id
score
```

Production output:

```text
entity_id
score
risk_bucket
model_name
model_version
feature_schema_version
feature_pipeline_version
scoring_run_id
partition_id
scored_at
feature_timestamp
score_valid_until
data_quality_flags
```

Why?

Because when something goes wrong, metadata helps debug.

Example:

```text
Only partition 003 has abnormal scores.
All rows in partition 003 were scored by model_v18 accidentally.
```

Without metadata, this is hard to diagnose.

---

## 5.9.17 Distributed Scoring for Host Junk Scoring

Let us connect this to host junk scoring.

Suppose we are scoring host junk risk daily.

### Input

```text
20 million active hosts
```

### Model

```text
host_junk_xgboost_v17
```

### Features

```text
host_error_ratio_10d
host_crawl_ratio_10d
host_dsat_ratio_30d
host_redirect_ratio_30d
host_static_rank
host_spam_signal_count
```

### Partitioning

```text
partition_id = hash(host_id) % 200
```

So:

```text
20 million hosts / 200 partitions
≈ 100,000 hosts per partition
```

### Worker flow

```text
Worker gets partition 037
loads host_junk_xgboost_v17
validates feature schema
scores 100,000 hosts in batches
writes output to partition 037 path
```

### Output

```text
host_id      junk_score      risk_bucket      model_version      partition_id      scoring_run_id
abc.com      0.91            high             v17                037               2026_09_10
xyz.net      0.08            low              v17                037               2026_09_10
```

### Validation

```text
all 200 partitions present
20 million output rows
no duplicate host_id
no null scores
all model_version = v17
score distribution stable
missing feature rates acceptable
```

### Publish

If validation passes:

```text
publish /host_junk_scores/latest
```

If validation fails:

```text
do not publish
alert owner
keep yesterday's scores active
```

This is production-safe batch scoring.

---

# Final Summary

Part 5.8 and Part 5.9 are about what happens after we have prepared the feature table.

## 5.8 Loading the Ensemble Model

The key idea:

```text
Loading the model means loading the full inference contract.
```

That contract includes:

```text
model version
feature schema
feature order
preprocessing
missing-value policy
categorical encoders
calibration
threshold policy
runtime metadata
```

The danger is not only that model loading may fail.

The bigger danger is that the wrong model or wrong inference contract loads successfully.

## 5.9 Distributed Scoring

The key idea:

```text
Distributed scoring means applying the same model contract across many data partitions in parallel.
```

A production distributed scoring system must handle:

```text
partitioning
worker-level model loading
batch-wise prediction
memory management
fault tolerance
idempotent outputs
partition validation
global validation
atomic publishing
```

Most important memory line:

```text
In batch inference, loading the model means loading the full inference contract, and distributed scoring means applying that contract consistently across many partitions.
```

---

# Sources and References

```text
1. MLflow Model Registry documentation
   Used for understanding model registry workflows, model versions, and model metadata/signature concepts.

2. MLflow Models documentation
   Used for model signature and input/output schema contract concepts.

3. XGBoost model IO documentation
   Used for XGBoost model-saving and loading format concepts.

4. Azure Machine Learning Batch Endpoints documentation
   Used for batch endpoint and asynchronous batch scoring workflow concepts.

5. Apache Spark ML Pipeline documentation
   Used for the idea of treating model scoring as a DataFrame transformation in distributed systems.
```

---

Next part:

```text
5.10 Prediction Output Design
5.11 Incremental Scoring
```
