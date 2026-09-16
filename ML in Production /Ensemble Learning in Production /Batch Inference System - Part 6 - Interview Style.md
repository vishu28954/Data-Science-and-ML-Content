# Batch Inference System - Part 6 - Interview Style

# Failure Handling and Monitoring Batch Inference Jobs

This note belongs with:

```text
Ensemble Learning in Production
Part 5 — Batch Inference System
Part 5.12 — Failure Handling
Part 5.13 — Monitoring Batch Inference Jobs
```

These notes explain Part 5.12 and Part 5.13 in interview-speaking style.

---

# 5.12 Failure Handling — Interview Style

## 1. What does failure handling mean in a production batch inference system?

### Interview answer

Failure handling means making sure that unreliable predictions do not become trusted production output.

In production, failure is not limited to a job crashing. A batch job can finish successfully and still be wrong because the input data was incomplete, a feature pipeline was stale, one partition was missing, the wrong model version was loaded, or prediction distributions were abnormal.

So I would handle both hard failures and silent failures. Hard failures include worker crashes, missing model artifacts, storage write errors, or out-of-memory errors. Silent failures include missing features, unexpected row-count drops, wrong feature order, stale snapshots, or inconsistent model versions across workers.

The key principle is that publishing should happen only after the entire run passes validation.

---

## 2. What is the difference between hard failures and silent failures?

### Interview answer

A hard failure is obvious because the pipeline throws an error or stops. Examples include a worker crash, missing model file, permission error, corrupt input, or out-of-memory exception.

A silent failure is more dangerous because the job may still report success while the output is wrong. Examples include a feature coverage drop from 95% to 20%, using stale feature snapshots, loading the wrong model version, or generating only 70% of the expected predictions.

So I would not rely only on job status. I would validate data quality, feature health, model metadata, output completeness, and prediction distributions before publishing.

---

## 3. How would you categorize failures in batch inference?

### Interview answer

I would categorize failures by stage because the recovery action depends on where the failure occurs.

Typical categories are:

```text
input failures
feature failures
model-loading failures
scoring failures
output failures
publishing failures
downstream-consumption failures
```

For example, if the input snapshot is incomplete, I would stop the pipeline before scoring. If one scoring partition fails, I would retry that partition. If publishing fails after validation, I would keep the previous successful snapshot active instead of exposing partial output.

This stage-based design helps isolate failures and reduce blast radius.

---

## 4. How would you handle input failures?

### Interview answer

For input failures, I would validate the scoring population before feature generation or model scoring.

I would check row count, duplicate entity IDs, missing IDs, partition completeness, source freshness, and large deviations from the previous run.

For example, if yesterday I had 20 million eligible hosts and today only 3 million appear, I would not immediately score the 3 million. I would first investigate whether an upstream partition is missing, a date filter changed, or a join failed.

The rule is that if the input population violates the expected contract, I would block the scoring run rather than producing incomplete predictions.

---

## 5. How would you handle feature failures?

### Interview answer

Feature failures should be handled based on feature criticality and the model's missing-value behavior.

I would validate feature availability, missing rate, valid range, freshness, and distribution before scoring.

For example, if an optional metadata feature is missing and the model was trained with a safe missing-value policy, I may continue and mark the prediction as degraded. But if a critical feature group like crawl-error signals disappears, I may block publishing entirely.

So I would classify features as critical, important, or optional and define explicit behavior for each category.

---

## 6. How would you handle model-loading failures?

### Interview answer

Before scoring begins, I would validate the model artifact itself.

I would verify model name, approved version, checksum, feature-schema version, preprocessing version, class mapping, and calibration metadata.

If any of these do not match the expected production contract, I would stop scoring.

If availability is important, I may fall back to a known-good previous model version, but I would never silently load an arbitrary artifact.

The main goal is to prevent the system from producing scores with an unverified model.

---

## 7. How would you handle a failed scoring partition?

### Interview answer

If one partition fails, I would retry that partition rather than rerunning the entire batch job.

For example, if 199 out of 200 partitions succeed and partition 037 fails, I would retry only partition 037. If it fails repeatedly, I would mark the run incomplete and block final publishing unless the system explicitly supports partial output.

I would not let downstream consumers see a table that is accidentally missing one partition because they may assume the snapshot is complete.

This is why partition-level fault isolation and validation are important.

---

## 8. What is your retry strategy?

### Interview answer

I would retry only transient failures and use bounded retries with backoff.

For example:

```text
attempt 1
wait 30 seconds
attempt 2
wait 2 minutes
attempt 3
fail permanently
```

Transient failures like network timeouts, worker preemption, or temporary storage issues are good retry candidates.

But if a partition keeps failing because the data is corrupt, infinite retries are wasteful. So I would classify errors, use a maximum retry count, and escalate persistent failures.

Retries also need idempotent output so the same partition can be safely recomputed without creating duplicates.

---

## 9. Why is idempotency important for failure recovery?

### Interview answer

Idempotency makes retries safe.

Suppose a worker writes half of partition 037 and then crashes. If the retry simply appends output, the final table may contain duplicate predictions.

A better design is to write each partition to a deterministic path like:

```text
/run_2026_09_16/partition=037/
```

Then a retry can overwrite the same partition output.

The final system checks that exactly one complete output exists for each expected partition.

So retry strategy and idempotent output design should be implemented together.

---

## 10. How would you handle a few malformed records?

### Interview answer

I would distinguish between isolated bad records and systemic upstream failure.

If 5 out of 20 million records are malformed, I may quarantine those records and continue. The quarantine table would store entity ID, error type, raw input, and scoring run ID so the issue can be investigated later.

But if millions of records are malformed, I would treat that as an upstream data failure and block publishing.

So I would define an acceptable bad-record threshold rather than always failing the entire job or always ignoring errors.

---

## 11. How would you handle output failures?

### Interview answer

After scoring, I would validate the output before publishing.

I would check that all expected partitions exist, output row count matches the scoring population, entity IDs are unique, scores are not null, score ranges are valid, and every partition used the same model and feature-schema versions.

If these validations fail, the run should remain in a temporary location and should not become the production snapshot.

This ensures that downstream systems only consume complete and validated predictions.

---

## 12. What happens if publishing fails?

### Interview answer

Publishing should be atomic and should never destroy the previous successful snapshot.

The new output should first be completely generated and validated. Then the production pointer or table should switch to the new snapshot in one controlled step.

If that publish step fails, the previous successful snapshot should remain active.

This prevents downstream systems from seeing partial or missing predictions.

---

## 13. What fallback strategy would you use if today's batch run fails?

### Interview answer

The most common fallback is to use the previous successful prediction snapshot.

For example, if today's host-risk scoring job fails, downstream systems may continue using yesterday's scores for a limited period.

However, I would mark those predictions as stale and enforce a maximum fallback age. For example, reusing a one-day-old score may be acceptable, while using a five-day-old score may not be.

So fallback should be explicit, time-bounded, and visible in the output metadata.

---

## 14. Would you ever use a fallback model?

### Interview answer

Yes, if prediction availability is very important and the system can support it safely.

For example, the primary model may be a large XGBoost model using many feature groups, while a fallback model may use only robust core features.

If a non-core feature pipeline fails, the fallback model can still produce a score.

But this adds complexity, so every prediction must record which model generated it and whether fallback mode was used.

---

## 15. What is degraded mode?

### Interview answer

Degraded mode means the system continues operating with reduced information instead of treating the run as fully healthy or fully failed.

For example, if DSAT features are unavailable but crawl and spam features are healthy and the model can safely handle missing DSAT values, the system may continue scoring.

The prediction output should then contain a degraded-quality flag.

A downstream policy may decide that degraded predictions can be used for manual review but not for automatic blocking.

---

## 16. What is fail-open vs fail-closed behavior?

### Interview answer

Fail-open means the system allows the normal action when the ML prediction is unavailable. Fail-closed means the system blocks or holds the action when the prediction is unavailable.

For example, in a host-risk system, fail-open may mean not automatically blocking a host when the risk score is missing.

In a high-risk fraud or compliance system, fail-closed may mean holding the transaction until a score is available.

The correct choice depends on business risk and should be treated as a policy decision rather than purely an ML decision.

---

## 17. How would you roll back a bad model deployment?

### Interview answer

If a new model version produces bad scores, I would stop publishing its outputs, restore the previous successful prediction snapshot, switch the production model alias back to the known-good version, and rescore affected entities if necessary.

Rollback is possible only if model artifacts, feature versions, scoring runs, and prediction snapshots are versioned.

That is why lineage metadata is essential for recovery.

---

## Complete 5.12 Answer

If asked:

```text
How would you design failure handling for a production batch inference system?
```

Answer:

```text
I would design the batch inference pipeline so that every major stage has a validation contract before the pipeline can move forward.

I would validate input population, feature quality, model artifact, scoring completeness, and final output separately. I would distinguish hard failures such as worker crashes from silent failures such as feature coverage drops or stale snapshots.

For transient worker or infrastructure failures, I would use bounded retries with backoff and idempotent partition outputs. For isolated bad records, I may quarantine them if the error rate is below a defined threshold. If a critical feature or model validation fails, I would block publishing.

Outputs would always be written to a temporary location first and only published after global validation. If today's run fails, I would fall back to the previous successful snapshot for a limited freshness window, or optionally use an approved fallback model.

I would also support degraded mode, rollback, and clear fail-open or fail-closed policies depending on the business risk.
```

---

# 5.13 Monitoring Batch Inference Jobs — Interview Style

## 18. What should be monitored in a production batch inference system?

### Interview answer

I would monitor the entire prediction pipeline, not just whether the batch job succeeded.

The main layers are:

```text
pipeline health
input health
feature health
model/scoring health
output health
business impact
```

This is important because a batch job can finish successfully but still produce incorrect predictions.

So monitoring should tell me not only whether the system ran, but also whether the data was correct and whether the predictions look believable.

---

## 19. What pipeline-health metrics would you monitor?

### Interview answer

For pipeline health, I would track job status, start and completion time, total runtime, retry count, failed partitions, worker failures, resource utilization, and SLA status.

For a daily batch job, completion deadline matters more than millisecond latency.

For example, if scores must be ready by 6 AM and the job finishes at 7:10 AM, the job technically succeeded but operationally failed its SLA.

I would also monitor runtime trends to catch gradual degradation before the SLA is missed.

---

## 20. What input-data metrics would you monitor?

### Interview answer

I would monitor entity count, duplicate rate, missing-ID rate, source freshness, new-entity count, inactive-entity count, and partition completeness.

For example, if the normal scoring population is around 20 million entities and suddenly falls to 4 million, I would immediately investigate before trusting the run.

Input population monitoring is simple but very effective for catching upstream failures.

---

## 21. What feature metrics would you monitor?

### Interview answer

For features, I would monitor missing rate, join coverage, freshness, valid ranges, and distributions.

For example, if DSAT feature coverage normally stays around 96% but drops to 31%, that suggests an upstream feature-pipeline problem even if the scoring job finishes successfully.

I would also monitor feature age because a value can exist but still be stale.

Finally, I would track distribution statistics like mean, median, percentiles, min, max, and histograms to detect silent corruption.

---

## 22. How would you distinguish a real data shift from a pipeline bug?

### Interview answer

Monitoring can detect that a distribution changed, but it cannot always tell whether the change is real or caused by a bug.

If a feature suddenly shifts, I would investigate supporting signals such as source volume, population composition, upstream logic changes, feature version changes, and related feature groups.

For example, if spam ratio increases while source volumes and related signals also change consistently, the shift may be real. If only one feature changes dramatically while everything else remains stable, a pipeline issue becomes more likely.

So anomaly detection raises the question; root-cause analysis answers it.

---

## 23. What model-related metrics would you monitor during batch scoring?

### Interview answer

I would monitor model name, model version, checksum, feature-schema version, model-load success, and consistency across partitions.

For example, if 199 partitions use model v17 and one partition uses model v18, the run should fail validation.

I would also track model-load time because unusually slow artifact loading can affect batch SLA.

---

## 24. What scoring-performance metrics would you monitor?

### Interview answer

I would monitor rows scored per second, partition runtime, model prediction time, input read time, output write time, CPU usage, memory usage, retries, and error counts.

If throughput drops from 50,000 rows per second to 8,000, I would investigate whether the cause is a larger model, smaller batch size, CPU contention, slow storage, or skewed partitions.

Breaking runtime into stages helps identify whether the bottleneck is data IO or model inference.

---

## 25. How would you monitor stragglers?

### Interview answer

I would monitor the distribution of partition runtimes, not just the average.

For example:

```text
p50 partition runtime = 8 min
p95 = 10 min
max = 70 min
```

That indicates a straggler.

I would correlate partition runtime with partition size, worker resources, and input source to determine whether the cause is data skew, a slow node, or storage issues.

---

## 26. What output metrics would you monitor?

### Interview answer

I would monitor output row count, unique entity count, duplicate rate, null-score rate, score range, score distribution, class rate, and risk-bucket distribution.

For example, if high-risk hosts usually represent 3% of the population and suddenly become 41%, I would investigate before publishing.

That change could be real, but it could also indicate a wrong feature, wrong threshold, model mismatch, or calibration problem.

---

## 27. Why should input and output row counts be compared?

### Interview answer

Because large discrepancies often reveal silent failures.

If I start with 20 million eligible entities but produce only 16 million predictions, I need to know where 4 million rows disappeared.

Possible causes include failed partitions, inner joins, invalid rows, or write failures.

So input-to-output row-count consistency is one of the simplest and most valuable checks in batch inference.

---

## 28. Why monitor duplicate predictions?

### Interview answer

If the current prediction snapshot should have one row per entity, duplicate IDs indicate a serious data-integrity issue.

Duplicates can come from one-to-many feature joins, retry append bugs, partition overlap, or incorrect merges.

So I would monitor duplicate entity count and block publishing if uniqueness is violated.

---

## 29. How would you monitor incremental scoring?

### Interview answer

For incremental scoring, I would monitor candidate count, rescored count, reused-score count, new-entity count, expired-score count, and candidate rate.

If the normal candidate rate is around 8% and suddenly becomes 95%, it may indicate a feature fingerprint or candidate-generation bug.

If the rate becomes almost zero, change detection may have stopped working.

So the incremental selection logic itself needs monitoring.

---

## 30. How would you monitor fallback usage?

### Interview answer

I would track fallback count, fallback rate, stale-score rate, and oldest score age.

If only 0.2% of entities use previous-day scores, that may be acceptable.

If 60% of the population is using fallback scores, the current pipeline is effectively unhealthy even if downstream systems still have predictions available.

Fallback usage should therefore be treated as an operational health signal.

---

## 31. What business-level metrics would you monitor?

### Interview answer

Technical health is not enough, so I would also monitor downstream business impact.

Examples include high-risk entity rate, auto-action rate, manual-review queue size, acceptance or rejection rate, precision from reviewed samples, and population coverage.

For example, if infrastructure metrics look normal but the review queue increases 10x overnight, that may indicate a change in model behavior or decision thresholds.

Business metrics help connect model output to real system impact.

---

## 32. How would you design alerts?

### Interview answer

Alerts should be actionable and include severity, affected run, failing metric, and fallback status.

A weak alert would say:

```text
Batch job issue.
```

A better alert would say:

```text
Host Junk Batch Run 2026-09-16:
DSAT feature coverage dropped from 96% to 31%.
Publishing blocked.
Previous successful score snapshot remains active.
```

I would also separate warnings from critical alerts to reduce alert fatigue.

---

## 33. Static thresholds or dynamic baselines?

### Interview answer

I would usually start with simple static thresholds because they are easy to understand and debug.

For example:

```text
missing_rate > 20% → critical
```

For naturally variable metrics, I may later add dynamic baselines such as comparing today's row count or score distribution with a trailing seven-day median or historical standard deviation.

The goal is to catch meaningful anomalies without creating excessive false alarms.

---

## 34. What would you put on a monitoring dashboard?

### Interview answer

I would organize the dashboard into sections:

```text
Run Health
Input Health
Feature Health
Scoring Health
Output Health
Business Impact
```

For example, I would show job status, runtime, SLA, input entity count, feature coverage, missing rates, feature freshness, model version, throughput, failed partitions, prediction distribution, high-risk rate, duplicate count, and fallback rate.

The goal is that an operator can quickly answer whether the run completed, whether the data was healthy, and whether the predictions look reasonable.

---

## Complete 5.13 Answer

If asked:

```text
How would you monitor a production batch inference pipeline?
```

Answer:

```text
I would monitor the full prediction pipeline rather than only the model process.

At the pipeline level, I would track job status, runtime, SLA, retries, and failed partitions. At the input level, I would monitor entity count, duplicates, missing IDs, source freshness, and partition completeness.

At the feature level, I would monitor missing rates, join coverage, freshness, valid ranges, and feature distributions. At the model and scoring level, I would verify model version and checksum consistency, track throughput, worker memory, partition runtimes, and stragglers.

For outputs, I would monitor row count, uniqueness, null scores, score distributions, class or risk-bucket rates, and compare those metrics with historical baselines. For incremental systems, I would also track candidate rates, reused scores, and fallback usage.

Finally, I would monitor business-level metrics such as auto-action rate and review-queue size, and configure severity-based alerts that explain what changed, which run is affected, and whether fallback output is currently active.
```

---

# Crisp Combined Answer

```text
For Failure Handling, I would validate every stage of the batch pipeline and prevent unreliable output from being published. I would distinguish hard failures from silent data failures, retry transient partition failures using idempotent outputs, quarantine isolated bad records, use previous successful snapshots as bounded fallbacks, and support rollback or degraded mode when appropriate.

For Monitoring, I would monitor the entire prediction pipeline: job health, input population, feature quality and freshness, model version, worker throughput, partition runtime, output completeness, prediction distributions, fallback usage, and downstream business impact.

The key principle is that a successful job is not enough. I need evidence that the data was correct and the predictions are trustworthy before publishing.
```

---

# Most Important Memory Lines

```text
In production ML, failure handling means preventing unreliable predictions from becoming trusted predictions.
```

```text
Monitoring a model means monitoring the entire prediction pipeline, not just the model process.
```
