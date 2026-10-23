# Phase 2 — Dataset Collection and Annotation

## Learning goals
Understand where labeled webpage screenshots come from, what makes a trustworthy label, and how to avoid biased data.

## 1. Unit of data
Each row represents one webpage capture: `sample_id`, image URI, capture timestamp, viewport dimensions, capture status, annotator label, annotation version, and optional metadata used for auditing (not fed to the screenshot-only model). Never put private URLs or sensitive screenshots in a public repository.

**Decision to make:** use initial viewport, full-page image, or multiple captures. The first is cheap but can miss below-the-fold content; the second is more complete but costly; the third balances coverage and complexity.

## 2. Label policy
- **LITTLE_CONTENT (1):** insufficient meaningful visible content for the page's apparent purpose.
- **NOT_LITTLE_CONTENT (0):** enough useful content for the page's purpose.
- **REVIEW/EXCLUDE:** broken capture, loading state, unreadable screenshot, or genuinely ambiguous intent. This is a dataset workflow state, not a third model class.

Do **not** equate few words, whitespace, login forms, calculators, or search boxes with low quality by default.

## 3. Sampling strategy
Sample across languages, countries, page types, device widths, traffic tiers, domains, and capture dates. Include hard negatives: sparse but functional pages, image-heavy useful pages, and templates with large whitespace. Include hard positives: skeleton screens, placeholder articles, thin affiliate pages, and pages whose main content is missing.

Use a **representative traffic sample** for final evaluation; a balanced training sample is allowed but changes class prevalence. Record sampling weights if estimating real-world precision.

## 4. Annotation workflow
1. Write a rubric with positive/negative examples and exclusion criteria.
2. Run a small pilot; review disagreements and revise definitions.
3. Independently double-label a subset; adjudicate disagreements.
4. Store annotator ID, rubric version, timestamp, and adjudication outcome.
5. Freeze a gold test set before training iterations.

Measure agreement (percent agreement and Cohen's kappa for two raters); investigate systematic disagreement rather than trusting a single score.

## 5. Data record example
```json
{
  "sample_id": "sample_0001",
  "image_uri": "s3://private-bucket/captures/sample_0001.png",
  "capture_time": "2026-10-01T10:00:00Z",
  "viewport": {"width": 1365, "height": 768},
  "capture_status": "complete",
  "label": 1,
  "label_source": "adjudicated",
  "rubric_version": "v1"
}
```
The URI is illustrative, not an actual dataset.

## 6. Dataset quality checks
Check missing images, corrupted files, mislabeled samples, duplicate captures, class imbalance, language coverage, and page-template concentration. Track counts and percentages for each failure reason.

## 7. Risks and decisions
- Screenshots may contain personal information: redact or exclude before storage.
- Copyright and internal-data permissions matter.
- Pages change over time; save capture time and version.
- Screenshots can be misleading if the page was not fully loaded.
- Avoid allowing the same page or template family to leak across splits.

## Deliverables
Annotation rubric v1, capture specification, sampling plan, dataset schema, annotation QA report, and approved storage/access policy.

## Interview recall
**Q: Why not label pages only using word count?**  
A: A page can have few words but still be useful; the screenshot and page purpose matter.

**Q: Why sample hard negatives?**  
A: They teach the model not to mistake a simple design for little content.

**Next:** Phase 3 — preprocessing and leakage-safe splitting.
