# Merge Intervals

## Problem Statement

Given an array of intervals where each interval is represented as:

```python
[start, end]
```

merge all overlapping intervals and return a list of non-overlapping intervals.

---

## Example

```python
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
```

Output:

```python
[[1, 6], [8, 10], [15, 18]]
```

Explanation:

```text
[1, 3] and [2, 6] overlap.

Merged interval:
[1, 6]
```

So the final result is:

```python
[[1, 6], [8, 10], [15, 18]]
```

---

## Difficulty

```text
Medium
```

---

## Pattern

```text
Sorting + Intervals
```

The key idea is:

```text
Sort intervals by start time.
Then scan from left to right and merge overlapping intervals.
```

---

## Core Idea

If intervals are sorted by their start time, then overlapping intervals will appear next to each other.

Example:

```python
intervals = [[8, 10], [1, 3], [2, 6]]
```

After sorting by start time:

```python
intervals = [[1, 3], [2, 6], [8, 10]]
```

Now we can process intervals one by one.

---

## Overlap Condition

Suppose we have:

```python
previous = [1, 3]
current = [2, 6]
```

They overlap because:

```text
current_start <= previous_end
```

Here:

```text
2 <= 3
```

So these intervals overlap.

---

## Merge Operation

When two intervals overlap:

```python
previous = [1, 3]
current = [2, 6]
```

The merged interval becomes:

```python
[1, max(3, 6)]
```

So:

```python
[1, 6]
```

Since intervals are sorted by start time, we keep the previous start.

We only update the end:

```python
previous[1] = max(previous[1], current[1])
```

---

## Non-Overlap Condition

Suppose:

```python
previous = [1, 6]
current = [8, 10]
```

They do not overlap because:

```text
current_start > previous_end
```

Here:

```text
8 > 6
```

So we add the current interval as a new interval.

---

## Optimized Code

```python
def merge(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]

        current_start = current[0]
        current_end = current[1]
        previous_end = previous[1]

        if current_start <= previous_end:
            previous[1] = max(previous_end, current_end)
        else:
            merged.append(current)

    return merged
```

---

## Dry Run

Input:

```python
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
```

After sorting:

```python
[[1, 3], [2, 6], [8, 10], [15, 18]]
```

Initial state:

```python
merged = [[1, 3]]
```

---

### Step 1

Current interval:

```python
[2, 6]
```

Previous merged interval:

```python
[1, 3]
```

Check overlap:

```text
current_start <= previous_end
2 <= 3
True
```

So they overlap.

Merge them:

```python
previous[1] = max(3, 6)
```

Now:

```python
merged = [[1, 6]]
```

---

### Step 2

Current interval:

```python
[8, 10]
```

Previous merged interval:

```python
[1, 6]
```

Check overlap:

```text
current_start <= previous_end
8 <= 6
False
```

So they do not overlap.

Append current interval:

```python
merged = [[1, 6], [8, 10]]
```

---

### Step 3

Current interval:

```python
[15, 18]
```

Previous merged interval:

```python
[8, 10]
```

Check overlap:

```text
current_start <= previous_end
15 <= 10
False
```

So they do not overlap.

Append current interval:

```python
merged = [[1, 6], [8, 10], [15, 18]]
```

Final answer:

```python
[[1, 6], [8, 10], [15, 18]]
```

---

## Follow-Up Questions

### Q1. Why do we sort the intervals first?

We sort the intervals by start time because after sorting, overlapping intervals will appear next to each other.

Without sorting, overlapping intervals may be far apart.

Example:

```python
[[8, 10], [1, 3], [2, 6]]
```

After sorting:

```python
[[1, 3], [2, 6], [8, 10]]
```

Now merging becomes easy in one pass.

---

### Q2. What is the overlap condition?

For sorted intervals, two intervals overlap if:

```text
current_start <= previous_end
```

Example:

```python
previous = [1, 3]
current = [2, 6]
```

Since:

```text
2 <= 3
```

they overlap.

---

### Q3. How do we merge two overlapping intervals?

We keep the previous start and update the end:

```python
previous[1] = max(previous[1], current[1])
```

Example:

```python
previous = [1, 3]
current = [2, 6]
```

Merged interval:

```python
[1, 6]
```

---

### Q4. What if intervals just touch?

Example:

```python
[1, 4] and [4, 5]
```

In the standard version of this problem, they are considered overlapping because:

```text
4 <= 4
```

So they are merged into:

```python
[1, 5]
```

In an interview, you can clarify:

```text
Should touching intervals like [1, 4] and [4, 5] be considered overlapping?
```

Usually, the answer is yes.

---

### Q5. What if the input is empty?

If:

```python
intervals = []
```

Return:

```python
[]
```

The code handles this using:

```python
if not intervals:
    return []
```

---

### Q6. What is the time complexity?

```text
O(n log n)
```

Why?

Sorting takes:

```text
O(n log n)
```

Scanning through the intervals takes:

```text
O(n)
```

So total time complexity is:

```text
O(n log n)
```

---

### Q7. What is the space complexity?

Simple interview answer:

```text
O(n)
```

because we store the merged output.

If the interviewer asks for extra space excluding output, it can be:

```text
O(1) or O(log n)
```

depending on the sorting implementation.

---

### Q8. Can we do better than O(n log n)?

If the intervals are unsorted, generally we need sorting, so the time complexity is:

```text
O(n log n)
```

If the intervals are already sorted by start time, then we can merge them in one scan:

```text
O(n)
```

---

## Common Mistakes

### Mistake 1: Not sorting first

If we do not sort, overlapping intervals may not be adjacent.

Example:

```python
[[8, 10], [1, 3], [2, 6]]
```

Without sorting, we may miss the overlap between:

```python
[1, 3] and [2, 6]
```

---

### Mistake 2: Wrong overlap condition

Correct condition:

```python
current_start <= previous_end
```

Wrong condition:

```python
current_start < previous_end
```

Why?

Because intervals like:

```python
[1, 4] and [4, 5]
```

should usually be merged in this problem.

---

### Mistake 3: Appending overlapping interval instead of merging

If intervals overlap, do not append the current interval.

Instead, update the end of the previous merged interval.

Correct:

```python
previous[1] = max(previous[1], current[1])
```

---

### Mistake 4: Forgetting to take max of end values

Example:

```python
previous = [1, 10]
current = [2, 3]
```

These intervals overlap.

The merged interval should remain:

```python
[1, 10]
```

So we must use:

```python
max(previous[1], current[1])
```

not simply:

```python
current[1]
```

---

## Interview Explanation

I will first sort the intervals by their start time.

After sorting, overlapping intervals will appear next to each other.

I initialize a merged list with the first interval.

Then for every current interval, I compare it with the last interval in the merged list.

If the current interval starts before or at the end of the previous merged interval, then they overlap, so I update the end of the previous interval using the maximum end value.

Otherwise, they do not overlap, so I append the current interval as a new interval.

Sorting takes `O(n log n)` and the scan takes `O(n)`, so the total time complexity is `O(n log n)`.

The space complexity is `O(n)` for the output.

---

## Key Takeaways

```text
Problem:
Merge Intervals

Difficulty:
Medium

Pattern:
Sorting + Intervals

Sort by:
Start time

Overlap condition:
current_start <= previous_end

Merge operation:
previous[1] = max(previous[1], current[1])

Time Complexity:
O(n log n)

Space Complexity:
O(n)

Important follow-up:
If intervals are already sorted, time becomes O(n).
```
