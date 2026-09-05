# Search in Rotated Sorted Array

## Problem Statement

You are given a rotated sorted array `nums` and an integer `target`.

Return the index of `target` if it exists in the array.

If `target` does not exist, return `-1`.

---

## Example

```python
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
```

Output:

```python
4
```

Explanation:

```text
target = 0 is present at index 4.
```

---

## What is a Rotated Sorted Array?

A rotated sorted array is an originally sorted array that has been shifted from some pivot point.

Original sorted array:

```python
[0, 1, 2, 4, 5, 6, 7]
```

Rotated array:

```python
[4, 5, 6, 7, 0, 1, 2]
```

The array is not fully sorted anymore, but it still has sorted parts.

---

## Difficulty

```text
Medium
```

---

## Pattern

```text
Modified Binary Search
```

This problem is solved using binary search, but with an extra condition to identify which half of the array is sorted.

---

## Core Idea

In a rotated sorted array, at least one half is always sorted.

For any `mid`, we divide the array into two parts:

```text
left half  = nums[left ... mid]
right half = nums[mid ... right]
```

At least one of these halves will be sorted.

Once we identify the sorted half, we check whether the target lies inside that sorted range.

If it does, we search that half.

Otherwise, we search the other half.

---

## Key Observation

Example:

```python
nums = [4, 5, 6, 7, 0, 1, 2]
```

Initial pointers:

```text
left = 0
right = 6
mid = 3
```

Values:

```text
nums[left] = 4
nums[mid] = 7
nums[right] = 2
```

Since:

```python
nums[left] <= nums[mid]
```

the left half is sorted:

```python
[4, 5, 6, 7]
```

Now we check whether the target lies inside this sorted half.

---

## Optimized Code

```python
def search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

---

## Dry Run

Input:

```python
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
```

Initial state:

```text
left = 0
right = 6
```

---

### Step 1

Calculate mid:

```python
mid = left + (right - left) // 2
```

```text
mid = 0 + (6 - 0) // 2
mid = 3
```

Values:

```text
nums[left] = 4
nums[mid] = 7
nums[right] = 2
```

Check if target is found:

```python
nums[mid] == target
```

```text
7 == 0
False
```

Now check which half is sorted.

```python
nums[left] <= nums[mid]
```

```text
4 <= 7
True
```

So the left half is sorted:

```python
[4, 5, 6, 7]
```

Check whether target lies in the left sorted half:

```python
nums[left] <= target < nums[mid]
```

```text
4 <= 0 < 7
False
```

So target is not in the left half.

Search the right half:

```python
left = mid + 1
```

```text
left = 4
right = 6
```

---

### Step 2

Calculate mid:

```text
left = 4
right = 6

mid = 4 + (6 - 4) // 2
mid = 5
```

Values:

```text
nums[left] = 0
nums[mid] = 1
nums[right] = 2
```

Check if target is found:

```text
nums[mid] = 1
target = 0
```

Not found.

Check whether left half is sorted:

```python
nums[left] <= nums[mid]
```

```text
0 <= 1
True
```

So the left half is sorted:

```python
[0, 1]
```

Check whether target lies in this sorted half:

```python
nums[left] <= target < nums[mid]
```

```text
0 <= 0 < 1
True
```

So target is in the left half.

Move right:

```python
right = mid - 1
```

```text
left = 4
right = 4
```

---

### Step 3

Calculate mid:

```text
left = 4
right = 4

mid = 4 + (4 - 4) // 2
mid = 4
```

Value:

```text
nums[mid] = 0
```

Check:

```python
nums[mid] == target
```

```text
0 == 0
True
```

Return:

```python
4
```

---

## Why Binary Search Works Here

Normal binary search works on a fully sorted array.

This array is not fully sorted because it has been rotated.

However, at every step:

```text
At least one half is sorted.
```

So we can still eliminate half the search space.

That is why the time complexity remains:

```text
O(log n)
```

---

## Follow-Up Questions

### Q1. Why do we use binary search?

Because even though the array is rotated, one half of the array is always sorted.

This lets us eliminate half the search space at every step.

---

### Q2. How do we check if the left half is sorted?

We check:

```python
nums[left] <= nums[mid]
```

If this is true, then the left half is sorted.

---

### Q3. How do we check if the target lies in the left sorted half?

We check:

```python
nums[left] <= target < nums[mid]
```

If true, then the target lies in the sorted left half.

So we move:

```python
right = mid - 1
```

Otherwise, we search the right half:

```python
left = mid + 1
```

---

### Q4. How do we check if the target lies in the right sorted half?

If the right half is sorted, we check:

```python
nums[mid] < target <= nums[right]
```

If true, then the target lies in the right sorted half.

So we move:

```python
left = mid + 1
```

Otherwise, we search the left half:

```python
right = mid - 1
```

---

### Q5. What if the target is not present?

If the target is not found, eventually:

```text
left > right
```

Then the loop ends and we return:

```python
-1
```

---

### Q6. What is the time complexity?

```text
O(log n)
```

Because every step eliminates half of the search space.

---

### Q7. What is the space complexity?

```text
O(1)
```

Because we only use a few variables:

```text
left
right
mid
```

---

### Q8. What if the array is not rotated?

Example:

```python
nums = [1, 2, 3, 4, 5]
```

The same code still works.

The left half or right half checks will behave like normal binary search.

---

### Q9. What if duplicates are present?

This standard solution assumes no duplicate values or mostly unique values.

If duplicates are present, this case can become tricky:

```python
nums = [2, 2, 2, 3, 2, 2]
```

When:

```python
nums[left] == nums[mid] == nums[right]
```

we may not be able to decide which half is sorted.

In that case, one approach is to shrink both ends:

```python
left += 1
right -= 1
```

But this can degrade the worst-case time complexity to:

```text
O(n)
```

---

## Duplicate-Aware Version

If duplicates are allowed, we can write:

```python
def search_with_duplicates(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return True

        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1

        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return False
```

Note:

```text
This duplicate-aware version returns True/False.
It can be modified to return an index if needed.
```

---

## Common Mistakes

### Mistake 1: Applying normal binary search directly

Normal binary search assumes the whole array is sorted.

A rotated sorted array is not fully sorted.

So we first need to identify which half is sorted.

---

### Mistake 2: Forgetting equality in boundary checks

Use:

```python
nums[left] <= nums[mid]
```

not always:

```python
nums[left] < nums[mid]
```

The equality case matters for small windows.

---

### Mistake 3: Incorrect range checks

For left sorted half:

```python
nums[left] <= target < nums[mid]
```

For right sorted half:

```python
nums[mid] < target <= nums[right]
```

These inequalities avoid double-counting `mid`, since `nums[mid]` is already checked earlier.

---

### Mistake 4: Infinite loop

Always update pointers correctly:

```python
left = mid + 1
right = mid - 1
```

Do not write:

```python
left = mid
right = mid
```

in this version, because that can cause an infinite loop.

---

## Interview Explanation

I will use modified binary search.

In a rotated sorted array, at least one half is always sorted.

At every step, I calculate `mid`.

If `nums[mid]` is the target, I return `mid`.

Otherwise, I check whether the left half is sorted using:

```python
nums[left] <= nums[mid]
```

If the left half is sorted, I check whether the target lies between `nums[left]` and `nums[mid]`.

If yes, I search the left half.

Otherwise, I search the right half.

If the left half is not sorted, then the right half must be sorted, and I do the symmetric check.

Since we eliminate half the array each time, the time complexity is `O(log n)` and the space complexity is `O(1)`.

---

## Key Takeaways

```text
Problem:
Search in Rotated Sorted Array

Difficulty:
Medium

Pattern:
Modified Binary Search

Core idea:
At least one half is always sorted.

Check left sorted:
nums[left] <= nums[mid]

Target in left sorted half:
nums[left] <= target < nums[mid]

Target in right sorted half:
nums[mid] < target <= nums[right]

Time Complexity:
O(log n)

Space Complexity:
O(1)

Duplicate follow-up:
If duplicates exist, worst-case can degrade to O(n).
```
