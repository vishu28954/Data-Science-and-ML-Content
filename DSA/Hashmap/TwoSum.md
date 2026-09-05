# Two Sum Using HashMap

## Problem Statement

Given an integer array `nums` and an integer `target`, return the indices of two numbers such that they add up to `target`.

Example:

```python
nums = [2, 7, 11, 15]
target = 9
```

Output:

```python
[0, 1]
```

Explanation:

```text
nums[0] + nums[1] = 2 + 7 = 9
```

---

## Brute Force Approach

The simplest approach is to check every possible pair.

```python
def twoSum(nums, target):
    n = len(nums)

    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]

    return []
```

### Complexity

```text
Time Complexity: O(n²)
Space Complexity: O(1)
```

Why?

Because for every element, we check many other elements to form a pair.

---

## Optimized Approach Using HashMap

We need two numbers such that:

```text
a + b = target
```

If the current number is `b`, then the number we need is:

```text
a = target - b
```

So for every number, we calculate:

```python
need = target - num
```

Then we ask:

```text
Have I already seen this needed number before?
```

To answer this quickly, we use a HashMap.

---

## HashMap Meaning

We store:

```text
number → index
```

Example:

```python
seen = {
    2: 0,
    7: 1
}
```

This means:

```text
Number 2 was seen at index 0
Number 7 was seen at index 1
```

---

## Optimized Code

```python
def twoSum(nums, target):
    seen = {}

    for i, num in enumerate(nums):
        need = target - num

        if need in seen:
            return [seen[need], i]

        seen[num] = i

    return []
```

---

## Dry Run

Input:

```python
nums = [2, 7, 11, 15]
target = 9
```

Start:

```python
seen = {}
```

### Step 1

```text
i = 0
num = 2
need = 9 - 2 = 7
```

Check:

```text
Is 7 in seen?
No
```

Store:

```python
seen = {
    2: 0
}
```

---

### Step 2

```text
i = 1
num = 7
need = 9 - 7 = 2
```

Check:

```text
Is 2 in seen?
Yes
```

`seen[2] = 0`

So return:

```python
[0, 1]
```

---

## Why Do We Check Before Inserting?

This is important.

Consider:

```python
nums = [3, 3]
target = 6
```

If we insert the current number before checking, we may accidentally use the same element twice.

Correct order:

```python
if need in seen:
    return [seen[need], i]

seen[num] = i
```

This ensures the current number only matches with a number from a previous index.

---

## Complexity

```text
Time Complexity: O(n)
Space Complexity: O(n)
```

### Why Time Complexity is O(n)

We scan the array once.

For each element:

```text
HashMap lookup → average O(1)
HashMap insertion → average O(1)
```

So total time is:

```text
O(n)
```

### Why Space Complexity is O(n)

In the worst case, we may store every number in the HashMap.

So space complexity is:

```text
O(n)
```

---

## Important Follow-Up Questions

### Q1. Why do we use a HashMap?

We use a HashMap because it allows us to check whether the required complement was already seen in average `O(1)` time.

Instead of searching the previous elements again and again, we store them.

---

### Q2. What does the HashMap store?

The HashMap stores:

```text
number → index
```

We need the index because the problem asks us to return indices, not just values.

---

### Q3. Why not use a Set?

A Set can only tell us whether a number exists.

But this problem asks for indices.

So we need:

```text
number → index
```

Therefore, we use a HashMap instead of a Set.

---

### Q4. What if duplicate numbers exist?

The solution still works.

Example:

```python
nums = [3, 3]
target = 6
```

At the first `3`, the map is empty, so we store:

```python
seen = {
    3: 0
}
```

At the second `3`, we need another `3`.

Since `3` already exists in the map, we return:

```python
[0, 1]
```

---

### Q5. What if there is no answer?

If the problem does not guarantee an answer, we can return:

```python
[]
```

or

```python
None
```

depending on the requirement.

In an interview, clarify this by asking:

```text
If no valid pair exists, should I return an empty list?
```

---

### Q6. What if the array is sorted?

If the array is sorted, we can also solve it using two pointers.

Two-pointer approach:

```text
left = 0
right = len(nums) - 1
```

If:

```text
nums[left] + nums[right] < target
```

move `left` forward.

If:

```text
nums[left] + nums[right] > target
```

move `right` backward.

This gives:

```text
Time Complexity: O(n)
Space Complexity: O(1)
```

So:

```text
Unsorted array → HashMap
Sorted array → Two Pointers
```

---

## Interview Explanation

The brute-force solution checks all pairs and takes `O(n²)` time.

We can optimize this using a HashMap. For each number, we calculate the complement needed to reach the target:

```text
need = target - num
```

If this complement already exists in the HashMap, we return the stored index and the current index.

Otherwise, we store the current number with its index.

This gives `O(n)` average time complexity and `O(n)` space complexity.

---

## Key Takeaways

```text
Pattern: HashMap

Map stores: number → index

Core idea: need = target - num

Check before inserting to avoid using the same element twice.

Time Complexity: O(n)

Space Complexity: O(n)

Sorted-array follow-up: use two pointers.
```
