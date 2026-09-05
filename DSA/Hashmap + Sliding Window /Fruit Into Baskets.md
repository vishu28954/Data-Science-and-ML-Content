# Fruit Into Baskets

## Problem Statement

You are given an integer array `fruits`, where `fruits[i]` represents the type of fruit on the `i-th` tree.

You have two baskets.

Each basket can hold only one type of fruit, but it can hold unlimited fruits of that type.

Starting from any tree, you must move to the right and pick exactly one fruit from each tree until you cannot place the fruit into any basket.

Return the maximum number of fruits you can pick.

---

## Example

```python
fruits = [1, 2, 1]
```

Output:

```python
3
```

Explanation:

```text
We can pick all fruits: [1, 2, 1]

There are only 2 fruit types:
1 and 2

So the maximum number of fruits is 3.
```

---

## Simplifying the Problem

The story can feel confusing, but the actual problem is simpler.

Two baskets means:

```text
At most 2 different fruit types
```

Moving continuously to the right means:

```text
Contiguous subarray
```

So the problem becomes:

```text
Find the length of the longest contiguous subarray with at most 2 distinct values.
```

---

## Pattern

This problem is solved using:

```text
Sliding Window + Frequency Map
```

Why?

Because we need to maintain a current window that contains at most 2 distinct fruit types.

---

## Window Invariant

A window invariant means the condition that must remain true for the current window.

For this problem, the invariant is:

```text
The current window should contain at most 2 distinct fruit types.
```

So the window is valid when:

```python
len(freq) <= 2
```

The window is invalid when:

```python
len(freq) > 2
```

---

## HashMap Meaning

We use a frequency map.

The HashMap stores:

```text
fruit_type → count inside current window
```

Example:

```python
freq = {
    1: 2,
    2: 1
}
```

This means:

```text
Fruit type 1 appears 2 times in the current window.
Fruit type 2 appears 1 time in the current window.
```

The number of distinct fruit types is:

```python
len(freq)
```

---

## Optimized Approach

We use two pointers:

```python
left = 0
right = 0
```

The current window is:

```python
fruits[left : right + 1]
```

Steps:

```text
1. Expand the window by moving right.
2. Add fruits[right] to the frequency map.
3. If the window has more than 2 distinct fruit types, shrink from the left.
4. While shrinking, reduce the count of fruits[left].
5. If any fruit count becomes 0, delete it from the map.
6. After the window becomes valid, update the maximum length.
```

---

## Optimized Code

```python
def totalFruit(fruits):
    left = 0
    freq = {}
    ans = 0

    for right in range(len(fruits)):
        fruit = fruits[right]
        freq[fruit] = freq.get(fruit, 0) + 1

        while len(freq) > 2:
            left_fruit = fruits[left]
            freq[left_fruit] -= 1

            if freq[left_fruit] == 0:
                del freq[left_fruit]

            left += 1

        ans = max(ans, right - left + 1)

    return ans
```

---

## Dry Run

Input:

```python
fruits = [1, 2, 1, 3]
```

Initial state:

```text
left = 0
freq = {}
ans = 0
```

---

### Step 1

```text
right = 0
fruit = 1
```

Add fruit `1`:

```python
freq = {
    1: 1
}
```

Current window:

```text
[1]
```

Valid because:

```python
len(freq) = 1
```

Update answer:

```text
ans = 1
```

---

### Step 2

```text
right = 1
fruit = 2
```

Add fruit `2`:

```python
freq = {
    1: 1,
    2: 1
}
```

Current window:

```text
[1, 2]
```

Valid because:

```python
len(freq) = 2
```

Update answer:

```text
ans = 2
```

---

### Step 3

```text
right = 2
fruit = 1
```

Add fruit `1`:

```python
freq = {
    1: 2,
    2: 1
}
```

Current window:

```text
[1, 2, 1]
```

Valid because:

```python
len(freq) = 2
```

Update answer:

```text
ans = 3
```

---

### Step 4

```text
right = 3
fruit = 3
```

Add fruit `3`:

```python
freq = {
    1: 2,
    2: 1,
    3: 1
}
```

Current window:

```text
[1, 2, 1, 3]
```

Invalid because:

```python
len(freq) = 3
```

We are allowed only 2 fruit types.

Now shrink from the left.

---

### Shrink 1

Current `left`:

```text
left = 0
fruits[left] = 1
```

Remove one fruit of type `1`:

```python
freq = {
    1: 1,
    2: 1,
    3: 1
}
```

Move left:

```text
left = 1
```

Still invalid because:

```python
len(freq) = 3
```

---

### Shrink 2

Current `left`:

```text
left = 1
fruits[left] = 2
```

Remove one fruit of type `2`:

```python
freq = {
    1: 1,
    2: 0,
    3: 1
}
```

Since count of fruit `2` becomes `0`, delete it:

```python
freq = {
    1: 1,
    3: 1
}
```

Move left:

```text
left = 2
```

Now the window is valid because:

```python
len(freq) = 2
```

Current window:

```text
[1, 3]
```

Update answer:

```text
ans = max(3, 2)
ans = 3
```

Final answer:

```python
3
```

---

## Why Do We Delete a Fruit When Count Becomes Zero?

This is an important follow-up question.

We use:

```python
len(freq)
```

to know how many distinct fruit types exist in the current window.

Suppose we have:

```python
freq = {
    1: 0,
    2: 1,
    3: 1
}
```

Here, `len(freq)` is:

```python
3
```

But fruit type `1` is not actually present in the current window because its count is `0`.

So this is wrong.

That is why when a count becomes zero, we delete that key:

```python
if freq[left_fruit] == 0:
    del freq[left_fruit]
```

Now `len(freq)` correctly represents the number of distinct fruit types in the current window.

---

## Complexity

### Time Complexity

```text
O(n)
```

Why?

The `right` pointer moves from left to right once.

The `left` pointer also only moves forward.

Each fruit enters the window once and leaves the window at most once.

So total work is linear.

---

### Space Complexity

```text
O(1)
```

Why?

At any point, the frequency map stores at most a small number of fruit types.

In this problem, the window is allowed to have at most 2 fruit types, and temporarily it may have 3 before shrinking.

So space is constant.

For the generalized version with `k` baskets:

```text
Space Complexity = O(k)
```

---

## Follow-Up Questions

### Q1. What is this problem really asking?

It is asking for the length of the longest contiguous subarray with at most 2 distinct values.

The two baskets represent the two allowed fruit types.

---

### Q2. Why do we use sliding window?

Because the problem asks for the longest contiguous subarray satisfying a condition.

The condition is:

```text
At most 2 distinct fruit types.
```

So we maintain a window and keep it valid.

---

### Q3. What does the HashMap store?

The HashMap stores:

```text
fruit_type → count inside current window
```

We need counts because when we shrink the window from the left, we need to know whether that fruit type still exists in the current window.

---

### Q4. Why do we delete a fruit when its count becomes zero?

Because `len(freq)` is used to count the number of distinct fruit types.

If a fruit has count `0` but remains in the map, `len(freq)` will be incorrect.

So we delete it.

---

### Q5. What if there are `k` baskets instead of 2?

Then the problem becomes:

```text
Find the longest contiguous subarray with at most k distinct values.
```

The code changes from:

```python
while len(freq) > 2:
```

to:

```python
while len(freq) > k:
```

---

### Q6. Why is the time complexity O(n) even though there is a while loop inside the for loop?

Because both pointers only move forward.

The `right` pointer moves across the array once.

The `left` pointer also moves across the array at most once.

So total pointer movement is:

```text
O(n) + O(n) = O(n)
```

Each element is added once and removed at most once.

---

### Q7. What happens if the input is empty?

If:

```python
fruits = []
```

then the answer is:

```python
0
```

The code handles this because the loop never runs and `ans` remains `0`.

---

## Interview Explanation

The problem can be simplified to finding the longest contiguous subarray with at most two distinct values.

I use a sliding window with two pointers, `left` and `right`.

The frequency map stores the count of each fruit type inside the current window.

I expand the window by moving `right`. If the number of distinct fruit types becomes more than two, I shrink the window from the left until it becomes valid again.

Whenever the window is valid, I update the maximum length.

The time complexity is `O(n)` because each fruit enters and leaves the window at most once. The space complexity is `O(1)` because we store at most a few fruit types.

---

## Key Takeaways

```text
Problem:
Fruit Into Baskets

Simplified problem:
Longest contiguous subarray with at most 2 distinct values

Pattern:
Sliding Window + Frequency Map

HashMap stores:
fruit_type → count inside current window

Valid condition:
len(freq) <= 2

Invalid condition:
len(freq) > 2

Important deletion:
if freq[left_fruit] == 0:
    del freq[left_fruit]

Time Complexity:
O(n)

Space Complexity:
O(1)

Generalized version:
Longest subarray with at most k distinct values
```
