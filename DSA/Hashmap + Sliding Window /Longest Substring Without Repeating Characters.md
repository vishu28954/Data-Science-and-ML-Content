# Longest Substring Without Repeating Characters

## Problem Statement

Given a string `s`, find the length of the longest substring without repeating characters.

Example:

```python
s = "abcabcbb"
```

Output:

```python
3
```

Explanation:

```text
The longest substring without repeating characters is "abc".
Length = 3
```

---

## Important Keyword

The important word in the problem is:

```text
substring
```

A substring means a continuous part of the string.

Example substrings of `"abcde"`:

```text
"abc"
"bcd"
"de"
```

Not a substring:

```text
"ace"
```

because the characters are not continuous.

---

## Pattern

This problem is solved using:

```text
Sliding Window + HashMap
```

Why?

Because we need to find the longest continuous window where:

```text
No character repeats.
```

---

## Brute Force Approach

The brute-force approach is to check all possible substrings.

For each substring, check whether it contains duplicate characters.

This is inefficient.

Approximate complexity:

```text
Time Complexity: O(n²) or O(n³)
```

depending on implementation.

---

## Optimized Approach

We use two pointers:

```python
left = 0
right = 0
```

The current window is:

```python
s[left : right + 1]
```

We also use a HashMap:

```text
character → last seen index
```

Example:

```python
last_seen = {
    "a": 0,
    "b": 1,
    "c": 2
}
```

This means:

```text
a was last seen at index 0
b was last seen at index 1
c was last seen at index 2
```

---

## Optimized Code

```python
def lengthOfLongestSubstring(s):
    left = 0
    last_seen = {}
    ans = 0

    for right, ch in enumerate(s):
        if ch in last_seen:
            left = max(left, last_seen[ch] + 1)

        last_seen[ch] = right
        ans = max(ans, right - left + 1)

    return ans
```

---

## Dry Run

Input:

```python
s = "abcabcbb"
```

Initial state:

```text
left = 0
last_seen = {}
ans = 0
```

---

### Step 1

```text
right = 0
ch = "a"
```

`a` is not seen before.

Current window:

```text
"a"
```

Update:

```python
last_seen = {
    "a": 0
}
```

```text
ans = 1
```

---

### Step 2

```text
right = 1
ch = "b"
```

`b` is not seen before.

Current window:

```text
"ab"
```

Update:

```python
last_seen = {
    "a": 0,
    "b": 1
}
```

```text
ans = 2
```

---

### Step 3

```text
right = 2
ch = "c"
```

`c` is not seen before.

Current window:

```text
"abc"
```

Update:

```python
last_seen = {
    "a": 0,
    "b": 1,
    "c": 2
}
```

```text
ans = 3
```

---

### Step 4

```text
right = 3
ch = "a"
```

`a` was last seen at index `0`.

So move `left`:

```python
left = max(left, last_seen["a"] + 1)
```

```text
left = max(0, 0 + 1)
left = 1
```

Current window becomes:

```text
"bca"
```

Update:

```python
last_seen = {
    "a": 3,
    "b": 1,
    "c": 2
}
```

```text
ans = 3
```

The answer remains `3`.

---

## Why Do We Use `max`?

This is the most important follow-up question.

Code:

```python
left = max(left, last_seen[ch] + 1)
```

We use `max` because the repeated character may be outside the current window.

The left pointer should never move backward.

---

## Example: Why `max` is needed

Consider:

```python
s = "abba"
```

Indices:

```text
Index: 0 1 2 3
Char:  a b b a
```

At index `2`, character `b` repeats.

Previous `b` was at index `1`.

So:

```text
left = 2
```

Now the current window is:

```text
"b"
```

At index `3`, we see `a`.

Previous `a` was at index `0`.

If we simply write:

```python
left = last_seen["a"] + 1
```

then:

```text
left = 1
```

But `left` was already `2`.

This moves `left` backward, which is incorrect.

So we use:

```python
left = max(left, last_seen[ch] + 1)
```

This ensures:

```text
left never moves backward.
```

---

## Why Time Complexity is O(n)

At first, sliding window may look like nested logic, but each character is processed a constant number of times.

The `right` pointer moves from left to right once.

The `left` pointer also only moves forward.

So total work is linear.

```text
Time Complexity: O(n)
```

---

## Space Complexity

The HashMap stores characters and their latest indices.

In the worst case, all characters are unique.

```text
Space Complexity: O(n)
```

More precise answer:

```text
Space Complexity: O(min(n, character_set_size))
```

If the character set is fixed, such as ASCII, space can be considered `O(1)`.

---

## Follow-Up Questions

### Q1. Why do we use sliding window?

Because the problem asks for the longest continuous substring satisfying a condition.

The condition is:

```text
No duplicate characters inside the window.
```

So we maintain a valid window using left and right pointers.

---

### Q2. What does the HashMap store?

The HashMap stores:

```text
character → last seen index
```

This helps us quickly know where the repeated character appeared previously.

---

### Q3. Why not use a frequency map?

A frequency map can also solve the problem.

But a last-seen-index map is cleaner because when a duplicate appears, we can directly move `left` after the previous occurrence of that character.

Frequency map usually shrinks one step at a time.

Last-seen index can jump directly.

---

### Q4. Why do we use `max(left, last_seen[ch] + 1)`?

Because the previous occurrence of the character may be outside the current window.

We should never move `left` backward.

Example:

```python
s = "abba"
```

Without `max`, the algorithm can incorrectly move `left` backward.

---

### Q5. What if the input string is empty?

Example:

```python
s = ""
```

Answer:

```python
0
```

The code handles this because the loop does not run and `ans` remains `0`.

---

### Q6. What if all characters are the same?

Example:

```python
s = "bbbb"
```

The longest substring without repeating characters is:

```text
"b"
```

Answer:

```python
1
```

---

### Q7. What if all characters are unique?

Example:

```python
s = "abcdef"
```

The whole string has no repeating characters.

Answer:

```python
6
```

---

### Q8. What is the time and space complexity?

```text
Time Complexity: O(n)
Space Complexity: O(n)
```

More precise space complexity:

```text
O(min(n, character_set_size))
```

---

## Interview Explanation

The brute-force solution would check all substrings and verify whether they contain duplicates, which is inefficient.

I will use a sliding window. The invariant is that the current window should not contain duplicate characters.

I maintain a `left` pointer and scan the string using the `right` pointer. I also store each character's last seen index in a HashMap. If the current character was seen before, I move `left` to one position after its previous occurrence, but I use `max` to ensure `left` never moves backward.

At every step, I update the maximum window length.

This gives `O(n)` time complexity and `O(n)` space complexity in the worst case.

---

## Key Takeaways

```text
Problem:
Longest substring without repeating characters

Pattern:
Sliding Window + HashMap

HashMap stores:
character → last seen index

Important line:
left = max(left, last_seen[ch] + 1)

Why max?
Because left should never move backward.

Time Complexity:
O(n)

Space Complexity:
O(n)
```
