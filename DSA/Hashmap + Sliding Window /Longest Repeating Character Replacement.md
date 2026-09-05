# Longest Repeating Character Replacement

## Problem Statement

You are given a string `s` and an integer `k`.

You can choose any character in the string and replace it with any uppercase English character.

Return the length of the longest substring containing the same letter after performing at most `k` replacements.

---

## Example

```python
s = "AABABBA"
k = 1
```

Output:

```python
4
```

Explanation:

One possible substring is:

```text
"AABA"
```

If we replace `B` with `A`, it becomes:

```text
"AAAA"
```

So the maximum length is:

```python
4
```

---

## Pattern

This problem is solved using:

```text
Sliding Window + Frequency Map
```

We maintain a window and check whether it can be converted into a substring containing only one repeated character using at most `k` replacements.

---

## Core Idea

For any current window, the best character to keep is the character that appears the most.

All other characters need to be replaced.

Example:

```text
Window = "AABA"
```

Frequency:

```text
A → 3
B → 1
```

The best character to keep is `A`.

Window length:

```text
4
```

Maximum frequency:

```text
3
```

Replacements needed:

```text
window length - max frequency
= 4 - 3
= 1
```

If `k = 1`, this window is valid.

---

## Window Invariant

The window is valid when:

```text
window_length - max_freq <= k
```

The window is invalid when:

```text
window_length - max_freq > k
```

Why?

Because:

```text
max_freq characters can remain unchanged
all other characters need to be replaced
```

So:

```text
characters_to_replace = window_length - max_freq
```

---

## HashMap Meaning

We use a frequency map.

The HashMap stores:

```text
character → count inside current window
```

Example:

```python
freq = {
    "A": 3,
    "B": 1
}
```

This means:

```text
A appears 3 times inside the current window.
B appears 1 time inside the current window.
```

---

## Important Variables

```python
left
```

Left boundary of the window.

```python
right
```

Right boundary of the window.

```python
freq
```

Frequency map of characters inside the current window.

```python
max_freq
```

Highest frequency of any character seen inside the current window.

```python
ans
```

Maximum valid window length found so far.

---

## Optimized Code

```python
def characterReplacement(s, k):
    left = 0
    freq = {}
    max_freq = 0
    ans = 0

    for right, ch in enumerate(s):
        freq[ch] = freq.get(ch, 0) + 1
        max_freq = max(max_freq, freq[ch])

        while (right - left + 1) - max_freq > k:
            left_ch = s[left]
            freq[left_ch] -= 1
            left += 1

        ans = max(ans, right - left + 1)

    return ans
```

---

## Dry Run

Input:

```python
s = "AABABBA"
k = 1
```

Initial state:

```text
left = 0
freq = {}
max_freq = 0
ans = 0
```

---

### Step 1

```text
right = 0
ch = "A"
```

Add `A`:

```python
freq = {
    "A": 1
}
```

Update:

```text
max_freq = 1
```

Current window:

```text
"A"
```

Window length:

```text
1
```

Replacements needed:

```text
1 - 1 = 0
```

Valid.

Update answer:

```text
ans = 1
```

---

### Step 2

```text
right = 1
ch = "A"
```

Add `A`:

```python
freq = {
    "A": 2
}
```

Update:

```text
max_freq = 2
```

Current window:

```text
"AA"
```

Window length:

```text
2
```

Replacements needed:

```text
2 - 2 = 0
```

Valid.

Update answer:

```text
ans = 2
```

---

### Step 3

```text
right = 2
ch = "B"
```

Add `B`:

```python
freq = {
    "A": 2,
    "B": 1
}
```

Current `max_freq`:

```text
2
```

Current window:

```text
"AAB"
```

Window length:

```text
3
```

Replacements needed:

```text
3 - 2 = 1
```

Valid because:

```text
1 <= k
```

Update answer:

```text
ans = 3
```

---

### Step 4

```text
right = 3
ch = "A"
```

Add `A`:

```python
freq = {
    "A": 3,
    "B": 1
}
```

Update:

```text
max_freq = 3
```

Current window:

```text
"AABA"
```

Window length:

```text
4
```

Replacements needed:

```text
4 - 3 = 1
```

Valid.

Update answer:

```text
ans = 4
```

---

### Step 5

```text
right = 4
ch = "B"
```

Add `B`:

```python
freq = {
    "A": 3,
    "B": 2
}
```

Current `max_freq`:

```text
3
```

Current window:

```text
"AABAB"
```

Window length:

```text
5
```

Replacements needed:

```text
5 - 3 = 2
```

Invalid because:

```text
2 > k
```

So we shrink from the left.

Remove `s[left]`:

```text
s[left] = "A"
```

Update frequency:

```python
freq = {
    "A": 2,
    "B": 2
}
```

Move left:

```text
left = 1
```

Current window:

```text
"ABAB"
```

Window length:

```text
4
```

Using stored `max_freq = 3`:

```text
replacements needed = 4 - 3 = 1
```

Valid according to the algorithm.

Answer remains:

```text
ans = 4
```

---

## Why Do We Not Decrease `max_freq` While Shrinking?

This is the most important follow-up question.

In the code, when we shrink the window, we decrease the frequency of the left character:

```python
freq[left_ch] -= 1
```

But we do not decrease or recompute:

```python
max_freq
```

This may make `max_freq` slightly stale.

However, this is acceptable.

The reason is:

```text
We are interested in the maximum valid window length, not necessarily maintaining the exact max frequency at every moment.
```

A stale `max_freq` may delay shrinking slightly, but it does not cause us to miss the correct answer.

The window length recorded is still controlled by the largest frequency seen during the expansion process.

---

## Simpler Interview Answer for `max_freq`

If the interviewer asks:

```text
Why don't we decrease max_freq when left moves?
```

Say:

```text
We do not decrease max_freq for efficiency.

Even if max_freq becomes stale, the algorithm still returns the correct maximum length because max_freq is used to control the maximum window size, and both pointers only move forward.

If needed, we could recompute max_freq from the frequency map after shrinking. Since the string contains only uppercase English letters, recomputing over 26 characters would still be O(n). But the optimized version avoids this recomputation.
```

---

## Follow-Up Questions

### Q1. What is the key condition for a valid window?

The valid condition is:

```text
window_length - max_freq <= k
```

This means the number of characters that need replacement is at most `k`.

---

### Q2. Why do we subtract `max_freq` from window length?

Because the most frequent character can stay unchanged.

All other characters need to be replaced.

So:

```text
characters_to_replace = window_length - max_freq
```

---

### Q3. What does the HashMap store?

The HashMap stores:

```text
character → count inside current window
```

We need this to track the frequency of each character in the current window.

---

### Q4. Why do we shrink the window?

We shrink when:

```text
window_length - max_freq > k
```

This means the window requires more than `k` replacements, so it is invalid.

---

### Q5. Why don't we decrease `max_freq` when left moves?

Because recomputing it every time is unnecessary for finding the maximum length.

A stale `max_freq` may delay shrinking slightly, but it does not affect the final maximum answer.

If the interviewer wants a stricter version, we can recompute `max_freq` from the frequency map because there are only 26 uppercase English letters.

---

### Q6. What is the time complexity?

```text
O(n)
```

The `right` pointer scans the string once.

The `left` pointer only moves forward.

Each character enters and leaves the window at most once.

---

### Q7. What is the space complexity?

```text
O(1)
```

Because the input contains only uppercase English letters.

So the frequency map can store at most 26 characters.

More general answer:

```text
If the character set is not fixed, the space complexity can be O(number of distinct characters), up to O(n).
```

---

### Q8. What if `k = 0`?

Then we cannot replace any character.

The problem becomes:

```text
Find the longest substring that already contains only one repeated character.
```

Example:

```python
s = "AABBB"
k = 0
```

Output:

```python
3
```

because:

```text
"BBB"
```

is the longest substring containing the same character.

The same code works.

---

### Q9. What if `k >= len(s)`?

Then we can replace all characters if needed.

The answer is:

```python
len(s)
```

The same code also works.

---

## Complexity

### Time Complexity

```text
O(n)
```

Why?

The string is scanned once by the `right` pointer.

The `left` pointer also only moves forward.

HashMap operations are average `O(1)`.

Since the character set is fixed to uppercase English letters, frequency-map operations are constant time.

---

### Space Complexity

```text
O(1)
```

Why?

There are only 26 uppercase English letters.

So the frequency map size is bounded by 26.

---

## Interview Explanation

I will use a sliding window.

For any current window, the best character to keep unchanged is the most frequent character in that window.

All other characters need to be replaced.

So the number of replacements needed is:

```text
window length - max frequency
```

If this value is less than or equal to `k`, the window is valid.

If it is greater than `k`, I shrink the window from the left.

I maintain a frequency map to store character counts and a `max_freq` variable to track the highest frequency.

The time complexity is `O(n)` because each character is processed at most a constant number of times.

The space complexity is `O(1)` because the input contains only uppercase English letters.

---

## Key Takeaways

```text
Problem:
Longest Repeating Character Replacement

Difficulty:
Medium

Pattern:
Sliding Window + Frequency Map

HashMap stores:
character → count inside current window

Key variable:
max_freq

Valid condition:
window_length - max_freq <= k

Invalid condition:
window_length - max_freq > k

Core idea:
Keep the most frequent character and replace the rest.

Most important follow-up:
Why don't we decrease max_freq when left moves?

Time Complexity:
O(n)

Space Complexity:
O(1)
```
