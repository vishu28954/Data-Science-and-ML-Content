# Number of Islands

## Problem Statement

You are given a 2D grid containing:

```text
"1" → land
"0" → water
```

Return the number of islands.

An island is formed by connected land cells.

Land cells are connected only horizontally or vertically.

Diagonal connection does not count.

---

## Example

```python
grid = [
    ["1", "1", "0", "0"],
    ["1", "1", "0", "0"],
    ["0", "0", "1", "0"],
    ["0", "0", "0", "1"]
]
```

Output:

```python
3
```

Explanation:

```text
Island 1:
Top-left block of connected 1s

Island 2:
Single 1 at row 2, column 2

Island 3:
Single 1 at row 3, column 3
```

So the total number of islands is:

```python
3
```

---

## Difficulty

```text
Medium
```

---

## Pattern

```text
DFS / BFS / Connected Components
```

Even though this looks like a matrix problem, we can think of it as a graph problem.

Each land cell is like a graph node.

Each land cell can connect to its neighboring land cells in four directions:

```text
up
down
left
right
```

So the problem becomes:

```text
Count the number of connected components of land cells.
```

---

## Core Idea

We scan every cell in the grid.

Whenever we find an unvisited land cell `"1"`, that means we have found a new island.

So we increment the island count.

Then we run DFS or BFS from that cell to mark the entire connected island as visited.

This prevents counting the same island multiple times.

---

## Why DFS Works

Suppose we find a land cell:

```text
grid[r][c] = "1"
```

This means a new island starts here.

From this cell, DFS explores all connected land cells:

```text
up
down
left
right
```

Every connected land cell belongs to the same island.

So we mark all of them as visited.

In this implementation, we mark visited cells by changing:

```python
"1" → "0"
```

---

## DFS Code

```python
def numIslands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    islands = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        if grid[r][c] != "1":
            return

        grid[r][c] = "0"

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                islands += 1
                dfs(r, c)

    return islands
```

---

## Explanation of Important Lines

### Boundary Check

```python
if r < 0 or r >= rows or c < 0 or c >= cols:
    return
```

This checks whether the current cell is outside the grid.

If it is outside, we stop.

---

### Water or Already Visited Check

```python
if grid[r][c] != "1":
    return
```

This stops DFS if the current cell is not land.

It may be:

```text
"0" → water
"0" → already visited land changed to water
```

---

### Mark as Visited

```python
grid[r][c] = "0"
```

This marks the current land cell as visited.

We do this so we do not count or visit the same land cell again.

---

### Explore Four Directions

```python
dfs(r + 1, c)
dfs(r - 1, c)
dfs(r, c + 1)
dfs(r, c - 1)
```

These calls explore:

```text
down
up
right
left
```

Diagonal cells are not included.

---

## Dry Run

Input:

```python
grid = [
    ["1", "1", "0"],
    ["1", "0", "0"],
    ["0", "0", "1"]
]
```

Initial state:

```text
islands = 0
```

---

### Step 1

Start scanning from top-left.

At cell:

```text
(r, c) = (0, 0)
```

Value:

```python
grid[0][0] = "1"
```

This is unvisited land.

So we found a new island.

Update:

```text
islands = 1
```

Now run DFS from `(0, 0)`.

DFS marks all connected land cells as visited:

```text
(0, 0)
(0, 1)
(1, 0)
```

These all belong to the same island.

After DFS, those cells become `"0"`.

---

### Step 2

Continue scanning.

The cells from the first island are now already visited.

Later we reach:

```text
(r, c) = (2, 2)
```

Value:

```python
grid[2][2] = "1"
```

This is unvisited land.

So we found another island.

Update:

```text
islands = 2
```

Run DFS from `(2, 2)`.

It has no connected land neighbors.

So it is a single-cell island.

---

## Final Answer

```python
2
```

---

## BFS Version

DFS and BFS both solve this problem.

DFS uses recursion or stack.

BFS uses a queue.

```python
from collections import deque

def numIslands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    islands = 0

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                islands += 1
                grid[r][c] = "0"

                queue = deque([(r, c)])

                while queue:
                    row, col = queue.popleft()

                    for dr, dc in directions:
                        nr = row + dr
                        nc = col + dc

                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                            grid[nr][nc] = "0"
                            queue.append((nr, nc))

    return islands
```

---

## Follow-Up Questions

### Q1. Why is this a graph problem?

Because each land cell is like a node.

Each land cell is connected to neighboring land cells in four directions:

```text
up
down
left
right
```

So counting islands is equivalent to counting connected components in a graph.

---

### Q2. Why do we run DFS or BFS?

DFS/BFS helps us visit all land cells connected to the current land cell.

All connected land cells belong to the same island.

So once we start DFS/BFS from one land cell, we mark the whole island as visited.

---

### Q3. Why not count every `"1"` as a separate island?

Because multiple `"1"` cells may be connected.

Example:

```python
[
    ["1", "1"],
    ["1", "1"]
]
```

This grid has four land cells, but only one island.

So we count an island only when we find an unvisited land cell.

---

### Q4. Why do we change `"1"` to `"0"`?

We change `"1"` to `"0"` to mark it as visited.

This prevents revisiting the same land cell and counting the same island again.

---

### Q5. What if we cannot modify the input grid?

If we cannot modify the grid, we can use a separate `visited` set.

Example:

```python
visited = set()
```

Then mark visited cells as:

```python
visited.add((r, c))
```

Instead of changing:

```python
grid[r][c] = "0"
```

---

### Q6. Do diagonal connections count?

Usually no.

Only four directions count:

```text
up
down
left
right
```

Diagonal cells are not connected in the standard version.

In an interview, clarify:

```text
Should diagonal land cells be considered connected?
```

If diagonals are allowed, add four more directions:

```python
directions = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1)
]
```

---

### Q7. DFS or BFS: which one is better?

Both are correct.

DFS is usually simpler to code recursively.

BFS avoids deep recursion issues and uses a queue.

For very large grids, recursive DFS may hit recursion depth limits in Python, so BFS or iterative DFS can be safer.

---

### Q8. What is the time complexity?

```text
O(rows * cols)
```

Why?

Every cell is visited at most once.

Even though DFS/BFS explores neighbors, each cell becomes visited and is not processed again.

---

### Q9. What is the space complexity?

For recursive DFS:

```text
O(rows * cols)
```

in the worst case due to recursion stack.

This happens when the whole grid is land.

For BFS:

```text
O(rows * cols)
```

in the worst case due to the queue.

---

## Common Mistakes

### Mistake 1: Forgetting to mark visited

If we do not mark visited cells, DFS may revisit the same land again and again.

This can cause infinite recursion or overcounting.

---

### Mistake 2: Counting every land cell as an island

Connected land cells belong to the same island.

Only unvisited land should start a new island count.

---

### Mistake 3: Including diagonals by mistake

The standard version only uses four directions:

```text
up
down
left
right
```

Do not include diagonals unless the interviewer says so.

---

### Mistake 4: Boundary errors

Always check:

```python
0 <= r < rows
0 <= c < cols
```

before accessing:

```python
grid[r][c]
```

---

### Mistake 5: Forgetting empty grid case

If the grid is empty:

```python
grid = []
```

Return:

```python
0
```

---

## Interview Explanation

I will treat the grid as a graph.

Each land cell is a node, and it is connected to neighboring land cells in four directions.

I scan every cell in the grid.

Whenever I find a land cell `"1"`, that means I have found a new island, so I increment the island count.

Then I run DFS from that cell to mark the entire connected island as visited by changing all connected `"1"` cells to `"0"`.

This ensures each island is counted only once.

The time complexity is `O(rows * cols)` because every cell is visited at most once.

The space complexity is `O(rows * cols)` in the worst case due to recursion stack.

---

## Key Takeaways

```text
Problem:
Number of Islands

Difficulty:
Medium

Pattern:
DFS / BFS / Connected Components

Grid cell:
Graph node

Land:
"1"

Water:
"0"

When we find unvisited land:
islands += 1
run DFS/BFS to mark the full island visited

Directions:
up, down, left, right

Time Complexity:
O(rows * cols)

Space Complexity:
O(rows * cols) worst case

Important follow-up:
If we cannot modify the grid, use a visited set.
```
