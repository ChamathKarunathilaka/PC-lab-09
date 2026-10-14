"""Check the deliverables and launch arithmetic without claiming a CUDA run."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
notebook = json.loads((root / "SE3082_Lab09_completed.ipynb").read_text(encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
embedded = [source for source in code_cells if source.startswith("%%writefile ")]
expected_files = [
    "hello.cu", "ex2_add.cu", "ex3_blocks.cu", "ex4_threads.cu",
    "ex5_vecmul.cu", "ex6_matrix.cu",
]
assert [source.splitlines()[0].split()[1] for source in embedded] == expected_files
for source, name in zip(embedded, expected_files):
    assert "".join(source.splitlines(True)[1:]) == (root / name).read_text(encoding="utf-8")
    assert "TODO" not in source
for source in code_cells:
    if not source.startswith("%%writefile "):
        compile(source, "<notebook cell>", "exec")

vector_count = 10_000_000
vector_threads = 512
vector_blocks = (vector_count + vector_threads - 1) // vector_threads
assert (vector_blocks, vector_blocks * vector_threads - vector_count) == (19_532, 384)
assert (vector_blocks - 1) * vector_threads == 9_999_872
assert vector_count - (vector_blocks - 1) * vector_threads == 128

rows = cols = 10_000
block_x, block_y = 32, 16
grid_x = (cols + block_x - 1) // block_x
grid_y = (rows + block_y - 1) // block_y
assert (grid_x, grid_y) == (313, 625)
assert grid_x * grid_y * block_x * block_y - rows * cols == 160_000
assert (grid_x - 1) * block_x == 9_984
assert (grid_y - 1) * block_y == 9_984
assert (rows - 1) * cols + (cols - 1) == 99_999_999

# Check 1D and 2D mapping on non-multiple sizes; each valid index must
# be visited once and threads beyond the bounds must be excluded.
for count, threads in [(1, 512), (513, 512), (1_007, 32)]:
    blocks = (count + threads - 1) // threads
    visited = [block * threads + lane for block in range(blocks)
               for lane in range(threads) if block * threads + lane < count]
    assert visited == list(range(count))
for test_rows, test_cols in [(1, 1), (17, 33), (31, 37)]:
    gx = (test_cols + block_x - 1) // block_x
    gy = (test_rows + block_y - 1) // block_y
    visited = [0] * (test_rows * test_cols)
    for by in range(gy):
        for bx in range(gx):
            for ty in range(block_y):
                for tx in range(block_x):
                    row, col = by * block_y + ty, bx * block_x + tx
                    if row < test_rows and col < test_cols:
                        visited[row * test_cols + col] += 1
    assert all(hits == 1 for hits in visited)

print("PASS: Notebook contains all six exact CUDA sources; Python cells parse.")
print("PASS: 1D and 2D index mapping covers valid elements once on edge cases.")
print("PASS: Exercise 5 launches 19,532 blocks, with 128 active threads in the last block.")
print("PASS: Exercise 6 launches a 313 x 625 grid, with 160,000 guarded threads.")
print("NOT RUN: CUDA compilation and GPU execution require a GPU-enabled Colab runtime.")
