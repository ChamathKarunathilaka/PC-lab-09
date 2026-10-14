# Lab 09 verification and expected output

The completed submission is `SE3082_Lab09_completed.ipynb`. It contains all six CUDA
programs, compilation and run cells, the Exercise 4 limit experiment, and written
answers. The `.cu` files are also provided separately.

## Checks run locally

`python verify_local.py` passes. It confirms that the notebook embeds the exact
source files, all Python cells parse, the 1D and 2D mappings cover every valid
element once on boundary cases, and the grid sizes match the lab sheet.

CUDA compilation and GPU execution were **not run locally**. This machine has no
`nvcc` or `nvidia-smi`; WSL access was denied, and the Docker Linux engine is not
running. No GPU output or zero-mismatch result is claimed as observed.

## Expected results from the lab handout

| Exercise | Expected result after running in Colab |
| --- | --- |
| 1 | `hello.cu` prints four GPU greeting lines, in any order. |
| 2 | `Result is 9` |
| 3 | `Verification: 0 mismatches out of 512` |
| 4 | `Verification: 0 mismatches out of 512`; the optional 2,048-thread launch reports `invalid configuration argument`. |
| 5 | `Launching 19532 blocks x 512 threads`; first of the final 1,000: `9999000) 76 x 52 = 3952`; last: `9999999) 31 x 73 = 2263`; `Verification: 0 mismatches out of 10000000`. |
| 6 | `Grid: 313 x 625 blocks`; `C[9999][9000]: 29 x 56 = 1624`; `C[9999][9999]: 86 x 21 = 1806`; `Verification: 0 mismatches out of 100000000`. |

The handout's sample numbers assume the Colab/Linux C `rand()` sequence and that
the entire first input array is generated before the second.

## Remaining Colab step

Open the notebook in Colab, select a GPU runtime, run all cells, and check that
the observed output matches the expected results above. Rename the notebook
with your student ID and save or download it for submission.
