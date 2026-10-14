"""Package the completed CUDA sources as a self-contained Colab notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SE3082_Lab09_completed.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def markdown(source):
    return cell("markdown", source)


def code(source):
    return cell("code", source)


def program(filename, title, notes=None):
    cells = [markdown(f"## {title}\n" + (f"\n{notes}\n" if notes else ""))]
    source = (ROOT / filename).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile {filename}\n{source}"))
    exe = Path(filename).stem
    cells.append(code(
        f"import subprocess\n"
        f"subprocess.run([NVCC, f'-arch={{ARCH}}', '{filename}', '-o', '{exe}'], check=True)\n"
        f"subprocess.run(['./{exe}'], check=True)\n"
    ))
    return cells


cells = [
    markdown("# SE3082 Lab 09 — Introduction to CUDA\n\n"
             "Completed Google Colab notebook. Select a GPU runtime under **Runtime → Change runtime type**, "
             "then run the cells in order. Rename this notebook with your student ID before submitting. "
             "The output cells are blank until you run it on a CUDA GPU.\n"),
    markdown("## Exercise 1 — Set up CUDA\n\n"
             "The first cell reports the GPU and the CUDA compiler. The architecture is detected from "
             "the GPU's compute capability so the compilation cells work with a T4, L4, A100, or H100.\n"),
    code("import shutil, subprocess\n"
         "subprocess.run(['nvidia-smi'], check=True)\n"
         "NVCC = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'\n"
         "subprocess.run([NVCC, '--version'], check=True)\n"
         "try:\n"
         "    capability = subprocess.check_output(\n"
         "        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],\n"
         "        text=True, stderr=subprocess.DEVNULL\n"
         "    ).splitlines()[0].strip()\n"
         "except (subprocess.CalledProcessError, IndexError):\n"
         "    gpu_name = subprocess.check_output(\n"
         "        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True\n"
         "    ).splitlines()[0]\n"
         "    capability = next((cap for name, cap in [('T4', '7.5'), ('L4', '8.9'),\n"
         "                       ('A100', '8.0'), ('H100', '9.0')] if name in gpu_name), None)\n"
         "    if capability is None:\n"
         "        raise RuntimeError(f'Unknown GPU {gpu_name}: set ARCH manually')\n"
         "ARCH = 'sm_' + capability.replace('.', '')\n"
         "print(f'Compute capability: {capability}; compiler architecture: {ARCH}')\n"),
]

cells += program("hello.cu", "Exercise 1 — GPU smoke test",
                 "Expected: four greeting lines, one per GPU thread; their order can vary.")
cells += program("ex2_add.cu", "Exercise 2 — Add two integers",
                 "Expected: `Result is 9`.")
cells.append(markdown(
    "**Think answers:** Host variables `a`, `b`, and `c` live in CPU memory; the kernel "
    "uses device addresses, held in `d_a`, `d_b`, and `d_c`. `&a` gives the address "
    "of the host integer to copy, whereas `d_a` already contains a device address. "
    "The launch `<<<1, 1>>>` runs one thread. `<<<1, 10>>>` would run ten, but all ten "
    "would write the same `c` location, creating a data race even though the arithmetic result is the same.\n"
))
cells += program("ex3_blocks.cu", "Exercise 3 — 512 blocks, one thread each",
                 "Expected: 512 result lines and `Verification: 0 mismatches out of 512`.")
cells.append(markdown(
    "**Prediction:** `<<<512, 1>>>` launches 512 blocks and 512 threads. A block with "
    "one thread still occupies a warp with 32 lanes, leaving 31 lanes inactive.\n"
))
cells += program("ex4_threads.cu", "Exercise 4 — One block, 512 threads",
                 "Expected: 512 result lines and `Verification: 0 mismatches out of 512`.")
cells.append(markdown(
    "### Exercise 4 experiment — 2,048 threads in one block\n\n"
    "The next cell recompiles the same source with `N=2048`. Its launch is expected to fail "
    "with `invalid configuration argument` because 2,048 exceeds the device's maximum "
    "threads per block (normally 1,024). The `cudaGetLastError` check reports it. "
    "The source retains its default `N=512`; the final cell recompiles that version.\n"
))
cells.append(code(
    "subprocess.run([NVCC, f'-arch={ARCH}', '-DN=2048', 'ex4_threads.cu', '-o', 'ex4_limit'], check=True)\n"
    "limit_run = subprocess.run(['./ex4_limit'])\n"
    "assert limit_run.returncode != 0, 'Expected the 2,048-thread launch to fail'\n"
    "print('Expected launch-configuration failure observed.')\n"
    "subprocess.run([NVCC, f'-arch={ARCH}', 'ex4_threads.cu', '-o', 'ex4_threads'], check=True)\n"
))
cells += program("ex5_vecmul.cu", "Exercise 5 — Multiply 10,000,000 vector elements",
                 "Expected: `Launching 19532 blocks x 512 threads`, 1,000 final results, "
                 "and `Verification: 0 mismatches out of 10000000`.")
cells.append(markdown(
    "**Think answers:** 512 is a multiple of the warp width, 32, so all lanes in "
    "each full block can do useful work. With 500 threads per block, ceiling division "
    "gives 20,000 blocks and exactly 10,000,000 launched threads; each block's last "
    "warp has only 20 active lanes. The bounds check is still needed for the general case.\n"
))
cells += program("ex6_matrix.cu", "Exercise 6 — Multiply 10,000 × 10,000 matrix elements",
                 "This is element-wise multiplication. It uses three flat arrays of 400 MB "
                 "on the host and three on the device. Expected: `Grid: 313 x 625 blocks`, "
                 "the final 1,000 elements of row 9999, and "
                 "`Verification: 0 mismatches out of 100000000`.")
cells.append(markdown(
    "## Finish\n\n"
    "Check the verification lines and GPU output, save the notebook, and submit it as "
    "your lecturer instructs. Release the GPU using **Runtime → Disconnect and delete runtime**.\n"
))

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
