import os
import subprocess
import time
import shutil
from typing import List, Tuple

# Configuration
SOURCE = "jacobi_openmp.c"
EXE = "jacobi_openmp.exe" if os.name == "nt" else "jacobi_openmp"
THREADS = [1, 2, 4, 8, 16]
# Approximate serial time (ms) provided by user
SERIAL_MS = 100.0

# If your program needs input file(s), set them here
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


def which_compiler() -> Tuple[str, List[str]]:
    """Return (compiler_name, flags) for compiling OpenMP on this system.
    Tries gcc then clang.
    """
    if shutil.which("gcc"):
        return ("gcc", ["-O3", "-fopenmp"])
    if shutil.which("clang"):
        # Windows clang may require -Xpreprocessor -fopenmp and linking to libomp
        # Try the simpler variant first (works on many setups)
        return ("clang", ["-O3", "-fopenmp"])
    return ("", [])


def compile_program() -> None:
    comp, flags = which_compiler()
    if not comp:
        raise RuntimeError(
            "No C compiler found. Install MinGW-w64 (gcc) or LLVM clang."
        )

    cmd = [comp, SOURCE, "-o", EXE] + flags
    print(f"Compiling: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=WORKING_DIR, check=True)


def run_once(threads: int) -> float:
    """Run the program with given threads and return elapsed time in ms."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    start = time.perf_counter()
    proc = subprocess.run([os.path.join(WORKING_DIR, EXE)], cwd=WORKING_DIR, env=env)
    end = time.perf_counter()

    # If the program prints its own timing, we ignore and measure externally
    elapsed_ms = (end - start) * 1000.0
    return elapsed_ms


def format_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


def main():
    # Compile if the executable is missing or older than source
    exe_path = os.path.join(WORKING_DIR, EXE)
    src_path = os.path.join(WORKING_DIR, SOURCE)

    needs_build = (
        not os.path.exists(exe_path)
        or (os.path.getmtime(exe_path) < os.path.getmtime(src_path))
    )

    if needs_build:
        try:
            compile_program()
        except Exception as e:
            print("Compilation failed:", e)
            print(
                "Hint: On Windows, install MinGW-w64 and ensure 'gcc' is in PATH, "
                "or install LLVM + OpenMP support."
            )
            return

    results: List[Tuple[int, float, float]] = []

    print("Running benchmarks...")
    for t in THREADS:
        # Warm-up run (optional): helps stabilize timing
        _ = run_once(t)
        # Timed run
        elapsed_ms = run_once(t)
        speedup = SERIAL_MS / elapsed_ms if elapsed_ms > 0 else float('inf')
        results.append((t, elapsed_ms, speedup))

    # Prepare table
    headers = ["Threads", "Time (ms)", f"Speedup vs {SERIAL_MS:.0f} ms"]
    rows = [
        [str(t), f"{time_ms:.2f}", f"{sp:.2f}x"]
        for (t, time_ms, sp) in results
    ]

    # Compute column widths
    widths = [
        max(len(h), *(len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    sep = "-+-".join("-" * w for w in widths)

    print()
    print(format_row(headers, widths))
    print(sep)
    for r in rows:
        print(format_row(r, widths))

    # Optional: overall best
    best = min(results, key=lambda x: x[1])
    print()
    print(f"Best time: {best[1]:.2f} ms with {best[0]} threads ({SERIAL_MS/best[1]:.2f}x speedup)")


if __name__ == "__main__":
    main()
