import subprocess
import time
import sys
from typing import List, Tuple

def run_benchmark(executable: str, proc_counts: List[int]) -> List[Tuple[int, float]]:
    """
    Run the MPI executable with different process counts and measure wall-clock time.
    Returns list of (processes, elapsed_ms).
    """
    results: List[Tuple[int, float]] = []
    for n in proc_counts:
        # Windows PowerShell expects ".\\" for local executable
        cmd = ["mpiexec", "-n", str(n), executable]
        start = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed for n={n}: {e}")
            results.append((n, float("nan")))
            continue
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        results.append((n, elapsed_ms))
    return results

def display_results(results: List[Tuple[int, float]], serial_ms: float = None):
    """Display benchmark results in a nicely formatted table."""
    print("\n" + "="*45)
    print("      MPI JACOBI BENCHMARK RESULTS")
    print("="*45)
    
    # Table header
    print(f"{'Processes':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 45)
    
    # Use first result as baseline if no serial time provided
    if serial_ms is None and results:
        serial_ms = results[0][1]
    
    for n, t in results:
        if t != t:  # NaN check
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<10}")
        else:
            speedup = serial_ms / t if t > 0 and serial_ms else 0.0
            print(f"{n:<10} {t:<12.3f} {speedup:<10.2f}")
    
    print("-" * 45)
    
    # Summary statistics
    valid_results = [(n, t) for n, t in results if t == t]  # Filter out NaN
    if len(valid_results) > 1:
        best_speedup = max(serial_ms / t for _, t in valid_results if t > 0)
        best_proc = next(n for n, t in valid_results if serial_ms / t == best_speedup)
        print(f"Best speedup: {best_speedup:.2f}x with {best_proc} processes")
    
    print("="*45 + "\n")

def main():
    # Detect platform path to the MPI executable
    if sys.platform.startswith("win"):
        exe = ".\\jacobi_mpi.exe"
    else:
        exe = "./jacobi_mpi.exe"

    proc_counts = [1, 2, 4, 8, 16]
    print(f"Running MPI benchmark with executable: {exe}")
    print(f"Process counts: {proc_counts}")
    print("\nStarting benchmark...")
    
    results = run_benchmark(exe, proc_counts)
    
    # Display results nicely
    display_results(results)
    
    # Also save CSV for further analysis
    with open("benchmark_results.csv", "w") as f:
        f.write("Processes,Time_ms,Speedup\n")
        serial_ms = results[0][1] if results else 100.0
        for n, t in results:
            if t == t:  # Not NaN
                speedup = serial_ms / t if t > 0 else 0.0
                f.write(f"{n},{t:.3f},{speedup:.3f}\n")
            else:
                f.write(f"{n},ERROR,ERROR\n")
    
    print("Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
