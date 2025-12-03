import subprocess
import os
import time

# Path to your compiled OpenMP Jacobi executable
exe_path = "jacobi_omp.exe"  # change to your exe name
input_file = "dataset_500.txt"    # your input matrix file

# Thread counts to test
thread_counts = [1, 2, 4, 8, 16]

# Dictionary to store execution times
times = {}

# Check if executable exists
if not os.path.exists(exe_path):
    print(f"Error: Executable '{exe_path}' not found!")
    exit(1)

if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found!")
    exit(1)

print(f"Testing OpenMP program: {exe_path}")
print(f"Input file: {input_file}")
print(f"Thread counts: {thread_counts}")
print("=" * 50)

for threads in thread_counts:
    print(f"\nRunning with {threads} thread(s)...")

    # On Windows, set OMP_NUM_THREADS before running
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    try:
        # Run the executable and capture output
        start_time = time.time()
        result = subprocess.run([exe_path, input_file], capture_output=True, text=True, env=env, timeout=60)
        wall_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Error: Program exited with code {result.returncode}")
            print(f"stderr: {result.stderr}")
            continue

        # Parse execution time from output
        # Assuming your program prints: "Execution time: X.XXXXXX seconds"
        time_sec = None
        for line in result.stdout.splitlines():
            if "Execution time:" in line or "Time:" in line:
                # Try to extract number after colon
                parts = line.split(":")
                if len(parts) > 1:
                    try:
                        time_sec = float(parts[1].split()[0])
                        break
                    except (ValueError, IndexError):
                        continue

        # Fallback to wall clock time if parsing fails
        if time_sec is None:
            print("Could not parse execution time from output. Using wall clock time.")
            print("Program output:")
            print(result.stdout)
            time_sec = wall_time

        times[threads] = time_sec
        print(f"Execution time: {time_sec:.6f} s")
        
    except subprocess.TimeoutExpired:
        print(f"Timeout: Program took longer than 60 seconds with {threads} threads")
        continue
    except Exception as e:
        print(f"Error running program: {e}")
        continue

# Compute speedups and efficiency
print("\n" + "=" * 50)
print("PERFORMANCE RESULTS")
print("=" * 50)

serial_time = times.get(1, None)
if serial_time is None:
    print("No serial (1-thread) time found. Cannot compute speedups.")
else:
    print(f"{'Threads':<8} {'Time(s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 45)
    
    for t in thread_counts:
        if t in times:
            time_val = times[t]
            speedup = serial_time / time_val
            efficiency = speedup / t * 100  # Efficiency as percentage
            print(f"{t:<8} {time_val:<12.6f} {speedup:<10.2f} {efficiency:<12.1f}%")
        else:
            print(f"{t:<8} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    # Find best speedup
    if len(times) > 1:
        best_threads = max(times.keys(), key=lambda x: serial_time / times[x] if x in times else 0)
        best_speedup = serial_time / times[best_threads]
        print(f"\nBest performance: {best_threads} threads with {best_speedup:.2f}x speedup")

print("\nDone!")
