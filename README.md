# Jacobi Method Implementation - Serial vs OpenMP Parallel

A performance comparison project implementing the Jacobi iterative method for solving systems of linear equations, featuring both serial and OpenMP parallel implementations.

## Project Structure

```
PC-Assignment/
├── DataSet/
│   └── generate_data.c          # Matrix data generator
├── OpenMP/
│   └── jacobi_openmp.c         # Parallel implementation using OpenMP
├── Serial-Program/
│   └── jacobi_serial.c         # Serial implementation
├── Scripts/
│   └── run_bench.py            # Benchmarking script
└── Screenshots/                # Performance analysis results
```

## Features

- **Serial Implementation**: Basic Jacobi method implementation
- **Parallel Implementation**: OpenMP-accelerated version for multi-core performance
- **Data Generator**: Creates diagonally dominant matrices for testing
- **Benchmarking Suite**: Automated performance testing across different thread counts
- **Performance Analysis**: Execution time measurement and speedup calculations

## Requirements

### Compiler
- GCC with OpenMP support, or
- Clang with OpenMP support
- Windows: MinGW-w64 recommended

### Dependencies
- Python 3.x (for benchmarking script)
- OpenMP library

## Quick Start

### 1. Generate Test Data
```bash
cd DataSet
gcc generate_data.c -o generate_data.exe
generate_data.exe matrix_data.txt 1000
```

### 2. Compile Programs
```bash
# Serial version
cd Serial-Program
gcc jacobi_serial.c -o jacobi_serial.exe

# OpenMP version
cd OpenMP
gcc -O3 -fopenmp jacobi_openmp.c -o jacobi_openmp.exe
```

### 3. Run Benchmarks
```bash
cd Scripts
python run_bench.py
```

## Algorithm Details

The Jacobi method solves linear systems Ax = b iteratively:
- **Convergence**: Tolerance of 1e-5
- **Max Iterations**: 1000
- **Matrix Requirements**: Diagonally dominant for guaranteed convergence

## Performance Configuration

The benchmarking script tests with thread counts: 1, 2, 4, 8, 16

Modify `THREADS` array in `run_bench.py` to customize test configurations.

## Results

Performance results are automatically calculated showing:
- Execution time per thread count
- Speedup ratios compared to serial execution
- Best performing configuration

See `Screenshots/` folder for sample performance analysis.

## Usage Examples

### Manual Execution
```bash
# Set thread count for OpenMP version
set OMP_NUM_THREADS=4
jacobi_openmp.exe
```

### Custom Matrix Size
```bash
generate_data.exe custom_matrix.txt 2000
# Update source code to read from custom_matrix.txt
```

## Technical Notes

- Uses Windows high-resolution timing (QueryPerformanceCounter)
- Memory allocation optimized for large matrices
- Displays partial results for matrices > 5x5
- Automatic convergence detection

## License

Academic/Educational use