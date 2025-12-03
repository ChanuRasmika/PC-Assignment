# Jacobi Method Implementation - Multi-Platform Performance Comparison

A comprehensive performance comparison project implementing the Jacobi iterative method for solving systems of linear equations across multiple parallel computing platforms: Serial, OpenMP, MPI, and CUDA.

## Project Structure

```
PC-Assignment/
├── DataSet/
│   └── generate_data.c                    # Matrix data generator
├── Serial-Program/
│   └── jacobi_serial.c                   # Serial implementation
├── OpenMP/
│   └── jacobi_openmp.c                   # OpenMP parallel implementation
├── MPI/
│   └── jacobi_mpi.c                      # MPI distributed implementation
├── CUDA/
│   └── Jacobi_Impl_CUDA.ipynb           # CUDA GPU implementation (Jupyter)
├── Scripts/
│   ├── run_bench_omp.py                 # OpenMP benchmarking script
│   └── run_bench_mpi.py                 # MPI benchmarking script
├── ColabNotebook for generate Graph/
│   └── Grapghs_MPI_OMP.ipynb           # Graph generation notebook
└── Screenshots/                         # Performance analysis results
    ├── OMP/                            # OpenMP performance graphs
    ├── MPI/                            # MPI performance graphs
    ├── CUDA/                           # CUDA performance graphs
    └── System-Information.png          # System specifications
```

## Features

- **Serial Implementation**: Baseline single-threaded Jacobi method
- **OpenMP Implementation**: Shared-memory parallel version for multi-core systems
- **MPI Implementation**: Distributed-memory parallel version for clusters
- **CUDA Implementation**: GPU-accelerated version for NVIDIA graphics cards
- **Data Generator**: Creates diagonally dominant matrices ensuring convergence
- **Comprehensive Benchmarking**: Automated performance testing across platforms
- **Performance Visualization**: Jupyter notebooks for generating performance graphs

## Requirements

### Compilers & Runtime
- **C Compiler**: GCC or Clang with OpenMP support
- **MPI**: OpenMPI or MPICH for distributed computing
- **CUDA**: NVIDIA CUDA Toolkit (for GPU implementation)
- **Python**: 3.x with matplotlib, numpy (for benchmarking and visualization)

### Platform-Specific
- **Windows**: MinGW-w64 or Visual Studio with CUDA support
- **Linux**: Standard development tools, CUDA drivers
- **GPU**: NVIDIA GPU with compute capability 3.0+ (for CUDA version)

## Quick Start

### 1. Generate Test Data
```bash
cd DataSet
gcc generate_data.c -o generate_data.exe
generate_data.exe matrix_data.txt 1000
```

### 2. Compile All Implementations
```bash
# Serial version
cd Serial-Program
gcc jacobi_serial.c -o jacobi_serial.exe

# OpenMP version
cd OpenMP
gcc -O3 -fopenmp jacobi_openmp.c -o jacobi_openmp.exe

# MPI version
cd MPI
mpicc -O3 jacobi_mpi.c -o jacobi_mpi.exe

# CUDA version (use Jupyter notebook or compile directly)
nvcc -O3 -arch=sm_75 jacobi_cuda.cu -o jacobi_cuda
```

### 3. Run Benchmarks
```bash
# OpenMP benchmarking
cd Scripts
python run_bench_omp.py

# MPI benchmarking
python run_bench_mpi.py

# CUDA benchmarking (via Jupyter notebook)
jupyter notebook ../CUDA/Jacobi_Impl_CUDA.ipynb
```

## Algorithm Details

The Jacobi method solves linear systems Ax = b iteratively:
- **Convergence**: Tolerance of 1e-5
- **Max Iterations**: 1000
- **Matrix Requirements**: Diagonally dominant for guaranteed convergence

## Performance Configuration

### OpenMP Configuration
- **Thread counts**: 1, 2, 4, 8, 16
- Modify `THREADS` array in `run_bench_omp.py`

### MPI Configuration
- **Process counts**: 1, 2, 4, 8, 16
- Modify `proc_counts` array in `run_bench_mpi.py`

### CUDA Configuration
- **Block sizes**: 64, 128, 256, 512
- Configurable via command line: `--block <size>`

## Results

Performance metrics include:
- **Execution time** per configuration (threads/processes/block size)
- **Speedup ratios** compared to serial baseline
- **Efficiency analysis** across different platforms
- **Scalability comparison** between OpenMP, MPI, and CUDA

Results are saved as:
- CSV files for numerical data
- PNG graphs in `Screenshots/` organized by platform
- Jupyter notebooks with interactive visualizations

## Usage Examples

### Manual Execution
```bash
# OpenMP with specific thread count
set OMP_NUM_THREADS=4
jacobi_openmp.exe

# MPI with multiple processes
mpiexec -n 4 jacobi_mpi.exe

# CUDA with custom block size
jacobi_cuda --block 256
```

### Custom Matrix Sizes
```bash
# Generate different matrix sizes
generate_data.exe small_matrix.txt 500
generate_data.exe large_matrix.txt 2000

# All implementations read from matrix_data.txt by default
```

## Technical Implementation

### Algorithm Details
- **Convergence tolerance**: 1e-5
- **Maximum iterations**: 1000
- **Matrix type**: Diagonally dominant (ensures convergence)
- **Timing**: High-resolution performance counters

### Platform-Specific Features
- **OpenMP**: Parallel for loops with shared memory
- **MPI**: Domain decomposition with message passing
- **CUDA**: GPU kernels with configurable block sizes
- **Memory optimization**: Efficient allocation for large matrices

### Output Format
- Displays first 5x5 submatrix and solution elements
- Reports iteration count and execution time
- Automatic convergence detection

## Performance Analysis

The project includes comprehensive performance analysis tools:
- Automated benchmarking scripts for each platform
- Jupyter notebooks for visualization and comparison
- Speedup and efficiency calculations
- Cross-platform performance comparison

## License

Academic/Educational use