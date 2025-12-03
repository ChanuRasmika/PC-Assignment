#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_ITER 1000    // Maximum iterations
#define TOLERANCE 1e-5   // Convergence tolerance (L1 norm of delta x)

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    double *A_full = NULL; // flattened NxN matrix (row-major) on rank 0
    double *b_full = NULL; // full b on rank 0

    // Rank 0 reads the matrix and vector from file and broadcasts
    if (rank == 0) {
        FILE *file = fopen("matrix_data.txt", "r");
        if (!file) die("Error opening file 'matrix_data.txt'");
        if (fscanf(file, "%d", &N) != 1) die("Failed to read N");

        A_full = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
        b_full = (double *)malloc((size_t)N * sizeof(double));
        if (!A_full || !b_full) die("Allocation failed on rank 0");

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (fscanf(file, "%lf", &A_full[(size_t)i * N + j]) != 1) die("Failed to read A");
            }
        }
        for (int i = 0; i < N; i++) {
            if (fscanf(file, "%lf", &b_full[i]) != 1) die("Failed to read b");
        }
        fclose(file);
    }

    // Broadcast N to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid N: %d\n", N);
        MPI_Finalize();
        return 1;
    }

    // Compute row distribution
    int base = N / size;
    int rem = N % size;
    int local_rows = base + (rank < rem ? 1 : 0);
    int start_row = rank * base + (rank < rem ? rank : rem);
    int end_row = start_row + local_rows; // exclusive

    // Allocate local slices
    double *A_local = (double *)malloc((size_t)local_rows * (size_t)N * sizeof(double));
    double *b_local = (double *)malloc((size_t)local_rows * sizeof(double));
    if (!A_local || !b_local) die("Local allocation failed");

    // Create counts and displacements for scattering rows
    int *sendcounts = NULL, *displsA = NULL, *sendcounts_b = NULL, *displsb = NULL;
    if (rank == 0) {
        sendcounts   = (int *)malloc(size * sizeof(int));
        displsA      = (int *)malloc(size * sizeof(int));
        sendcounts_b = (int *)malloc(size * sizeof(int));
        displsb      = (int *)malloc(size * sizeof(int));
        if (!sendcounts || !displsA || !sendcounts_b || !displsb) die("Scatter arrays alloc failed");

        int offset_rows = 0;
        int offset_elemsA = 0;
        for (int r = 0; r < size; r++) {
            int rows_r = base + (r < rem ? 1 : 0);
            sendcounts[r]   = rows_r * N; // number of doubles for A
            displsA[r]      = offset_elemsA;
            sendcounts_b[r] = rows_r;     // number of doubles for b
            displsb[r]      = offset_rows;
            offset_rows += rows_r;
            offset_elemsA += rows_r * N;
        }
    }

    // Scatter A rows and b entries
    MPI_Scatterv(A_full, sendcounts, displsA, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b_full, sendcounts_b, displsb, MPI_DOUBLE,
                 b_local, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate global x and buffers
    double *x = (double *)malloc((size_t)N * sizeof(double));
    double *x_old = (double *)malloc((size_t)N * sizeof(double));
    double *x_local = (double *)malloc((size_t)local_rows * sizeof(double));
    if (!x || !x_old || !x_local) die("x allocation failed");

    // Initialize x to zero
    for (int i = 0; i < N; i++) x[i] = 0.0;
    for (int i = 0; i < N; i++) x_old[i] = 0.0;

    int iterations = 0;
    double global_error = 0.0;

    // Jacobi iterations
    do {
        // Copy current solution to x_old
        for (int i = 0; i < N; i++) x_old[i] = x[i];

        // Compute local new x for assigned rows using x_old (global)
        for (int li = 0; li < local_rows; li++) {
            int i = start_row + li;
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i) sum += A_local[(size_t)li * N + j] * x_old[j];
            }
            x_local[li] = (b_local[li] - sum) / A_local[(size_t)li * N + i];
        }

        // Gather all local x segments into global x
        // Prepare counts and displacements for gatherv
        int *rcounts = NULL, *rdispls = NULL;
        if (rank == 0) {
            rcounts = (int *)malloc(size * sizeof(int));
            rdispls = (int *)malloc(size * sizeof(int));
            if (!rcounts || !rdispls) die("Gather arrays alloc failed");

            int offset = 0;
            for (int r = 0; r < size; r++) {
                int rows_r = base + (r < rem ? 1 : 0);
                rcounts[r] = rows_r;
                rdispls[r] = offset;
                offset += rows_r;
            }
        }

        MPI_Gatherv(x_local, local_rows, MPI_DOUBLE,
                    x, rcounts, rdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Broadcast updated x to all ranks for next iteration
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Compute local error and reduce to global
        double local_error = 0.0;
        for (int li = 0; li < local_rows; li++) {
            int i = start_row + li;
            local_error += fabs(x[i] - x_old[i]);
        }
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        iterations++;

        if (rank == 0) {
            free(rcounts);
            free(rdispls);
        }
    } while (global_error > TOLERANCE && iterations < MAX_ITER);

    if (rank == 0) {
        int print_size = (N < 5) ? N : 5;
        printf("Solving system using MPI Jacobi Method (size=%d, procs=%d)\n", N, size);
        printf("===========================================================\n\n");
        printf("Solution (first %d elements):\n", print_size);
        for (int i = 0; i < print_size; i++) {
            printf("x%d = %.5f\n", i, x[i]);
        }
        printf("\nIterations: %d\n", iterations);
        printf("Final L1 error: %.6e\n", global_error);
    }

    // Cleanup
    free(A_local);
    free(b_local);
    free(x_local);
    free(x);
    free(x_old);
    if (rank == 0) {
        free(A_full);
        free(b_full);
        free(sendcounts);
        free(displsA);
        free(sendcounts_b);
        free(displsb);
    }

    MPI_Finalize();
    return 0;
}