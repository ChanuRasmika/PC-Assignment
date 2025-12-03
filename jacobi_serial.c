#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>   // For omp_get_wtime()

#define MAX_ITER 10000
#define TOLERANCE 1e-5

void jacobi_serial_optimized(double *Adata, double **A, double *b, double *x_out, int N, int *iterations) {
    double *x_old = (double*)calloc(N, sizeof(double));
    double *x_new = (double*)malloc(N * sizeof(double));
    if (!x_old || !x_new) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    double *diag = (double*)malloc(N * sizeof(double));
    if (!diag) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Cache diagonal elements
    for (int i = 0; i < N; i++) {
        diag[i] = A[i][i];
        if (fabs(diag[i]) < 1e-20) {
            fprintf(stderr, "Zero diagonal element at row %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    int iter = 0;
    double error;

    // Use wall-clock time
    double start = omp_get_wtime();

    do {
        // Jacobi update
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++)
                sum += A[i][j] * x_old[j];
            sum -= diag[i] * x_old[i]; // branch-free
            x_new[i] = (b[i] - sum) / diag[i];
        }

        // Infinity norm
        error = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = fabs(x_new[i] - x_old[i]);
            if (diff > error) error = diff;
        }

        // Swap buffers
        double *tmp = x_old;
        x_old = x_new;
        x_new = tmp;

        iter++;
    } while (error > TOLERANCE && iter < MAX_ITER);

    double end = omp_get_wtime();
    *iterations = iter;

    // Copy result to output
    for (int i = 0; i < N; i++) x_out[i] = x_old[i];

    // Print summary
    printf("Converged in %d iterations\n", iter);
    printf("Final error (inf-norm): %.12e\n", error);
    printf("Execution time: %.6f seconds\n", end - start);

    // Print first few entries only
    int print_limit = (N > 10) ? 10 : N;
    printf("\nSolution (first %d entries):\n", print_limit);
    for (int i = 0; i < print_limit; i++)
        printf("x[%d] = %.12f\n", i, x_out[i]);
    if (N > print_limit) printf("... (total %d values)\n", N);

    free(x_old);
    free(x_new);
    free(diag);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s input_file.txt\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        printf("Error: Could not open %s\n", argv[1]);
        return 1;
    }

    int N;
    if (fscanf(fp, "%d", &N) != 1) {
        fprintf(stderr, "Failed to read matrix size\n");
        return 1;
    }

    // Contiguous allocation
    double *Adata = (double*)malloc(N * N * sizeof(double));
    double **A = (double**)malloc(N * sizeof(double*));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    if (!Adata || !A || !b || !x) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++)
        A[i] = &Adata[i * N];

    // Read matrix and RHS
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            fscanf(fp, "%lf", &A[i][j]);
        fscanf(fp, "%lf", &b[i]);
    }
    fclose(fp);

    int iterations;

    printf("Running Optimized Serial Jacobi Solver on %d×%d system\n", N, N);

    // Optional: Repeat computation for small matrices to get measurable time
    int repeat = (N < 50) ? 1000 : 1; // repeat 1000 times if very small
    double start_total = omp_get_wtime();
    for (int r = 0; r < repeat; r++)
        jacobi_serial_optimized(Adata, A, b, x, N, &iterations);
    double end_total = omp_get_wtime();

    if (repeat > 1)
        printf("Average execution time over %d runs: %.9f seconds\n", repeat, (end_total - start_total)/repeat);

    free(Adata);
    free(A);
    free(b);
    free(x);

    return 0;
}
