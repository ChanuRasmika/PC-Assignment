#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_ITER 100000
#define TOLERANCE 1e-8

void jacobi_optimized(double *Adata, double **A, double *b, double *x_out, int N, int *iterations) {
    double *x_old = (double*)calloc(N, sizeof(double)); // start with zeros
    double *x_new = (double*)malloc(N * sizeof(double));
    double *diag  = (double*)malloc(N * sizeof(double));
    if (!x_old || !x_new || !diag) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Cache diagonal and check it's non-zero
    for (int i = 0; i < N; ++i) {
        diag[i] = A[i][i];
        if (fabs(diag[i]) < 1e-20) {
            fprintf(stderr, "Zero (or nearly zero) diagonal element at row %d — cannot proceed.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    int iter = 0;
    double error;

    double start = omp_get_wtime();

    do {
        // Jacobi update: compute x_new from x_old
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double *Ai = A[i];          // row pointer
            double sum = 0.0;

            // compute full dot(Ai, x_old)
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                sum += Ai[j] * x_old[j];
            }
            // remove diagonal contribution to get sum_{j != i} A[i][j] * x_old[j]
            sum -= diag[i] * x_old[i];

            // Jacobi formula
            x_new[i] = (b[i] - sum) / diag[i];
        }

        // Infinity norm (max abs diff) reduction
        error = 0.0;
        #pragma omp parallel for reduction(max:error)
        for (int i = 0; i < N; ++i) {
            double diff = fabs(x_new[i] - x_old[i]);
            if (diff > error) error = diff; // reduction(max:error) will combine correctly
        }

        // swap buffers (cheap pointer swap)
        double *tmp = x_old; x_old = x_new; x_new = tmp;

        iter++;
    } while (error > TOLERANCE && iter < MAX_ITER);

    double end = omp_get_wtime();

    // copy final result to user buffer
    for (int i = 0; i < N; ++i) x_out[i] = x_old[i];

    *iterations = iter;

    // print summary (optional)
    #pragma omp critical
    {
        printf("Jacobi finished in %d iterations\n", iter);
        printf("Final error (inf-norm): %.12e\n", error);
        printf("Elapsed time: %.6f s\n", end - start);
        printf("OpenMP threads used: %d\n", omp_get_max_threads());
    }

    free(x_old);
    free(x_new);
    free(diag);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s input.txt [num_threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 3) {
        int t = atoi(argv[2]);
        if (t > 0) omp_set_num_threads(t);
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        perror("fopen");
        return EXIT_FAILURE;
    }

    int N;
    if (fscanf(fp, "%d", &N) != 1) {
        fprintf(stderr, "Failed to read matrix size N\n");
        return EXIT_FAILURE;
    }

    // allocate contiguous matrix block
    double *Adata = (double*)malloc((size_t)N * N * sizeof(double));
    double **A = (double**)malloc(N * sizeof(double*));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    if (!Adata || !A || !b || !x) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) A[i] = &Adata[(size_t)i * N];

    // read matrix rows and b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fscanf(fp, "%lf", &A[i][j]) != 1) {
                fprintf(stderr, "Failed to read A[%d][%d]\n", i, j);
                return EXIT_FAILURE;
            }
        }
        if (fscanf(fp, "%lf", &b[i]) != 1) {
            fprintf(stderr, "Failed to read b[%d]\n", i);
            return EXIT_FAILURE;
        }
    }
    fclose(fp);

    // initialize output vector
    for (int i = 0; i < N; ++i) x[i] = 0.0;

    int iterations = 0;
    printf("Solving %dx%d system using optimized parallel Jacobi\n", N, N);
    printf("Input file: %s\n", argv[1]);
    printf("=================================================\n");

    jacobi_optimized(Adata, A, b, x, N, &iterations);

    printf("\nSolution (first 10 entries shown if large):\n");
    int show = (N > 10) ? 10 : N;
    for (int i = 0; i < show; ++i) {
        printf("x[%d] = %.12f\n", i, x[i]);
    }
    if (N > show) printf("... (total %d values)\n", N);

    free(Adata);
    free(A);
    free(b);
    free(x);

    return EXIT_SUCCESS;
}
