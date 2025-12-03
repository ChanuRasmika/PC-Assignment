#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

#define MAX_ITER 1000    // Maximum iterations
#define TOLERANCE 1e-5   // Convergence tolerance

void jacobi(int N, double **A, double *b, double *x, int *iterations) {
    double *x_old = (double *)malloc(N * sizeof(double));
    double error;
    int iter = 0;

    // Initialize solution vector to zero
    for(int i = 0; i < N; i++) {
        x[i] = 0.0;
    }

    do {
        // Copy current solution to x_old
        for(int i = 0; i < N; i++) {
            x_old[i] = x[i];
        }

        // Compute new solution
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            double sum = 0.0;
            for(int j = 0; j < N; j++) {
                if(j != i) {
                    sum += A[i][j] * x_old[j];
                }
            }
            x[i] = (b[i] - sum) / A[i][i];
        }

        // Calculate error
        error = 0.0;
        for(int i = 0; i < N; i++) {
            error += fabs(x[i] - x_old[i]);
        }

        iter++;

    } while(error > TOLERANCE && iter < MAX_ITER);

    *iterations = iter;
    free(x_old);
}

int main() {
    int N;
    double **A;
    double *b;
    double *x;
    int iterations;
    FILE *file;

    // Open the data file
    file = fopen("matrix_data.txt", "r");
    if (file == NULL) {
        printf("Error opening file 'matrix_data.txt'\n");
        return 1;
    }

    // Read the size of the matrix
    fscanf(file, "%d", &N);

    // Allocate memory for matrices and vectors
    A = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    b = (double *)malloc(N * sizeof(double));
    x = (double *)malloc(N * sizeof(double));

    // Read matrix A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file, "%lf", &A[i][j]);
        }
    }

    // Read vector b
    for (int i = 0; i < N; i++) {
        fscanf(file, "%lf", &b[i]);
    }

    fclose(file);

    printf("Solving system of linear equations using Jacobi Method\n");
    printf("======================================================\n\n");

    // Print the system
    printf("System of equations (first 5x5 part for large matrices):\n");
    int print_size = (N < 5) ? N : 5;
    for(int i = 0; i < print_size; i++) {
        for(int j = 0; j < print_size; j++) {
            printf("%8.2f*x%d ", A[i][j], j);
            if(j < print_size-1) printf("+ ");
        }
        printf("... = %8.2f\n", b[i]);
    }
    printf("\n");

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double time_spent;

    QueryPerformanceFrequency(&frequency);

    // Start timing
    QueryPerformanceCounter(&start);

    // Solve using Jacobi method
    jacobi(N, A, b, x, &iterations);

    // End timing
    QueryPerformanceCounter(&end);
    time_spent = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    
    printf("Solution (first 5 elements for large matrices):\n");
    for(int i = 0; i < print_size; i++) {
        printf("x%d = %.5f\n", i, x[i]);
    }
    printf("\nIterations: %d\n", iterations);
    printf("Execution time: %.6f milliseconds\n", time_spent);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x);

    return 0;
}