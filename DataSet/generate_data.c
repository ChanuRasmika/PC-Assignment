#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


void generate_data_file(const char *filename, int n) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file '%s' for writing.\n", filename);
        exit(1);
    }

    // Write the size of the matrix
    fprintf(file, "%d\n", n);

    // Generate and write the diagonally dominant matrix A
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        // First, calculate the sum of non-diagonal elements for the row
        for (int j = 0; j < n; j++) {
            if (i != j) {
                row_sum += fabs((double)(rand() % 10));
            }
        }
        // Now, write the row with a dominant diagonal element
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Diagonal element is greater than the sum of the absolute values of other elements
                fprintf(file, "%f ", row_sum + (double)(rand() % 10) + 1.0);
            } else {
                fprintf(file, "%f ", (double)(rand() % 10));
            }
        }
        fprintf(file, "\n");
    }

    // Generate and write vector b
    for (int i = 0; i < n; i++) {
        fprintf(file, "%f ", (double)(rand() % 100));
    }
    fprintf(file, "\n");

    fclose(file);
    printf("Successfully generated data file '%s' with N=%d\n", filename, n);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <output_filename> <matrix_size>\n", argv[0]);
        fprintf(stderr, "Example: .\\generate_data.exe matrix_data.txt 1000\n");
        return 1;
    }

    const char *filename = argv[1];
    int n = atoi(argv[2]);

    if (n <= 0) {
        fprintf(stderr, "Matrix size must be a positive integer.\n");
        return 1;
    }

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    generate_data_file(filename, n);

    return 0;
}
