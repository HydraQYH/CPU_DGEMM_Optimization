#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "DGEMM.h"
#include "utils.h"
#define N 1024

int main()
{
    // Allocate Buffer for A, B, C
    double* p_A = (double*)malloc(N * N * sizeof(double));
    double* p_B = (double*)malloc(N * N * sizeof(double));
    double* p_C = (double*)malloc(N * N * sizeof(double));
    // Initialize & Print matrix values
    _init_matrix_value(p_A, N);
    _init_matrix_value(p_B, N);
    _init_matrix_value(p_C, N);
#ifdef DEBUG_INFO
    printf("Matrix A:\n");
    _print_matrix(p_A, N);
    printf("Matrix B:\n");
    _print_matrix(p_B, N);
    printf("Matrix C:\n");
    _print_matrix(p_C, N);
#endif
    clock_t start, end;
    start = clock();
    dgemm(N, p_A, p_B, p_C);
    end = clock();
    printf("time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(p_A);
    free(p_B);
    free(p_C);
    return 0;
}
