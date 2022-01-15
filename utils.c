#include "utils.h"

void _init_matrix_value(double* p, size_t n)
{
    for (size_t i = 0; i < n * n; i++)
    {
        double tmp = (double)rand();
        double max_value = (double)(RAND_MAX);
        double value = tmp / max_value;
        p[i] = value;
    }
}

void _print_matrix(double* p, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        printf("[ ");
        for (size_t j = 0; j < n; j++)
        {
            printf("%.3lf\t", p[i + j * n]);
        }
        printf("]\n");
    }
}
