#include <x86intrin.h>
#include "DGEMM.h"
#define UNROLL (4)
#define BLOCKSIZE 32

// Matrix is column major

#ifdef OPT_LEVEL_0
void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double cij = C[i + j * n];
            for (size_t k = 0; k < n; k++)
            {
                cij += A[i + k * n] + B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}
#endif

#ifdef OPT_LEVEL_1
void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i += 4)
    {
        for (size_t j = 0; j < n; j++)
        {
            // 取C矩阵的(i, j)到(i + 3, j)
            double* cij = C + i + j * n;
            __m256d c0 = _mm256_loadu_pd(cij);
            for (size_t k = 0; k < n; k++)
            {
                // 取B的(k, j) 并广播
                double* bkj = B + k + j * n;
                __m256d b0 = _mm256_broadcast_sd(bkj);
                // 取A的(i, k)到(i + 3, k)
                double* aik = A + i + k * n;
                __m256d a0 = _mm256_loadu_pd(aik);
                // 计算cij += aik * bkj
                c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
            }
            // 将计算好的c0存储到对应的位置
            _mm256_storeu_pd(cij, c0);
        }
    }
}
#endif

#ifdef OPT_LEVEL_2

void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i += 4 * UNROLL)
    {
        for (size_t j = 0; j < n; j++)
        {
            // 通过循环展开 取C的16行
            __m256d c[4];
            for (size_t x = 0; x < UNROLL; x++)
            {
                c[x] = _mm256_loadu_pd(C + i + 4 * x + j * n);
            }

            // 按照OPT_LEVEL_1的方法与矩阵B的一列做内积 但是这次要处理16行矩阵A 而不是4行
            for (size_t k = 0; k < n; k++)
            {
                // Load矩阵B的一个元素 并将其广播
                __m256d b0 = _mm256_broadcast_sd(B + k + j * n);
                for (size_t y = 0; y < UNROLL; y++)
                {
                    c[y] = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(A + i + 4 * y + k * n), b0), c[y]);
                }
            }
            
            // Store Back
            for (size_t x = 0; x < UNROLL; x++)
            {
                _mm256_storeu_pd(C + i + 4 * x + j * n, c[x]);
            }
        }
    }
}

#endif


#ifdef OPT_LEVEL_3

void do_block(int n, int si, int sj, int sk, double* A, double* B, double* C)
{
    for (size_t i = si; i < si + BLOCKSIZE; i += 4 * UNROLL)
    {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++)
        {
            __m256d c[4];
            for (size_t x = 0; x < UNROLL; x++)
            {
                c[x] = _mm256_loadu_pd(C + i + 4 * x + j * n);
            }

            for (size_t k = sk; k < sk + BLOCKSIZE; k++)
            {
                __m256d b0 = _mm256_broadcast_sd(B + k + j * n);
                for (size_t y = 0; y < UNROLL; y++)
                {
                    c[y] = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(A + i + 4 * y + k * n), b0), c[y]);
                }
            }
            
            // Store Back
            for (size_t x = 0; x < UNROLL; x++)
            {
                _mm256_storeu_pd(C + i + 4 * x + j * n, c[x]);
            }
        }
    }
}

void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i+=BLOCKSIZE)
    {
        for (size_t j = 0; j < n; j+=BLOCKSIZE)
        {
            for (size_t k = 0; k < n; k+=BLOCKSIZE)
            {
                // 将矩阵分块 每块作为一个新的矩阵“元素” 再做矩阵乘法
                do_block(n, i, j, k, A, B, C);
            }
        }
    }
}

#endif
