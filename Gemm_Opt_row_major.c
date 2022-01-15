#include <x86intrin.h>
#include "DGEMM.h"
#define UNROLL (4)
#define BLOCKSIZE 32

#ifdef OPT_LEVEL_1

void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j += 4)
        {
            // 取C矩阵的(i, j)到(i, j + 3)
            double* cij = C + i * n + j;
            __m256d c0 = _mm256_loadu_pd(cij);

            for (size_t k = 0; k < n; k++)
            {
                // 取B的(k, j)到(k, j + 3)
                double* bkj = B + k * n + j;
                __m256d b0 = _mm256_loadu_pd(bkj);

                // 取A的(i, k)并广播
                double* aik = A + i * n + k;
                __m256d a0 = _mm256_broadcast_sd(aik);

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
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j += 4 * UNROLL)
        {
            // 通过循环展开的方式 取C矩阵的(i, j)到(i, j + 15)
            __m256d c[4];
            for (size_t x = 0; x < UNROLL; x++)
            {
                c[x] = _mm256_loadu_pd(C + i * n + 4 * x + j);
            }
            
            for (size_t k = 0; k < n; k++)
            {
                // 取A的(i, k)并广播
                __m256d a = _mm256_broadcast_sd(A + i * n + k);

                // 循环展开 计算16列的值
                for (size_t y = 0; y < UNROLL; y++)
                {
                    c[y] = _mm256_add_pd(c[y], _mm256_mul_pd(a, _mm256_loadu_pd(B + k * n + y * 4 + j)));
                }
            }

            // Store Back
            for (size_t x = 0; x < UNROLL; x++)
            {
                _mm256_storeu_pd(C + i * n + 4 * x + j, c[x]);
            }
        }
    }
}

#endif


#ifdef OPT_LEVEL_3

void do_block(int n, int si, int sj, int sk, double* A, double* B, double* C)
{
    for (size_t i = si; i < si + BLOCKSIZE; i++)
    {
        for (size_t j = sj; j < sj + BLOCKSIZE; j += 4 * UNROLL)
        {
            // 通过循环展开的方式 取C矩阵的(i, j)到(i, j + 15)
            __m256d c[4];
            for (size_t x = 0; x < UNROLL; x++)
            {
                c[x] = _mm256_loadu_pd(C + i * n + 4 * x + j);
            }
            
            for (size_t k = sk; k < sk + BLOCKSIZE; k++)
            {
                // 取A的(i, k)并广播
                __m256d a = _mm256_broadcast_sd(A + i * n + k);

                // 循环展开 计算16列的值
                for (size_t y = 0; y < UNROLL; y++)
                {
                    c[y] = _mm256_add_pd(c[y], _mm256_mul_pd(a, _mm256_loadu_pd(B + k * n + y * 4 + j)));
                }
            }

            // Store Back
            for (size_t x = 0; x < UNROLL; x++)
            {
                _mm256_storeu_pd(C + i * n + 4 * x + j, c[x]);
            }
        }
    }
}

void dgemm(size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i += BLOCKSIZE)
    {
        for (size_t j = 0; j < n; j += BLOCKSIZE)
        {
            for (size_t k = 0; k < n; k += BLOCKSIZE)
            {
                // 计算矩阵分块A(i, k, i + block_size, k + block_size)
                // B(k, j, k + block_size, j + block_size)
                // C(i, j, i + block_size, j + block_size)
                do_block(n, i, j, k, A, B, C);
            }
        }
    }
}

#endif
