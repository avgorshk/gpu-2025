#include <vector>
#include <omp.h>
#include <algorithm>

void ProcessBlock(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &res,
                  int n, int row_start, int col_start, int k_start, int block_size)
{
    for (int row = row_start; row < std::min(row_start + block_size, n); ++row)
    {
        for (int k = k_start; k < std::min(k_start + block_size, n); ++k)
        {
            float temp = a[row * n + k];
            int col_end = std::min(col_start + block_size, n);
#pragma omp simd
            for (int col = col_start; col < col_end; ++col)
            {
                res[row * n + col] += temp * b[k * n + col];
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::vector<float> res(n * n, 0.0f);
    int block_size = 32;

#pragma omp parallel for
    for (int row_block = 0; row_block < n; row_block += block_size)
    {
        for (int col_block = 0; col_block < n; col_block += block_size)
        {
            for (int k_block = 0; k_block < n; k_block += block_size)
            {
                ProcessBlock(a, b, res, n, row_block, col_block, k_block, block_size);
            }
        }
    }

    return res;
}