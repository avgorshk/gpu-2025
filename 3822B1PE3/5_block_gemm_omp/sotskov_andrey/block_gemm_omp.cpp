#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &matrixA,
                                const std::vector<float> &matrixB, int size) {
  std::vector<float> resultMatrix(size * size, 0.0f);
  const int blockSize = 16;
  const int numBlocks = size / blockSize;

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int rowBlock = 0; rowBlock < numBlocks; ++rowBlock) {
    for (int colBlock = 0; colBlock < numBlocks; ++colBlock) {
      for (int innerBlock = 0; innerBlock < numBlocks; ++innerBlock) {
        for (int innerRow = 0; innerRow < blockSize; ++innerRow) {
          for (int innerCol = 0; innerCol < blockSize; ++innerCol) {
            float partialSum = 0.0f;
            for (int innerDepth = 0; innerDepth < blockSize; ++innerDepth) {
              partialSum +=
                  matrixA[(rowBlock * blockSize + innerRow) * size +
                          (innerBlock * blockSize + innerDepth)] *
                  matrixB[(innerBlock * blockSize + innerDepth) * size +
                          (colBlock * blockSize + innerCol)];
            }
#pragma omp atomic
            resultMatrix[(rowBlock * blockSize + innerRow) * size +
                         (colBlock * blockSize + innerCol)] += partialSum;
          }
        }
      }
    }
  }

  return resultMatrix;
}