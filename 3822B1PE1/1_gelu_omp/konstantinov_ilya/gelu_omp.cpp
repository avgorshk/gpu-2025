#include "gelu_omp.h"
#include <cmath>
#include <vector>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

constexpr float kAlpha = 0.044715f;
constexpr float kBeta = 0.7978845608028654f;
constexpr float kOne = 1.0f;
constexpr float kHalf = 0.5f;

inline float FastGelu(float x) {
  const float x_cube = x * x * x;
  const float inner = kBeta * (x + kAlpha * x_cube);

  const float exp_val = std::exp(-2.0f * inner);
  const float tanh_approx = 1.0f - 2.0f / (1.0f + exp_val);

  return kHalf * x * (1.0f + tanh_approx);
}

#ifdef __AVX2__
inline __m256 FastGeluSIMD(__m256 x) {
  const __m256 alpha = _mm256_set1_ps(kAlpha);
  const __m256 beta = _mm256_set1_ps(kBeta);
  const __m256 one = _mm256_set1_ps(kOne);
  const __m256 half = _mm256_set1_ps(kHalf);
  const __m256 two = _mm256_set1_ps(2.0f);

  __m256 x_square = _mm256_mul_ps(x, x);
  __m256 x_cube = _mm256_mul_ps(x_square, x);

  __m256 alpha_x_cube = _mm256_mul_ps(alpha, x_cube);
  __m256 x_plus_alpha = _mm256_add_ps(x, alpha_x_cube);
  __m256 inner = _mm256_mul_ps(beta, x_plus_alpha);

  __m256 two_inner = _mm256_mul_ps(two, inner);
  __m256 exp_val = _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), two_inner));
  __m256 exp_plus_one = _mm256_add_ps(exp_val, one);
  __m256 two_div = _mm256_div_ps(two, exp_plus_one);
  __m256 tanh_approx = _mm256_sub_ps(one, two_div);

  __m256 one_plus_tanh = _mm256_add_ps(one, tanh_approx);
  __m256 x_times = _mm256_mul_ps(x, one_plus_tanh);
  return _mm256_mul_ps(half, x_times);
}
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const size_t size = input.size();
  std::vector<float> output(size);

  if (size == 0) {
    return output;
  }

#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
#endif

  constexpr size_t simd_size = 8;
  const size_t simd_loop_size = (size / simd_size) * simd_size;

#ifdef _OPENMP
#pragma omp parallel
  {
#ifdef __AVX2__
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < simd_loop_size; i += simd_size) {
      __m256 x_vec = _mm256_loadu_ps(&input[i]);
      __m256 result_vec = FastGeluSIMD(x_vec);
      _mm256_storeu_ps(&output[i], result_vec);
    }
#endif

#pragma omp for schedule(static)
    for (size_t i = simd_loop_size; i < size; ++i) {
      output[i] = FastGelu(input[i]);
    }
  }
#else

#ifdef __AVX2__
  for (size_t i = 0; i < simd_loop_size; i += simd_size) {
    __m256 x_vec = _mm256_loadu_ps(&input[i]);
    __m256 result_vec = FastGeluSIMD(x_vec);
    _mm256_storeu_ps(&output[i], result_vec);
  }
#endif

  for (size_t i = simd_loop_size; i < size; ++i) {
    output[i] = FastGelu(input[i]);
  }
#endif

  return output;
}