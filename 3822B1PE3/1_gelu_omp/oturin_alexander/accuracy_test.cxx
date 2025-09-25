#include <iostream>

#include "fastRNG.hpp"
#include "gelu_omp.hpp"
#include "gelu_sequential.hpp"

#define SEQUENTIAL_RUN

int main() {
  size_t n = 2048;
  std::vector<float> input(n);
  std::vector<float> output1(n);
  std::vector<float> output2(n);
  std::vector<float> output3(n);

  std::cout << n << " elements" << std::endl;
  std::pair<float, float> range(-3, 3);
  std::cout << "In range: " << range.first << ' ' << range.second << std::endl;

  random_fill(input, range.first, range.second);
  output1 = GeluSEQ(input);
    
  output2 = GeluOMP(input);

  output3 = GeluOMPapprox(input);

  std::cout << "SEQ " << ((output1 == output2) ? "EQUALS" : "NOT EQUAL") << " OMP" << std::endl;

  float max_difference = 0;
  float min_difference = INFINITY;
  float avg_difference = 0;
  for (size_t i = 0; i < n; i++) {
    float diff = std::abs(output1[i] - output3[i]);
    if (diff > max_difference)
      max_difference = diff;
    if (diff < min_difference)
      min_difference = diff;
    avg_difference += diff;
  }

  avg_difference /= n;

  std::cout << "Max difference: " << max_difference << std::endl;
  std::cout << "Avg difference: " << avg_difference << std::endl;
  std::cout << "Min difference: " << min_difference << std::endl;
}
