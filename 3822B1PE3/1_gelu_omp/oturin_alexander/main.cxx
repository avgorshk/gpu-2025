#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "fastRNG.hpp"
#include "gelu_omp.hpp"
#include "gelu_sequential.hpp"

#define SEQUENTIAL_RUN

int main() {
  size_t n = 134217728;
  std::vector<float> input(n);
  std::vector<float> output(n);

  std::cout << n << " elements" << std::endl;

#ifdef SEQUENTIAL_RUN
  {
    // Warming-up
    random_fill(input, -3, 3);
    output = GeluSEQ(input);

    // Performance Measuring
    std::vector<double> time_list;
    for (size_t i = 0; i < 4; i++) {
      random_fill(input, -3, 3);
      auto start = std::chrono::high_resolution_clock::now();
      output = GeluSEQ(input);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << "Sequential run time: " << time << std::endl;
  }
#endif

  // Warming-up
  random_fill(input, -3, 3);
  output = GeluOMP(input);

  // Performance Measuring
  std::vector<double> time_list;
  for (size_t i = 0; i < 4; i++) {
    random_fill(input, -3, 3);
    auto start = std::chrono::high_resolution_clock::now();
    output = GeluOMP(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time_list.push_back(duration.count());
  }
  double time = *std::min_element(time_list.begin(), time_list.end());

  std::cout << "OpenMP run time: " << time << std::endl;
}
