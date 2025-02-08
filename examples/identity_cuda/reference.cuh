#ifndef __REFERENCE_CUH__
#define __REFERENCE_CUH__

#include <tuple>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <array>
#include <random>
#include <iostream>

#include "task.h"

static input_t generate_input(int seed, int size) {
  std::mt19937 rng(seed);
  input_t data;

  std::uniform_real_distribution<float> dist(0, 1);

  data.resize(size);
  for (int j = 0; j < size; ++j) {
    data[j] = dist(rng);
  }

  return data;
}

// The identity kernel
static output_t ref_kernel(input_t data) {
  return (output_t) data;
}

static void check_implementation(TestReporter& reporter, input_t data, output_t out, float epsilon = 1e-5) {
  // input_t data = generate_input();
  output_t ref = ref_kernel(data);

  if(out.size() != ref.size()) {
      if(!reporter.check_equal("size mismatch", out.size(), ref.size())) return;
  }

  for (int j = 0; j < ref.size(); ++j) {
    if (std::fabs(ref[j] - out[j]) > epsilon) {
        reporter.fail() << "error at " << j << ": " << ref[j] << " "  << std::to_string(out[j]);
        return;
    }
  }

  reporter.pass();
}

#endif
