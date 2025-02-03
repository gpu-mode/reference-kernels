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

static input_t generate_input(int seed) {
  std::mt19937 rng(seed);
  input_t data;

  std::uniform_real_distribution<float> dist(0, 1);

  for (int i = 0; i < N_SIZES; ++i) {
    data[i].resize(Ns[i]);
    for (int j = 0; j < Ns[i]; ++j) {
      data[i][j] = dist(rng);
    }
  }

  return data;
}

// The identity kernel
static output_t ref_kernel(input_t data) {
  return (output_t) data;
}

static bool check_implementation(output_t out, output_t ref, float epsilon = 1e-5) {
  // input_t data = generate_input();
  // output_t reference_out = reference(data);

  for (int i = 0; i < N_SIZES; ++i) {
    auto ref_ptr = ref[i];
    auto out_ptr = out[i];

    if(out[i].size() != Ns[i]) {
        std::cerr <<  "SIZE MISMATCH at " << i << ": " << Ns[i] << " " << out[i].size() << std::endl;
        return false;
    }

    for (int j = 0; j < Ns[i]; ++j) {
      if (std::fabs(ref_ptr[j] - out_ptr[j]) > epsilon) {
        std::cerr <<  "ERROR AT " << i << ", "<< j << ": " << ref_ptr[j] << " " << out_ptr[j] << std::endl;
        return false;
      }
    }
  }

  return true;
}

#endif
