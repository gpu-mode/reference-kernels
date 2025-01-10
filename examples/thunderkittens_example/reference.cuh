#ifndef __REFERENCE_CUH__
#define __REFERENCE_CUH__

#include <tuple>
#include <vector>

#define N_SIZES 5
const int Ns[N_SIZES] = {
    32, 32, 32, 32, 32,
};

using input_t = std::array<std::vector<float>, N_SIZES>;
using output_t = input_t;

input_t generate_input() {
  input_t data;

  for (int i = 0; i < N_SIZES; ++i) {
    data[i].resize(Ns[i]);
    for (int j = 0; j < Ns[i]; ++j) {
      data[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  return data;
}

output_t ref_kernel(input_t data) {
  output_t out;

  for (int i = 0; i < N_SIZES; ++i) {
    out[i].resize(Ns[i]);
    for (int j = 0; j < Ns[i]; ++j) {
      out[i][j] = data[i][j] + data[i][j];
    }
  }

  return out;
}

bool check_implementation(output_t out, output_t ref, float epsilon = 1e-5) {
  // input_t data = generate_input();
  // output_t reference_out = reference(data);
  bool same = true;

  for (int i = 0; i < N_SIZES; ++i) {
    auto ref_ptr = ref[i];
    auto out_ptr = out[i];

    for (int j = 0; j < Ns[i]; ++j) {
      if (std::fabs(ref_ptr[j] - out_ptr[j]) > epsilon) {
        same = false;
        break;
      }
    }
    if (!same)
      break;
  }

  return same;
}

#endif
