#ifndef __POPCORN_TASK_H__
#define __POPCORN_TASK_H__

#include <array>
#include <vector>

#define N_SIZES 5
const int Ns[N_SIZES] = {32, 32, 32, 32, 32};

using input_t = std::array<std::vector<float>, N_SIZES>;
using output_t = input_t;

#endif
