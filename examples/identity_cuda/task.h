#ifndef __TASK_H__
#define __TASK_H__

#include <vector>
#include <array>

#define N_SIZES 10
const int Ns[N_SIZES] = {128,  256,  512,   1024,  2048,
                         4096, 8192, 16384, 32768, 65536};

using input_t = std::array<std::vector<float>, N_SIZES>;
using output_t = input_t;

#endif
