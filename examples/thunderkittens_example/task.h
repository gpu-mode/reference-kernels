#ifndef __POPCORN_TASK_H__
#define __POPCORN_TASK_H__

#include <array>
#include <vector>

using input_t = std::vector<float>;
using output_t = input_t;

constexpr std::array<const char *, 2> ArgumentNames = {"seed", "size"};

#endif
