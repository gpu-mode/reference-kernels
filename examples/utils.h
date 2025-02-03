#ifndef POPCORN_UTILS_H
#define POPCORN_UTILS_H

#include <iostream>

// checks that a CUDA API call returned successfully, otherwise prints an error message and exits.
static inline void cuda_check_(cudaError_t status, const char* expr, const char* file, int line, const char* function)
{
    if(status != cudaSuccess) {
        std::cerr << "CUDA error (" << (int)status << ") while evaluating expression "
                  << expr << " at "
                  << file << '('
                  << line << ") in `"
                  << function << "`: "
                  << cudaGetErrorString(status) << std::endl;
        std::exit(110);
    }
}

// Convenience macro, automatically logs expression, file, line, and function name
// of the error.
#define CUDA_CHECK(expr) cuda_check_(expr, #expr, __FILE__, __LINE__, __FUNCTION__)

#endif
