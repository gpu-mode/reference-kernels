#include <array>
#include <vector>
#include "reference.cuh"

__global__ void copy_kernel(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = input[idx];
    }
}

output_t submission(input_t data)
{
    output_t result;

    for (int i = 0; i < N_SIZES; ++i)
    {
        int N = Ns[i];
        result[i].resize(N);

        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, data[i].data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        copy_kernel<<<numBlocks, blockSize>>>(d_input, d_output, N);

        // Copy result back to host
        cudaMemcpy(result[i].data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }

    return result;
}
