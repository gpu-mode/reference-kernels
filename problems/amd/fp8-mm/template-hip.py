# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr const int BLOCK = 128;

__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, 
                   __hip_bfloat16* c, int m, int n, int k) {
                   
    // Your implementation here
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    // split loop into an outer loop over different blocks, and an inner loop within one block.
    // we can assume k % BLOCK == 0.
    for(int i = 0; i < k; i += BLOCK) {
        // block results accumulates the inner product across a single block.
        // within each block, scales are constant, so we can lift the scaling 
        // outside of the inner loop.
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            // load input matrix elements and convert to float for computations
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }
        
        // before we can go to the next block, scale the result of the current block
        // and accumulate to final result
        // note the different indexing into as and bs
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    
    // finally, write the result as bf16
    c[cx * n + cy] = (__hip_bfloat16)result;
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    custom_kernel<<<dim3((m+15)/16, (n+15)/16), dim3(16, 16), 0, 0>>> ((__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(), 
    as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c

