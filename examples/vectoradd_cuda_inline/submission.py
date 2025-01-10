import torch
from torch.utils.cpp_extension import load_inline

add_cuda_source = """
template <typename scalar_t>
__global__ void add_kernel(const scalar_t* __restrict__ A, 
                           const scalar_t* __restrict__ B, 
                           scalar_t* __restrict__ C, 
                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same size");
    
    int N = A.numel();  
    auto C = torch::empty_like(A); 

    const int threads = 256; 
    const int blocks = (N + threads - 1) / threads;  
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "add_kernel", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

add_cpp_source = """
#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B);
"""



add_module = load_inline(
    name='add_cuda',
    cpp_sources=add_cpp_source,
    cuda_sources=add_cuda_source,
    functions=['add_cuda'],
    verbose=True,
)

def add(A, B):
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return add_module.add_cuda(A, B)

def custom_kernel(inputs):
    outputs = []
    for A, B in inputs:
        output = add(A, B)
        outputs.append(output)
    return outputs
