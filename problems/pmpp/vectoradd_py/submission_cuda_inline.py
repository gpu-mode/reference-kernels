import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

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

def custom_kernel(data: input_t) -> output_t:
    """
    Custom implementation of vector addition using CUDA inline function.
    Args:
        inputs: List of pairs of tensors [A, B] to be added.
    Returns:
        List of tensors containing element-wise sums.
    """
    A, B = data

    assert A.is_cuda and B.is_cuda, "Input tensors must be on GPU"
    assert A.shape == B.shape, "Input tensors must have the same shape"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Input tensors must be float16"
    
    M, N = A.shape
    C = torch.empty_like(A)
    
    n_threads = 256
    n_blocks = (M * N + n_threads - 1) // n_threads
    
    cuda_source = """
    extern "C" __global__ void add_kernel(
        const half* __restrict__ A,
        const half* __restrict__ B,
        half* __restrict__ C,
        const int n_elements
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_elements) {
            C[idx] = __hadd(A[idx], B[idx]);
        }
    }
    """
    
    module = torch.utils.cpp_extension.load_inline(
        name=f"add_kernel_{M}_{N}",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["add_kernel"],
        with_cuda=True,
        extra_cuda_cflags=["-arch=sm_70"],  # Adjust based on your GPU architecture
    )
    
    module.add_kernel(
        cuda_stream=torch.cuda.current_stream(),
        args=[
            A.reshape(-1), B.reshape(-1), C.reshape(-1),
            M * N,
        ],
        blocks=n_blocks,
        threads=n_threads,
    )
    
    return C
