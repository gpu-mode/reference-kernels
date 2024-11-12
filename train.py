import triton.language as tl
import triton
import torch


@triton.jit
def vector_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    # Get the unique program ID for each block
    pid = tl.program_id(0)

    # Calculate the start index for each block
    start = pid * BLOCK_SIZE

    # Load data from A and B into registers for vector addition
    offset = start + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offset, mask=offset < N)  # Load elements from A
    b = tl.load(B + offset, mask=offset < N)  # Load elements from B

    # Perform element-wise addition
    c = a + b

    # Store the result back into C
    tl.store(C + offset, c, mask=offset < N)


a = torch.Tensor([1, 2, 3, 4, 5]).cuda()
b = torch.Tensor([1, 2, 3, 4, 5]).cuda()

print(a)
print(b)
print(a + b)

