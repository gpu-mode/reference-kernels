import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()

def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled group GEMM.
    """
    abc_tensors, sfasfb_tensors, problem_sizes = data
    
    result_tensors = []
    for i, (
        (a_ref, b_ref, c_ref),
        (sfa_ref, sfb_ref),
        (m, n, k, l),
    ) in enumerate(
        zip(
            abc_tensors,
            sfasfb_tensors,
            problem_sizes,
        )
    ):
        for l_idx in range(l):
            # Convert the scale factor tensor to blocked format
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            # (m, k) @ (n, k).T -> (m, n)
            res = torch._scaled_mm(
                a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a.cuda(),
                scale_b.cuda(),
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, :, l_idx] = res
        result_tensors.append((c_ref))
    return result_tensors

def generate_input(
    m: int,
    n: int,
    k: int,
    g: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled group GEMM. 
    Each group can have different m, n, k, l.
    
    Args:
        problem_sizes: List of tuples (m, n, k, l) for each problem
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Number of columns in A and rows of B
        l: Batch size, always is 1
        groups: Number of groups
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (list(tuple(a, b, c)), list(tuple(sfa, sfb)), list(tuple(m, n, k, l))) where each group has its own a, b, c, sfa, sfb.
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [n, k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, n, l] - Output matrix in torch.float16 data type
    """
    torch.manual_seed(seed)
    
    abc_tensors = []
    sfasfb_tensors = []
    problem_sizes = []
    l = 1
    # Generate a, b, c, sfa, sfb tensors for all groups
    for group_idx in range(g):
        a_ref = torch.randint(
            0, 2, (l, m, k // 2), dtype=torch.uint8, device="cuda"
        ).permute(1, 2, 0)
        b_ref = torch.randint(
            0, 2, (l, n, k // 2), dtype=torch.uint8, device="cuda"
        ).permute(1, 2, 0)
        a_ref = a_ref.view(torch.float4_e2m1fn_x2)
        b_ref = b_ref.view(torch.float4_e2m1fn_x2)

        c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
            1, 2, 0
        )

        sf_k = ceil_div(k, sf_vec_size)
        sfa_ref_cpu = torch.randint(
            1, 3, (l, m, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)
        sfb_ref_cpu = torch.randint(
            1, 3, (l, n, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        problem_sizes.append((m, n, k, l))
    
    return (abc_tensors, sfasfb_tensors, problem_sizes)

check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-02)
