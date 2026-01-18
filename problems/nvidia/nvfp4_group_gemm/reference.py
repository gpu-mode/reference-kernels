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
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # Pad the input matrix if necessary
    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
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
    abc_tensors, sfasfb_tensors, _, problem_sizes = data
    
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


# Helper function to prepare the scale factor tensors for both reference
# kernel and customize kernel. The customized data layout can be found in:
# https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
    sf_k = ceil_div(k, sf_vec_size)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # batch size
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    # Create the reordered scale factor tensor (32, 4, rest_m, 4, rest_k, l) on GPU.
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    # Generate a random int8 tensor, then convert to float8_e4m3fn
    rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device='cuda')
    reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    # Permute according to mma_permute_order
    reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

    # Move ref_f8_tensor to GPU if not already there
    if ref_f8_tensor.device.type == 'cpu':
        ref_f8_tensor = ref_f8_tensor.cuda()

    # GPU-side vectorized reordering (replaces slow CPU nested loops)
    # Create index grids for all dimensions
    i_idx = torch.arange(mn, device='cuda')
    j_idx = torch.arange(sf_k, device='cuda')
    b_idx = torch.arange(l, device='cuda')
    
    # Create meshgrid for all combinations of (i, j, b)
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
    
    # Calculate target indices in vectorized manner
    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k
    
    # Perform the reordering with advanced indexing (all on GPU)
    reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[i_grid, j_grid, b_grid]
    
    return reordered_f8_tensor


def generate_input(
    m: tuple,
    n: tuple,
    k: tuple,
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
        Tuple of (list(tuple(a, b, c)), list(tuple(sfa, sfb)), list(tuple(sfa_reordered, sfb_reordered)), list(tuple(m, n, k, l))) where each group has its own a, b, c, sfa, sfb.
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            sfa: [m, k // 16, l] - Input scale factors in torch.float8e4m3fn data type
            sfb: [n, k // 16, l] - Input scale factors in torch.float8e4m3fn data type
            sfa_reordered: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            sfb_reordered: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, n, l] - Output matrix in torch.float16 data type
    """
    torch.manual_seed(seed)
    
    abc_tensors = []
    sfasfb_tensors = []
    sfasfb_reordered_tensors = []
    problem_sizes = []
    l = 1
    # Generate a, b, c, sfa, sfb tensors for all groups
    for group_idx in range(g):
        mi = m[group_idx]
        ni = n[group_idx]
        ki = k[group_idx]
        a_ref = torch.randint(
            -1, 2, (l, mi, ki // 2), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
        b_ref = torch.randint(
            -1, 2, (l, ni, ki // 2), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
        a_ref = a_ref.view(torch.float4_e2m1fn_x2)
        b_ref = b_ref.view(torch.float4_e2m1fn_x2)

        c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda").permute(
            1, 2, 0
        )

        sf_k = ceil_div(ki, sf_vec_size)
        sfa_ref_cpu = torch.randint(
            1, 3, (l, mi, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)
        sfb_ref_cpu = torch.randint(
            1, 3, (l, ni, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)

        sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu)
        sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
        problem_sizes.append((mi, ni, ki, l))
    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
