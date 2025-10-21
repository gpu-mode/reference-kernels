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
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, _, _, _, c_ref = data
    
    # Get dimensions from MxNxL layout
    m, n, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    ref1 = torch.empty(
        (l, m, n),
        dtype=torch.float32,
        device="cuda",
    ).permute(1, 2, 0)
    ref2 = torch.empty(
        (l, m, n),
        dtype=torch.float32,
        device="cuda",
    ).permute(1, 2, 0)
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref1[:, :, l_idx] = res1

        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = res2
    # Do silu on the first GEMM result and multiply with the second GEMM result
    c_ref = (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)
    return c_ref


def generate_input(
    m: int,
    n: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled dual GEMM with silu activation,
    C = silu(A @ B1) * (A @ B2).
    
    Args:
        m: Number of rows in matrix A
        n: Number of columns in matrix B1 and B2
        k: Number of columns in A and rows of B1 and B2
        l: Batch size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b1: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b2: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b1: [n, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b2: [n, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b1_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b2_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, n, l] - Output matrix in torch.float16 data type
    """
    torch.manual_seed(seed)
    
    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 2, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    b1_ref = torch.randint(
        0, 2, (l, n, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    b2_ref = torch.randint(
        0, 2, (l, n, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b1_ref = b1_ref.view(torch.float4_e2m1fn_x2)
    b2_ref = b2_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )
    
    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. Please note this data reordering function 
    # is very slow.
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, l, sf_k) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(1, 3, ref_shape, dtype=torch.int8)
        ref_f8_torch_tensor_cpu = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_cpu_permuted = ref_f8_torch_tensor_cpu.permute(
            *ref_permute_order
        )

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

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(0, 2, mma_shape, dtype=torch.int8)
        reordered_f8_torch_tensor_cpu = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor_cpu = reordered_f8_torch_tensor_cpu.permute(
            *mma_permute_order
        )

        for i in range(mn):
            for j in range(sf_k):
                for b in range(l):
                    # Calculate the location in MMA shape
                    mm = i // (atom_m[0] * atom_m[1])
                    mm32 = i % atom_m[0]
                    mm4 = (i % 128) // atom_m[0]
                    kk = j // atom_k
                    kk4 = j % atom_k
                    reordered_f8_torch_tensor_cpu[mm32, mm4, mm, kk4, kk, b] = ref_f8_torch_tensor_cpu_permuted[i, j, b]
        return ref_f8_torch_tensor_cpu_permuted, reordered_f8_torch_tensor_cpu.cuda()

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb1_ref_cpu, sfb1_ref_permuted = create_scale_factor_tensors(l, n, sf_k)
    sfb2_ref_cpu, sfb2_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    return (a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted, c_ref)

check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-02)
