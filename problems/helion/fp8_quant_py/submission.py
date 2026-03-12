from task import input_t, output_t

import torch
import helion
import helion.language as hl
from pathlib import Path

COFIG_DICT={
    "block_sizes": [1],
    "num_warps": 1,
    "num_stages": 1,
}

ACF_FILE = "booster_pack/fp8_group_quant_0.acf"
if Path(ACF_FILE).exists():
    print(f"Using ACF file: {ACF_FILE}")
    COFIG_DICT["advanced_controls_file"] = ACF_FILE

# NOTE: This is an intentionally inefficient baseline implementation.
@helion.kernel(
    static_shapes=True,
    config=helion.Config(**COFIG_DICT),
)
def normalize_to_range(
    data: torch.Tensor,       # [N, G] input rows
    scales_out: torch.Tensor,  # [N] output normalization factors
) -> torch.Tensor:
    nrows = data.size(0)
    ncols = hl.specialize(data.size(1))
    MAX_VAL = 448.0

    qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

    for rr in hl.tile(nrows):
        row = data[rr, :].to(torch.float32)

        abs1 = torch.abs(row)
        amax1 = torch.amax(abs1, -1)
        abs2 = torch.abs(row)
        amax2 = torch.amax(abs2, -1)
        abs3 = torch.abs(row)
        amax3 = torch.amax(abs3, -1)
        amax = (amax1 + amax2 + amax3) / 3.0
        amax = torch.clamp(amax, min=1e-10)
        scale = amax / MAX_VAL

        q1 = row / scale[:, None]
        q2 = row / scale[:, None]
        q3 = row / scale[:, None]
        qout[rr, :] = (q1 + q2 + q3) / 3.0
        scales_out[rr] = scale

    return qout


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    flat_in = x.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    flat_q = normalize_to_range(flat_in, flat_s)

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
