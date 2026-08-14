#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <cfloat>

using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::accumulator;
using rocwmma::row_major;
using rocwmma::col_major;

#define D_C 512
#define D_R 64
#define D_TOTAL 576
#define H_Q 128
#define M_TILE 16
#define T_TILE 32
#define THREADS_PER_BLOCK 256
#define PACKED_BYTES 288
#define SCALE_BLOCKS 18
#define MX_BLOCK 32

using A_frag     = rocwmma::fragment<matrix_a, 16, 16, 16, hip_bfloat16, row_major>;
using B_frag     = rocwmma::fragment<matrix_b, 16, 16, 16, hip_bfloat16, row_major>;
using B_col_frag = rocwmma::fragment<matrix_b, 16, 16, 16, hip_bfloat16, col_major>;
using Acc_frag   = rocwmma::fragment<accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float mxfp4_dequant(uint8_t nibble, uint8_t scale_raw) {
    int sign = (nibble >> 3) & 1;
    int exp  = (nibble >> 1) & 3;
    int man  = nibble & 1;
    float v;
    if (exp == 0) {
        v = man ? 0.5f : 0.0f;
    } else {
        v = ldexpf(1.0f + 0.5f * man, exp - 1);
    }
    if (sign) v = -v;
    v = ldexpf(v, (int)scale_raw - 127);
    return v;
}

__global__ void mla_main_kernel(
    const hip_bfloat16* __restrict__ q_nope,
    const hip_bfloat16* __restrict__ q_pe,
    const uint8_t* __restrict__ kv_cache,
    const uint8_t* __restrict__ kv_scale,
    float sm_scale,
    float* __restrict__ partial_states,
    int B, int S_q, int S_kv,
    int total_queries, int split_k)
{
    __shared__ hip_bfloat16 q_tile[M_TILE * D_TOTAL];
    __shared__ hip_bfloat16 k_tile[T_TILE * D_TOTAL];
    __shared__ float scores[M_TILE * T_TILE];
    __shared__ hip_bfloat16 p_bf16[M_TILE * T_TILE];
    __shared__ float m_shared[M_TILE];
    __shared__ float l_shared[M_TILE];
    __shared__ float m_old_shared[M_TILE];

    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;

    const int q_tile_idx = blockIdx.x / split_k;
    const int split = blockIdx.x % split_k;
    const int q_start = q_tile_idx * M_TILE;

    const int split_size = (S_kv + split_k - 1) / split_k;
    const int s_start = split * split_size;
    const int s_end = min(s_start + split_size, S_kv);

    if (tid < M_TILE) {
        m_shared[tid] = -FLT_MAX;
        l_shared[tid] = 0.0f;
    }

    for (int idx = tid; idx < M_TILE * D_TOTAL; idx += THREADS_PER_BLOCK) {
        int i = idx / D_TOTAL;
        int f = idx % D_TOTAL;
        int gq = q_start + i;
        if (gq >= total_queries) {
            q_tile[idx] = __float2bfloat16(0.0f);
            continue;
        }
        int b = gq / (S_q * H_Q);
        int rem = gq % (S_q * H_Q);
        int sq = rem / H_Q;
        int h = rem % H_Q;
        if (f < D_C) {
            q_tile[idx] = q_nope[((size_t)(b * S_q + sq) * H_Q + h) * D_C + f];
        } else {
            q_tile[idx] = q_pe[((size_t)(b * S_q + sq) * H_Q + h) * D_R + (f - D_C)];
        }
    }
    __syncthreads();

    Acc_frag out_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) rocwmma::fill_fragment(out_acc[i], 0.0f);

    for (int s_tile = s_start; s_tile < s_end; s_tile += T_TILE) {
        int tile_size = min(T_TILE, s_end - s_tile);

        for (int idx = tid; idx < T_TILE * D_TOTAL; idx += THREADS_PER_BLOCK) {
            int local_s = idx / D_TOTAL;
            int feat = idx % D_TOTAL;
            int global_s = s_tile + local_s;
            if (local_s >= tile_size || global_s >= S_kv) {
                k_tile[idx] = __float2bfloat16(0.0f);
                continue;
            }
            int byte_idx = feat / 2;
            uint8_t packed = kv_cache[(size_t)global_s * PACKED_BYTES + byte_idx];
            uint8_t nibble = (feat % 2 == 0) ? (packed & 0xF) : (packed >> 4);
            int scale_block = feat / MX_BLOCK;
            uint8_t scale_raw = kv_scale[(size_t)global_s * SCALE_BLOCKS + scale_block];
            float v = mxfp4_dequant(nibble, scale_raw);
            k_tile[idx] = __float2bfloat16(v);
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < 16) {
            Acc_frag acc0, acc1;
            rocwmma::fill_fragment(acc0, 0.0f);
            rocwmma::fill_fragment(acc1, 0.0f);

            for (int k0 = 0; k0 < D_TOTAL; k0 += 16) {
                A_frag a_frag;
                rocwmma::load_matrix_sync(a_frag, q_tile + k0, D_TOTAL);

                B_col_frag b_frag0;
                rocwmma::load_matrix_sync(b_frag0, k_tile + k0, D_TOTAL);
                rocwmma::mma_sync(acc0, a_frag, b_frag0, acc0);

                B_col_frag b_frag1;
                rocwmma::load_matrix_sync(b_frag1, k_tile + 16 * D_TOTAL + k0, D_TOTAL);
                rocwmma::mma_sync(acc1, a_frag, b_frag1, acc1);
            }

            rocwmma::store_matrix_sync(scores, acc0, T_TILE, rocwmma::mem_row_major);
            rocwmma::store_matrix_sync(scores + 16, acc1, T_TILE, rocwmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < M_TILE * T_TILE; idx += THREADS_PER_BLOCK) {
            scores[idx] *= sm_scale;
        }
        __syncthreads();

        if (tid < M_TILE) {
            int i = tid;
            int gq = q_start + i;
            float m_old = m_shared[i];
            m_old_shared[i] = m_old;

            if (gq >= total_queries) {
                for (int s = 0; s < T_TILE; ++s) {
                    scores[i * T_TILE + s] = 0.0f;
                    p_bf16[i * T_TILE + s] = __float2bfloat16(0.0f);
                }
            } else {
                float m_cur = -FLT_MAX;
                for (int s = 0; s < T_TILE; ++s) {
                    if (s_tile + s < S_kv) {
                        m_cur = fmaxf(m_cur, scores[i * T_TILE + s]);
                    }
                }
                float m_new = fmaxf(m_old, m_cur);
                float l_cur = 0.0f;
                for (int s = 0; s < T_TILE; ++s) {
                    float p;
                    if (s_tile + s < S_kv) {
                        p = expf(scores[i * T_TILE + s] - m_new);
                    } else {
                        p = 0.0f;
                    }
                    scores[i * T_TILE + s] = p;
                    p_bf16[i * T_TILE + s] = __float2bfloat16(p);
                    l_cur += p;
                }
                float l_old = l_shared[i];
                float l_new = (m_old == -FLT_MAX) ? l_cur : (l_old * expf(m_old - m_new) + l_cur);
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();

        {
            int f_tile_base = warp_id * 8;
            float scale_row[16];
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                float mo = m_old_shared[r];
                float mn = m_shared[r];
                scale_row[r] = (mo == -FLT_MAX) ? 0.0f : expf(mo - mn);
            }

            #pragma unroll
            for (int ft = 0; ft < 8; ++ft) {
                __shared__ float tmp_frag[4][16 * 16];
                rocwmma::store_matrix_sync(tmp_frag[warp_id], out_acc[ft], 16, rocwmma::mem_row_major);
                __syncthreads();
                for (int e = lane_id; e < 256; e += 64) {
                    int r = e / 16;
                    tmp_frag[warp_id][e] *= scale_row[r];
                }
                __syncthreads();
                rocwmma::load_matrix_sync(out_acc[ft], tmp_frag[warp_id], 16);
                __syncthreads();

                int feat_col = (f_tile_base + ft) * 16;

                A_frag p_frag0, p_frag1;
                rocwmma::load_matrix_sync(p_frag0, p_bf16, T_TILE);
                rocwmma::load_matrix_sync(p_frag1, p_bf16 + 16, T_TILE);

                B_frag v_frag0, v_frag1;
                rocwmma::load_matrix_sync(v_frag0, k_tile + feat_col, D_TOTAL);
                rocwmma::load_matrix_sync(v_frag1, k_tile + 16 * D_TOTAL + feat_col, D_TOTAL);

                rocwmma::mma_sync(out_acc[ft], p_frag0, v_frag0, out_acc[ft]);
                rocwmma::mma_sync(out_acc[ft], p_frag1, v_frag1, out_acc[ft]);
            }
        }
        __syncthreads();
    }

    {
        __shared__ float o_shared[4][16 * 16];
        int f_tile_base = warp_id * 8;
        #pragma unroll
        for (int ft = 0; ft < 8; ++ft) {
            rocwmma::store_matrix_sync(o_shared[warp_id], out_acc[ft], 16, rocwmma::mem_row_major);
            __syncthreads();
            int feat_col = (f_tile_base + ft) * 16;
            for (int e = lane_id; e < 256; e += 64) {
                int i = e / 16;
                int fcol = e % 16;
                int gq = q_start + i;
                if (gq >= total_queries) continue;
                size_t out_offset = ((size_t)blockIdx.x * M_TILE + i) * (D_C + 2);
                partial_states[out_offset + feat_col + fcol] = o_shared[warp_id][e];
            }
            __syncthreads();
        }
    }

    if (tid < M_TILE) {
        int i = tid;
        int gq = q_start + i;
        if (gq < total_queries) {
            size_t out_offset = ((size_t)blockIdx.x * M_TILE + i) * (D_C + 2);
            partial_states[out_offset + D_C] = m_shared[i];
            partial_states[out_offset + D_C + 1] = l_shared[i];
        }
    }
}

__global__ void mla_reduction_kernel(
    const float* __restrict__ partial_states,
    hip_bfloat16* __restrict__ out,
    int B, int S_q, int S_kv,
    int split_k, int total_outputs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    int gq = idx / D_C;
    int k = idx % D_C;
    int q_tile_idx = gq / M_TILE;
    int local = gq % M_TILE;

    float m_global = -FLT_MAX;
    for (int s = 0; s < split_k; ++s) {
        size_t block_id = (size_t)q_tile_idx * split_k + s;
        size_t base = (block_id * M_TILE + local) * (D_C + 2);
        float m_partial = partial_states[base + D_C];
        m_global = fmaxf(m_global, m_partial);
    }

    float l_global = 0.0f;
    float o_global = 0.0f;
    for (int s = 0; s < split_k; ++s) {
        size_t block_id = (size_t)q_tile_idx * split_k + s;
        size_t base = (block_id * M_TILE + local) * (D_C + 2);
        float m_partial = partial_states[base + D_C];
        float l_partial = partial_states[base + D_C + 1];
        float o_partial = partial_states[base + k];
        float correction = expf(m_partial - m_global);
        l_global += l_partial * correction;
        o_global += o_partial * correction;
    }

    float result = (l_global > 0.0f) ? (o_global / l_global) : 0.0f;
    out[idx] = __float2bfloat16(result);
}

torch::Tensor forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor kv_cache,
    torch::Tensor kv_scale,
    double sm_scale)
{
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be on device");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be on device");
    TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on device");
    TORCH_CHECK(kv_scale.is_cuda(), "kv_scale must be on device");

    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bf16");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "q_pe must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(kv_scale.scalar_type() == torch::kUInt8, "kv_scale must be uint8");

    q_nope = q_nope.contiguous();
    q_pe = q_pe.contiguous();
    kv_cache = kv_cache.contiguous();
    kv_scale = kv_scale.contiguous();

    int B = q_nope.size(0);
    int S_q = q_nope.size(1);
    int H_q_dim = q_nope.size(2);
    TORCH_CHECK(H_q_dim == H_Q, "H_q mismatch");
    int S_kv = kv_cache.size(1);

    auto out = torch::empty({B, S_q, H_Q, D_C}, q_nope.options());

    int total_queries = B * S_q * H_Q;
    int q_tiles = (total_queries + M_TILE - 1) / M_TILE;

    int split_k = 1;
    while (q_tiles * split_k < 304 && split_k * T_TILE < S_kv) split_k++;

    auto partial_states = torch::empty(
        {(long)q_tiles * split_k, (long)M_TILE, (long)(D_C + 2)},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid_main(q_tiles * split_k);
    dim3 block_main(THREADS_PER_BLOCK);

    mla_main_kernel<<<grid_main, block_main, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<const hip_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(),
        kv_scale.data_ptr<uint8_t>(),
        (float)sm_scale,
        partial_states.data_ptr<float>(),
        B, S_q, S_kv,
        total_queries, split_k);

    C10_HIP_CHECK(hipGetLastError());

    int total_outputs = B * S_q * H_Q * D_C;
    int threads_red = 256;
    int blocks_red = (total_outputs + threads_red - 1) / threads_red;

    mla_reduction_kernel<<<blocks_red, threads_red, 0, stream>>>(
        partial_states.data_ptr<float>(),
        reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, S_q, S_kv,
        split_k, total_outputs);

    C10_HIP_CHECK(hipGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MLA forward (MI300X, split-K, rocWMMA)");
}
