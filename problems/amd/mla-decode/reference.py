import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference

class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()
    
    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len
    
@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # Down-projection matrices
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, dtype=torch.bfloat16, bias=False)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, dtype=torch.bfloat16, bias=False)

        # Up-projection and rope projection matrices
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)

        # RoPE on half embeddings
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

        # Output projection
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6
   
    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        # seq_len = 1 always here
        batch_size, seq_len, model_dim = x.size()

        ################################################################################
        #                 Step 1: Handle down-projection + KV cache                    #
        ################################################################################
        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        ################################################################################
        #                  Step 2: Up-project and prepare NoPE + RoPE                  #
        ################################################################################

        # Handle queries Q first
        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Handle keys and values K/V. V does not need RoPE
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

        ################################################################################
        #                    Step 3: Handle RoPE Stream                                #
        ################################################################################

        # Compute RoPE for queries and combine with no-RoPE part
        q_rope = q_rope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q = torch.concat([q_nope, q_rope], dim=-1)


        # Compute RoPE for keys and combine with no-RoPE part
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1,self.n_heads,-1,-1)
        k_nope = k_nope.permute(0, 2, 1, 3) # bs x kv_len x n_heads x rope_head_dim
        k = torch.concat([k_nope, k_rope], dim=-1)
                
        ################################################################################
        #                        Compute Multi-head Attention                          #
        ################################################################################
        v = v.permute(0, 2, 1, 3) # bs x n_heads x kv_len x v_head_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)

        return y, kv_cache.get_data()

def generate_input(batchsize, dim, dq, prefill, seed):
    # Sizes derived from: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate weights for linear layers
    Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dq)
    KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(512)
    wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(128 * 128)

    config = Config(
        batch_size=batchsize,
        dim=dim,
        q_lora_rank=dq,
        n_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        seq_len=1,
        max_seq_len=8192,
        kv_cache_shape=(batchsize, 8192, 512 + 64),
        Q_proj_down_weight=Q_proj_down_weight,
        Q_proj_up_weight=Q_proj_up_weight,
        KV_proj_down_weight=KV_proj_down_weight,
        KV_proj_up_weight=KV_proj_up_weight,
        wo_weight=wo_weight,
    )
    x = torch.randn((config.batch_size, 1, config.dim), dtype=torch.bfloat16, generator=gen, device='cuda')
    
    # Pre-fill KV cache
    kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to('cuda')
    pre_filled_cache = torch.randn((config.batch_size, prefill, config.kv_lora_rank + config.qk_rope_head_dim), 
                                 dtype=torch.bfloat16, generator=gen, device='cuda')
    kv_cache(pre_filled_cache)

    return config, x, kv_cache

def ref_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data

    # Load in model weights
    model = MLA(config).to('cuda')
    model.Q_proj_down.weight = nn.Parameter(config.Q_proj_down_weight)
    model.Q_proj_up.weight = nn.Parameter(config.Q_proj_up_weight)
    model.KV_proj_down.weight = nn.Parameter(config.KV_proj_down_weight)
    model.KV_proj_up.weight = nn.Parameter(config.KV_proj_up_weight)
    model.wo.weight = nn.Parameter(config.wo_weight)

    output, kv_cache = model(x, kv_cache)
    return output, kv_cache

check_implementation = make_match_reference(ref_kernel, rtol=2e-02, atol=1e-03)  


def time_mla(model, x, kv_cache, num_warmup=3, num_trials=5):

    # Warmup runs
    for _ in range(1):
        output, _ = model(x, kv_cache)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_trials):
        kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to('cuda')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output, updated_kv = model(x, kv_cache)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    return output, updated_kv, avg_time, times

if __name__ == "__main__":
    # Generate test input
    batchsize = 128
    dim = 7168 
    dq = 1536
    prefill = 512
    seed = 97

    # Create model and inputs
    config, x, kv_cache = generate_input(batchsize, dim, dq, prefill, seed)
    model = MLA(config).to('cuda')

    # Run model with timing
    output, updated_kv, avg_time, times = time_mla(model, x, kv_cache)

    # Test reference kernel
    ref_output, ref_kv = ref_kernel((config, x, kv_cache))
    print("\nReference kernel output:")
    print(f"Output shape: {ref_output.shape}")
    print(f"KV cache shape: {ref_kv.shape}")
    print("\nFirst few values of reference output:")
    print(ref_output[0, :10])

    # Compare outputs
    print("\nOutput difference:")
    print(f"Max absolute difference: {torch.max(torch.abs(output - ref_output))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(output - ref_output))}")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Updated KV cache shape: {updated_kv.shape}")
    print("\nFirst few values of output:")
    print(output[0, :10])
    print(f"\nTiming results over {len(times)} runs (ms):")
    print(f"Average: {avg_time:.2f}")
    print(f"Individual times: {[f'{t:.2f}' for t in times]}")
