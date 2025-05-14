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
        theta = 10000 ** (-torch.arange(0, d_model//2, dtype=torch.float) / (d_model//2))
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
        cos = idx_theta2.cos()
        sin = idx_theta2.sin()
        return x * cos + self.rotate_half(x) * sin

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple, sk: int) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape))
        self.sk = sk
        self.zero()

    def zero(self) -> None:
        self.data.zero_()

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.sk + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.sk : self.sk + c_kv.size(1), :
        ] = c_kv
        self.sk += c_kv.size(1)

        return self.data[:, :self.sk], self.sk
    
@dataclass
class Config:
    bs: int
    sq: int
    sk: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    max_seq_len: int

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, dtype=torch.bfloat16, bias=False)
        self.wq_b = nn.Linear(self.q_lora_rank, (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.q_rope = RoPE(self.qk_rope_head_dim)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, dtype=torch.bfloat16, bias=False)
        self.wkv_b = nn.Linear(self.kv_lora_rank, (self.qk_nope_head_dim + self.v_head_dim)* self.n_heads, dtype=torch.bfloat16, bias=False)
        self.k_rope = RoPE(self.qk_rope_head_dim)
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps =1e-6
   
    def forward(self,x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.wq_a(x)
        q = self.wq_b(q) 
        q = q.view(bsz, seq_len, self.n_heads, self.qk_rope_head_dim + self.qk_nope_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = q_rope.view(bsz, self.n_heads, seq_len, self.qk_rope_head_dim)
        q_nope = q_nope.view(bsz, self.n_heads, seq_len, self.qk_nope_head_dim)
        start_pos = 0
        q_rope = self.q_rope(q_rope, start_pos)
        q = torch.cat([q_nope, q_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)
        
        kv = self.wkv_a(x)
        kv, sk = kv_cache(kv)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2)
        k_pe = self.k_rope(k_pe, start_pos).view(bsz, 1, sk, self.qk_rope_head_dim)
        kv = self.wkv_b(kv).view(bsz, self.n_heads, sk, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv,[self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.concat([k_nope, k_pe.expand(-1, self.n_heads, -1, -1)], dim=-1)
                
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.qk_rope_head_dim + self.qk_nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v)
        return y

def generate_input(bs, sq, sk, dim, seed):
    config = Config(
        bs=bs,
        sq=sq,
        sk=sk,
        dim=dim,
        n_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_seq_len=8192
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn((bs, sq, dim), dtype=torch.bfloat16, generator=gen)
    kv_cache = KVCache((config.bs, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim), sk=config.sk)
    return config, x, kv_cache

def ref_kernel(data: input_t) -> output_t:
    config, x, kv_cache= data
    model = MLA(config)
    output = model(x, kv_cache)
    return output

check_implementation = make_match_reference(ref_kernel, rtol=2e-02, atol=1e-03)  