import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t

class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config["n_routed_experts"])
        ])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat,
                                            flat_expert_indices,
                                            flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self,
                  x: torch.Tensor,
                  flat_expert_indices: torch.Tensor,
                  flat_expert_weights: torch.Tensor
                 ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out    = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

        return expert_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Submission template for DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        gate_proj_weight = weights[f'experts.{i}.0.weight']
        up_proj_weight = weights[f'experts.{i}.1.weight']
        down_proj_weight = weights[f'experts.{i}.2.weight']

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    # Run the model
    output = moe(input_tensor)

    return output