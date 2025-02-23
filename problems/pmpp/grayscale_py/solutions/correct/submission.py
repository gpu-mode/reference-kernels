from task import input_t, output_t
import torch

def custom_kernel(data: input_t) -> output_t:
    weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                         device=data.device, 
                         dtype=data.dtype)
    return torch.sum(data * weights, dim=-1)
