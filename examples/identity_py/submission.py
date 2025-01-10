from typing import List

import torch


# Here, InputType = OutputType = List[torch.Tensor]
def custom_kernel(input: List[torch.Tensor]) -> List[torch.Tensor]:
    return input
