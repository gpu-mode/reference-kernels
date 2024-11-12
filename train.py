import torch

a = torch.Tensor([1, 2, 3, 4, 5]).cuda()
b= torch.Tensor([1, 2, 3, 4, 5]).cuda()

print(a)
print(b)
print(a + b)