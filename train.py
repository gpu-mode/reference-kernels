import torch

a = torch.Tensor([1, 2, 3, 4, 5]).to("cuda")
b= torch.Tensor([1, 2, 3, 4, 5]).to("cuda")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("No GPU available")

print(a + b)