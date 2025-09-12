import torch

x = torch.tensor([1, 2, 3])
print(x)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())