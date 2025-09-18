import torch, platform

print("torch:", torch.__version__)
print("cuda in torch:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("sm arch capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
print("cudnn:", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "n/a")
print("python:", platform.python_version())
