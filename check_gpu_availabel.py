import torch
import time

print(torch.cuda.is_available())      # Should print: True
print(torch.cuda.get_device_name(0)) # Should print: NVIDIA GeForce RTX 3090

device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

start = time.time()
for _ in range(1000):
    z = torch.matmul(x, y)
print("Elapsed:", time.time() - start)