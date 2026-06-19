import torch

batch_size = 4
block_size = 256
n_embd = 32
n_head = 4
n_layer = 3

learning_rate = 0.001
max_iters = 20000
eval_interval = 100

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(f"device is: {device}")