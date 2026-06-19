import pandas as pd
import torch
from shakespeare_config import block_size, batch_size, device

shakespeare = pd.read_csv("shakespeare.csv")

text = "\n".join(shakespeare['text'].dropna().astype(str).tolist())
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
# print(f"stoi: {stoi}")
# print(f"\nitos: {itos}")
encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join(itos[i] for i in l) # deconder: list of integers -> string

# test_string = "Hello"
# print(f"Encoded '{test_string}': {encode(test_string)}")
# print(f"Decoded back: {decode(encode(test_string))}")

data = torch.tensor(encode(text), dtype=torch.long)
# print(f"Tensor shape: {data.shape}")
# print(f"Tensor data type: {data.dtype}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# print(f"train data: {len(train_data)}")
# print(f"validate data: {len(val_data)}")

# torch.manual_seed(42)

x_example = train_data[:block_size]
y_example = train_data[1:block_size+1]

# print("--- Target Shifting ---")
# print(f"Input sequence X: {x_example.tolist()}")
# print(f"Target sequence Y: {y_example.tolist()}")

def get_batch(split):
    data = train_data if split == 'train' else val_data # choose which dataset to pull from
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)