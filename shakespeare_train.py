import torch.optim as optim
import torch
import torch.nn as nn
from shakespeare_config import learning_rate, max_iters, eval_interval, device
from shakespeare_dataset import get_batch
from shakespeare_model import GPTLanguageModel

model = GPTLanguageModel().to(device)
# AdamW is the standard optimizar for transformers
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    print("Start training...")
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0:
            print(f"Step {iter}: Training Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "shakespeare_gpt.pth")
print(f"Training complete! Final loss = {loss.item():.4f}")