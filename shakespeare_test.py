import torch
from shakespeare_dataset import encode, decode
from shakespeare_config import device
from shakespeare_model import GPTLanguageModel

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('shakespeare_gpt.pth', map_location=device, weights_only=True))
model.eval()

context = torch.tensor([encode('\n')], dtype=torch.long).to(device)

print("Generating text...\n")

generated_indices = model.generate(context, max_new_tokens=500, temperature=0.7)
generated_text = decode(generated_indices[0].tolist())

print(generated_text)