import torch
from shakespeare_model import GPTLanguageModel
from shakespeare_dataset import encode, decode
from shakespeare_config import device, block_size

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('shakespeare_gpt.pth', map_location=device, weights_only=True))
model.eval()

print("==================================================")
print("WELCOME TO THE INFINITE PLAY")
print("You are co-writing a play with an unpredictable AI.")
print("Format your inputs like a script (e.g., 'ROMEO: Look out!').")
print("Type 'quit' to exit.")
print("==================================================\n")

story_context = "SCENE I. A mysterious digital realm.\n\n"
print(story_context)

while True:
    # 1. Get your move
    user_line = input("You: ")
    if user_line.lower() == 'quit':
        print("\nExeunt. (Thanks for playing!)")
        break

    story_context += user_line + "\n\nSHAKESPEARE-BOT: "

    context_idx = torch.tensor([encode(story_context)], dtype=torch.long).to(device)

    if context_idx.size(1) > block_size:
        context_idx = context_idx[:, -block_size:]

    generated_indices = model.generate(context_idx, max_new_tokens=100, temperature=0.9)

    new_text = decode(generated_indices[0][context_idx.size(1):].tolist())

    bot_turn = new_text.split('\n\n')[0]

    print(f"\nShakespeare-Bot: {bot_turn}\n")

    story_context += bot_turn + "\n\n"