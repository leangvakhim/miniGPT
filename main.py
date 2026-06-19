from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from shakespeare_dataset import encode, decode
from shakespeare_config import device
from shakespeare_model import GPTLanguageModel

# 1. Initialize the FastAPI app
app = FastAPI(title="Sports Chatbot API")

# 2. Setup CORS (Crucial for the Frontend)
# This allows a frontend running on a different port/domain to talk to your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('shakespeare_gpt.pth', map_location=device, weights_only=True))
model.eval()

# 3. Define the Request Data Schema
# This tells the API to expect a JSON object containing a "message" string
class ChatRequest(BaseModel):
    message: str

# 4. Define the Response Data Schema
# This structures how the API sends data back to the frontend
class ChatResponse(BaseModel):
    reply: str

# 5. Your Model Logic
def generate_sports_response(user_input: str) -> str:

    if not user_input:
        user_input = "\n"

    context = torch.tensor([encode(user_input)], dtype=torch.long).to(device)

    print("Generating text...\n")

    generated_indices = model.generate(context, max_new_tokens=500, temperature=0.7)
    generated_text = decode(generated_indices[0].tolist())

    return generated_text

@app.get("/")
async def root():
    return {"message": "API is running! Go to /docs to test the endpoints."}

# 6. Create the API Endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Extract the string from the frontend request
    user_message = request.message

    # Pass it to your model
    bot_reply = generate_sports_response(user_message)

    # Package it into the Pydantic response model and send it back
    return ChatResponse(reply=bot_reply)