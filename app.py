from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_name = "Qwen/Qwen2.5-0.5B"  
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = FastAPI()

# Define a request format
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_response(request: PromptRequest):
    prompt = request.prompt

    # Generate a response using the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Convert prompt to input IDs
    output = model.generate(
    input_ids, 
    max_length=100,  # Allow more tokens to generate a longer response
    num_beams=5,     # Use beam search for better quality responses
    repetition_penalty=2.0,  # Penalize repeated phrases
    temperature=0.7,  # Adjust creativity (lower = deterministic, higher = creative)
    top_k=50,         # Use top-k sampling to limit choices
    top_p=0.9         # Use nucleus sampling for balanced creativity
)
    response = tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the output into a human-readable string

    # Remove prompt text if echoed in the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return {"response": response}
