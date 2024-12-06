# Qwen-with-FastAPI

Steps I have taken : 

1. Setting Up the Server
Downloading the Model:
I downloaded the Qwen/Qwen2.5-0.5B model from the Hugging Face repository. The download included the required files such as config.json, pytorch_model.bin, and tokenizer files.

Installing Dependencies:
I installed the necessary libraries, including FastAPI, Uvicorn, transformers, and torch using the following commands:


pip install fastapi uvicorn transformers torch

Writing the API Code:
I created a Python script to serve the model using FastAPI. The steps were:

Loading the Model: I used the Hugging Face AutoModelForCausalLM and AutoTokenizer classes to load the Qwen model and tokenizer.
Creating API Endpoints: I defined an API endpoint /generate that accepts text input and returns the model’s generated response.
Running the Server:
I used Uvicorn to run the server:

uvicorn main:app --reload
The server started successfully and hosted the API at http://127.0.0.1:8000.

2. Performing Inference
Testing the API with Postman:
I used Postman to test the API functionality:

Created a POST request with the URL http://127.0.0.1:8000/generate.
Added a JSON body with a text prompt, such as:

{
  "prompt": "What is the capital of France?"
}
Received a JSON response containing the model-generated output:

{
  "response": "The capital of France is Paris."
}
Processing Responses:
The response text was extracted from the model’s output and returned via the API.
