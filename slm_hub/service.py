from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
from graph.config import Config

# Determine device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model registry with type definitions
MODEL_REGISTRY = {
    "objective_analysis": {
        "model": AutoModelForSequenceClassification.from_pretrained(Config.Path.APP_HOME / "slm_hub" / "models" / "objective_analysis").to(device),
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "type": "classification"
    },
    #"objective_generation": {
    #    "model": AutoModelForSeq2SeqLM.from_pretrained("models/objective_generation").to(device),
    #    "tokenizer": AutoTokenizer.from_pretrained("models/objective_generation"),
    #    "type": "generation"
    #}
}

app = FastAPI()

# Request structure
class ModelInput(BaseModel):
    model_name: str  # Model selection
    text: str        # Input text

def run_inference(model, tokenizer, text, model_type):
    """ Asynchronously runs inference based on model type. """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        if model_type == "classification":
            logits = model(**inputs).logits
            logits = torch.sigmoid(logits).squeeze(0).tolist()
            return {"prediction": logits}

        elif model_type == "generation":
            outputs = model.generate(**inputs, max_length=100)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"generated_text": generated_text}

@app.post("/predict")
def predict(input_data: ModelInput):
    model_name = input_data.model_name.lower()

    # Check if model exists
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")

    # Get model, tokenizer, and type
    model_data = MODEL_REGISTRY[model_name]
    model, tokenizer, model_type = model_data["model"], model_data["tokenizer"], model_data["type"]

    # Run inference
    result = run_inference(model, tokenizer, input_data.text, model_type)
    
    return {
        "model": model_name,
        "text": input_data.text,
        **result  # Includes either prediction or generated text
    }
