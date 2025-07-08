from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import uvicorn

# Add the parent directory to the Python path
# to handle ModuleNotFoundError: No module named 'models' error
import sys
sys.path.append('../') 

# 📦 Load models
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#show device
print(f"Using device: {device}")

# Replace paths with YOUR trained model checkpoints!
from models.transformer_model_definitions import PatchTST as PatchTSTModel   # Import your model classes
from models.lstm_forecast_model import LSTMForecast as LSTMModel
from fastapi import Request

# Initialize the LSTM model architecture
lstm_model = LSTMModel()  
lstm_model.load_state_dict(torch.load("../models/checkpoints/lstm_model.pth"))
lstm_model = lstm_model.to(device)

# Initialize the PatchTST model architecture
patchtst_model = PatchTSTModel() 
patchtst_model.load_state_dict(torch.load("../models/checkpoints/patchTST_transformer_model.pth"))
patchtst_model = patchtst_model.to(device)

# Load the fine-tuned LLM model
llm_tokenizer = AutoTokenizer.from_pretrained("../models/llm_forecaster/")
llm_model = AutoModelForCausalLM.from_pretrained("../models/llm_forecaster/").to(device)

lstm_model.eval()
patchtst_model.eval()
llm_model.eval()

# 📜 Prediction function
def predict(input_series):
    input_series = np.array(input_series)
    assert len(input_series) >= 96, "Input must be at least 96 values."

    # LSTM predict
    lstm_input = torch.tensor(input_series[-96:], dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    lstm_pred = lstm_model(lstm_input).cpu().detach().numpy().flatten()[0]

    # PatchTST predict
    patch_size = 8
    patches = torch.tensor(input_series[-96:], dtype=torch.float32).view(1, 96//patch_size, patch_size).to(device)
    patchtst_pred = patchtst_model(patches).cpu().detach().numpy().flatten()[0]

    # Fine-tuned LLM predict
    series_str = ", ".join(f"{x:.2f}" for x in input_series[-10:])
    prompt = f"Given the previous sensor readings: [{series_str}], predict the next value:"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    output = llm_model.generate(**inputs, max_length=inputs['input_ids'].shape[1]+10, do_sample=False)
    generated = llm_tokenizer.decode(output[0])
    
    try:
        prediction_text = generated[len(prompt):].strip().split()[0]
        llm_pred = float(prediction_text.replace(",", "").replace("[", "").replace("]", ""))
    except:
        llm_pred = input_series[-1]

    # Ensemble prediction
    final_prediction = (lstm_pred + patchtst_pred + llm_pred) / 3

    return {
        "LSTM_Prediction": float(lstm_pred),
        "PatchTST_Prediction": float(patchtst_pred),
        "LLM_Prediction": float(llm_pred),
        "Final_Ensembled_Prediction": float(final_prediction)
    }

# 🚀 FastAPI App
app = FastAPI()

class SensorInput(BaseModel):
    sensor_values: list  # expecting a list of last 96 values

@app.post("/predict")
async def forecast(input_data: SensorInput, request: Request):
    client_host = request.client.host
    print(f"Received a request from {client_host} with data: {input_data.sensor_values}")
    preds = predict(input_data.sensor_values)
    return preds

# 📢 Run server
if __name__ == "__main__":
    print("Server running on http://0.0.0.0:8000")
    uvicorn.run("iot_forecast_server:app", host="0.0.0.0", port=8000, reload=True)