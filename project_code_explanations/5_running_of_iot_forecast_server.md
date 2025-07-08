# IoT Time Series Forecasting Server

This repository contains a FastAPI-based server for time series forecasting using three different models: LSTM, PatchTST, and a fine-tuned language model (LLM). The server combines predictions from these models to produce an ensemble forecast.

## Features
- **LSTM Model**: Predicts the next value in the time series using a recurrent neural network.
- **PatchTST Model**: Uses a transformer-based architecture for time series forecasting.
- **Fine-Tuned LLM**: Leverages a language model to predict the next value based on recent sensor readings.
- **Ensemble Forecasting**: Combines predictions from all three models for improved accuracy.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- Uvicorn
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/iot-timeseries-forecasting.git
   cd iot-timeseries-forecasting/server
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model checkpoint files are placed in the `../models/checkpoints/` directory:
   - `lstm_model.pth`
   - `patchTST_transformer_model.pth`
   - Fine-tuned LLM model in `../models/llm_forecaster/`.

## Usage
1. Start the server:
   ```bash
   python iot_forecast_server.py
   ```

2. Send a POST request to the `/predict` endpoint:
   ```bash
   curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d '{"sensor_values": [1.0, 2.0, ..., 96.0]}'
   ```

3. Example response:
   ```json
   {
       "LSTM_Prediction": 1.23,
       "PatchTST_Prediction": 1.45,
       "LLM_Prediction": 1.34,
       "Final_Ensembled_Prediction": 1.34
   }
   ```

## File Structure
```
iot-timeseries-forecasting/
├── server/
│   ├── iot_forecast_server.py  # Main server script
│   ├── requirements.txt        # Dependencies
├── models/
│   ├── checkpoints/            # Model checkpoint files
│   ├── llm_forecaster/         # Fine-tuned LLM model
```

## Notes
- Ensure the input series contains at least 96 values.
- Modify the paths in the script to point to your model checkpoint files if necessary.

## License
This project is licensed under the MIT License.