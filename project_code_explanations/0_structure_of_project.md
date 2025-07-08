full strategy plan of real-time prediction system

![alt text](../outputs/strategy_for_real_time_prediction_system.png)


🧠 Simple System Architecture

New Real-Time Sensor Input (latest 96 points)
        ↓
Predict Function
        ↓
 ┌─────────────┬─────────────┬─────────────┐
 │   LSTM      │ PatchTST     │ Fine-Tuned LLM │
 └─────────────┴─────────────┴─────────────┘
        ↓
 Predictions from All Models
        ↓
 Compare / Save / Alert / Visualize