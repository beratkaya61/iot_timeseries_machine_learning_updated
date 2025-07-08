# IoT Time Series Real-Time Streaming and Prediction

This project demonstrates real-time streaming and prediction for IoT time series data using a rolling window approach and a FastAPI server for predictions.

## Features

- **Real-Time Data Simulation**: Simulates real-time data streaming from the ETTh2 dataset.
- **Prediction**: Sends rolling window data to a FastAPI server for predictions.
- **Anomaly Detection**: Detects anomalies based on the difference between real and predicted values.
- **Predictive Maintenance**: Triggers maintenance alerts based on prediction thresholds.
- **Live Plotting**: Visualizes real-time predictions, anomalies, and maintenance warnings.
- **Result Storage**: Saves results to an Excel file for further analysis.

## How to Run

1. **Install Dependencies**:
   Ensure you have the required Python libraries installed:
   ```bash
   pip install pandas numpy matplotlib requests openpyxl
   ```

2. **Start the FastAPI Server**:
   Run the FastAPI server that provides predictions:
   ```bash
   uvicorn main:app --reload
   ```

3. **Run the Notebook**:
   Open the `test_stream_real_time.ipynb` file in Jupyter Notebook and execute the cells.

4. **View Results**:
   - Real-time plots will be displayed during execution.
   - Results will be saved to `../data/results/prediction_results.xlsx`.

## File Structure

- `6_test_stream_real_time.md`: Documentation for the real-time streaming test.
- `test_stream_real_time.ipynb`: Jupyter Notebook implementing the real-time streaming and prediction.

## Notes

- Ensure the FastAPI server is running before executing the notebook.
- Adjust the file paths and server URL as needed for your environment.

## Example Output

- **Live Plot**:
  ![Live Plot Example](example_plot.png)

- **Excel Output**:
  The results are saved in `prediction_results.xlsx` with columns for time, real OT, predicted OT, anomalies, and maintenance alerts.