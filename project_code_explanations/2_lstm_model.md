# IoT Time Series Forecasting

This repository contains notebooks and scripts for preprocessing, analyzing, and forecasting IoT time series data. Below is an explanation of the steps in the `2_lstm_model.ipynb` notebook.

## 2. LSTM Model for IoT Time Series Forecasting

### Step 0: Import Required Libraries
- **Purpose**: Import necessary libraries for building, training, and evaluating the LSTM model.
- **Details**:
  - Libraries like `torch`, `pandas`, `numpy`, and `matplotlib` are imported.
  - The `LSTMForecast` model is imported from the `models` directory.
  - GPU support is enabled using `torch.device`.

### Step 1: Load the Preprocessed Dataset
- **Purpose**: Load the preprocessed time series data for training and evaluation.
- **Details**:
  - The dataset is read from the `data/processed/etth1_processed.csv` file.
  - The `OT` (outside temperature) column is selected as the target variable.

### Step 2: Create PyTorch Dataset Class
- **Purpose**: Define a custom PyTorch dataset class for time series data.
- **Details**:
  - The `TimeSeriesDataset` class is implemented to generate input-output pairs using a sliding window approach.
  - The `__getitem__` method returns a window of input data (`x`) and the corresponding target value (`y`).

### Step 3: Prepare DataLoader
- **Purpose**: Create a PyTorch DataLoader for batching and shuffling the dataset.
- **Details**:
  - A sliding window size of 48 and a batch size of 32 are used.
  - The `DataLoader` is initialized with the `TimeSeriesDataset`.

### Step 4: Train the Model
- **Purpose**: Train the LSTM model using the training data.
- **Details**:
  - The `LSTMForecast` model is instantiated and moved to the GPU (if available).
  - The model is trained for 10 epochs using the Mean Squared Error (MSE) loss function and the Adam optimizer.
  - The training loss is printed for each epoch.

### Step 5: Save the Trained Model
- **Purpose**: Save the trained model weights for future use.
- **Details**:
  - The model's state dictionary is saved to the `models/checkpoints/lstm_model.pth` file.

### Step 6: Plot a Few Predictions
- **Purpose**: Visualize the model's predictions against the ground truth.
- **Details**:
  - A batch of test data is passed through the model to generate predictions.
  - The predictions and ground truth values are plotted for the first 50 samples.
  - The plot is saved to the `outputs/lstm_model_prediction_plot.png` file.

### Step 7: Evaluate the Model
- **Purpose**: Evaluate the model's performance on the entire dataset.
- **Details**:
  - The model's predictions are compared to the ground truth values using the Mean Squared Error (MSE) metric.
  - The MSE is printed to the console.

### Step 8: Plot Predictions vs True Values
- **Purpose**: Plot the model's predictions against the true values for the first 100 samples.
- **Details**:
  - The predictions and true values are plotted to visually assess the model's performance.

## Directory Structure
- `notebooks/`: Contains Jupyter notebooks for data preprocessing and modeling.
- `data/`: Contains raw and processed datasets.
- `models/`: Contains the LSTM model implementation and checkpoints.
- `outputs/`: Contains plots and other outputs generated during training and evaluation.

## Requirements
- Python 3.13+
- Libraries: `torch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## Usage
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the notebooks in the `notebooks/` directory.
