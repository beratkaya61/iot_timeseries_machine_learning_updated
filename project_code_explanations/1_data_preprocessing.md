# IoT Time Series Forecasting

This repository contains notebooks and scripts for preprocessing, analyzing, and forecasting IoT time series data. Below is an explanation of the steps in the `1_data_preprocessing.ipynb` notebook.

## 1. Data Preprocessing for IoT Time Series Forecasting

### Step 1: Load Dataset from GitHub
- **Purpose**: Load the IoT sensor dataset (`ETTh1.csv`) from a GitHub repository.
- **Details**:
  - Libraries like `pandas`, `numpy`, `matplotlib`, and `seaborn` are imported for data manipulation and visualization.
  - The dataset is fetched using the `requests` library with SSL verification disabled.
  - The CSV data is read into a Pandas DataFrame, and the `date` column is set as the index.

### Step 2: Explore the Dataset
- **Purpose**: Perform basic exploration to understand the dataset.
- **Details**:
  - The shape of the dataset and descriptive statistics are printed.
  - A line plot of all columns is created to visualize the sensor data over time.

### Step 3: Handle Missing Values (Linear Interpolation)
- **Purpose**: Fill missing values in the dataset using linear interpolation.
- **Details**:
  - The number of missing values before and after interpolation is calculated.
  - Missing values are filled using the `interpolate` method with the `linear` option.

### Step 4: Normalize the Data
- **Purpose**: Scale the data to a range of [0, 1] for better performance in machine learning models.
- **Details**:
  - The `MinMaxScaler` from `sklearn.preprocessing` is used to normalize the data.
  - The scaled data is converted back into a Pandas DataFrame with the same index and column names.

### Step 5: Simple Anomaly Detection using Z-score
- **Purpose**: Detect anomalies in the data using the Z-score method.
- **Details**:
  - The Z-scores of the normalized data are calculated using `scipy.stats.zscore`.
  - Anomalies are identified where the absolute Z-score exceeds 3.
  - Anomalies in the `OT` column are visualized on a plot with red markers.

### Step 6: Save Cleaned and Normalized Data
- **Purpose**: Save the preprocessed data for further modeling.
- **Details**:
  - The normalized DataFrame is saved as a CSV file in the `data/processed` directory.

### Step 7: Display Raw and Processed Data
- **Purpose**: Compare the raw and processed data.
- **Details**:
  - The first few rows of the raw and processed data are printed.
  - A summary of the changes (normalization and missing value handling) is displayed.

## Directory Structure
- `notebooks/`: Contains Jupyter notebooks for data preprocessing and analysis.
- `data/`: Contains raw and processed datasets.
- `scripts/`: Contains Python scripts for automation and modeling.

## Requirements
- Python 3.13+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `requests`

## Usage
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the notebooks in the `notebooks/` directory.
