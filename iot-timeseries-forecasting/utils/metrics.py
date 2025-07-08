from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    smape = 100 * np.mean(2 * np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
    return {"MSE": mse, "MAE": mae, "SMAPE": smape}
