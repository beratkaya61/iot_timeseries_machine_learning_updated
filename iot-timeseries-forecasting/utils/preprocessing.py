import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.interpolate(method='linear', inplace=True)
    return df

def normalize(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler
