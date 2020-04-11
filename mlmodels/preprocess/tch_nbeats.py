import numpy as np

def preprocess(df, backcast_length, forecast_length):
    df = df.values  # just keep np array here for simplicity.
    norm_constant = np.max(df)
    df = df / norm_constant
    
    x_train_batch, y = [], []
    for i in range(backcast_length, len(df) - forecast_length):
        x_train_batch.append(df[i - backcast_length:i])
        y.append(df[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    y = np.array(y)[..., 0]
    return x_train_batch,y