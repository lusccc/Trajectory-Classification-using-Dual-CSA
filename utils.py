from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np



def scale_1d_data(data, scaler=StandardScaler()):

    data = np.array(data)
    shape_ = data.shape
    data = data.reshape((-1, 1))
    # scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = np.reshape(data, shape_)
    return data

def scale_data(data, scaler=StandardScaler()):

    data = np.array(data)
    data = scaler.fit_transform(data)
    return data

