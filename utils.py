from sklearn.preprocessing import StandardScaler
import numpy as np



def scale_data(data, scaler=StandardScaler()):

    data = np.array(data)
    shape_ = data.shape
    data = data.reshape((-1, 1))
    # scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = np.reshape(data, shape_)
    return data


