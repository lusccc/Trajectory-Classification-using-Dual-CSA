import math

from numba import jit
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from params import SCALER, MAX_SEGMENT_SIZE, MIN_N_POINTS


def scale_1d_data(data, scaler=SCALER):

    data = np.array(data)
    shape_ = data.shape
    data = data.reshape((-1, 1))
    # scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = np.reshape(data, shape_)
    return data

def scale_data(data, scaler=SCALER):

    data = np.array(data)
    data = scaler.fit_transform(data)
    return data


@jit(nopython=True)
def hampel_filter_forloop_numba(input_series, window_size=10, n_sigmas=3):
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []

    for i in range((window_size), (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)

    return new_series

def check_lat_lng(p):
    lat = p[0]
    lng = p[1]
    if lat < -90 or lat > 90:
        print('invalid lat:{}'.format(p))
        return False
    if lng < -180 or lng > 180:
        print('invalid lng:{}'.format(p))
        return False
    return True


def segment_single_series(series, max_size=MAX_SEGMENT_SIZE):
    size = len(series)
    if size <= max_size:
        return np.array([series])
    else:
        segments = []
        index = 0
        size_of_rest_series = size
        while size_of_rest_series > max_size:
            seg = series[index:index + max_size]  # [,)
            segments.append(seg)
            size_of_rest_series -= max_size
            index += max_size
        if size_of_rest_series > MIN_N_POINTS:
            rest_series = series[index:size]
            segments.append(rest_series)
        return np.array(segments)

def calc_initial_compass_bearing(pointA, pointB):
    # https://gist.github.com/jeromer/2005586
    # https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    """
        Calculates the bearing between two points.
        The formulae used is the following:
            θ = atan2(sin(Δlong).cos(lat2),
                      cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
        :Parameters:
          - `pointA: The tuple representing the latitude/longitude for the
            first point. Latitude and longitude must be in decimal degrees
          - `pointB: The tuple representing the latitude/longitude for the
            second point. Latitude and longitude must be in decimal degrees
        :Returns:
          The bearing in degrees
          direction heading in degrees (0-360 degrees, with 90 = North)
        :Returns Type:
          float
        """
    # if (type(pointA) != tuple) or (type(pointB) != tuple):
    #     raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def interp_single_series(series, target_size=MAX_SEGMENT_SIZE):
    # https://www.yiibai.com/scipy/scipy_interpolate.html
    size = len(series)
    if size == target_size:
        return series
    else:
        y = np.array(series)
        x = np.arange(size)
        # extend to target size
        interp_x = np.linspace(0, x.max(), target_size)
        interp_y = interp1d(x, y, kind='linear')(interp_x)
        return interp_y

def padzeros(series, target_size=MAX_SEGMENT_SIZE):
    new_series = np.zeros(target_size)
    new_series[:len(series)] = series
    return new_series





