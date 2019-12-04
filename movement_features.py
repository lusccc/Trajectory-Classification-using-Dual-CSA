import math

import numpy as np
from geopy.distance import geodesic
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

MAX_SEGMENT_SIZE = 48
MIN_N_POINTS = 5


def calc_trjs_movement_features():
    i = 0
    for trj, label in zip(trjs, labels):
        print('{}/{}'.format(i, n))
        if len(trj) < MIN_N_POINTS:
            print('small size:{}'.format(len(trj)))
            continue
        i += 1
        calc_single_trj_movement_features(trj, label)


def calc_single_trj_movement_features(trj, label):
    trj = np.array(trj)
    n_points = len(trj)
    distances = []
    velocities = []
    heading = []
    for i in range(n_points - 1):
        p_a = (trj[i][1], trj[i][2])
        p_b = (trj[i + 1][1], trj[i + 1][2])
        t_a = trj[i][0]
        t_b = trj[i + 1][0]
        delta_t = t_b - t_a
        if delta_t <= 0:
            print('invalid timestamp, t_a:{}, tb:{}, delta_t:{}'.format(t_a, t_b, delta_t))
            continue
        if not check_lat_lng(p_a) or not check_lat_lng(p_b):
            continue
        # distance
        d = geodesic(p_a, p_b).meters
        distances.append(d)
        # velocity
        v = d / delta_t
        velocities.append(v)
        # heading
        h = calc_initial_compass_bearing(p_a, p_b)
        heading.append(h)

    distances = np.array(distances)
    velocities = np.array(velocities)
    heading = np.array(heading)

    d_segs = segment_single_series(distances)
    v_segs = segment_single_series(velocities)
    h_segs = segment_single_series(heading)

    d_segs = interp_multiple_series(d_segs)
    v_segs = interp_multiple_series(v_segs)
    h_segs = interp_multiple_series(h_segs)
    for d_seg, v_seg, h_seg in zip(d_segs, v_segs, h_segs):
        d_seg = savitzky_golay(d_seg, 9, 3)
        v_seg = savitzky_golay(v_seg, 9, 3)
        h_seg = savitzky_golay(h_seg, 9, 3)
        # plt.figure()
        # plt.plot(d_seg,  c='red')
        # plt.plot(d_seg_, c='green')
        # plt.show()
        features_seg = np.array(
            [[d, v, h] for d, v, h in zip(d_seg, v_seg, h_seg)]
        )
        features_seg = np.expand_dims(features_seg, axis=0)

        features_segments.append(features_seg)
        features_segments_labels.append(label)


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


def interp_multiple_series(series, target_size=MAX_SEGMENT_SIZE):
    interped = []
    for s in series:
        s = interp_single_series(s, target_size)
        interped.append(s)
    return np.array(interped)


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
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


if __name__ == '__main__':
    trjs = np.load('./geolife/trjs.npy', allow_pickle=True)[:]
    labels = np.load('./geolife/labels.npy')[:]
    n = len(trjs)
    features_segments = []
    features_segments_labels = []
    calc_trjs_movement_features()
    features_segments = np.array(features_segments)
    features_segments_labels = np.array(features_segments_labels)
    print(features_segments.shape)
    print(features_segments_labels.shape)
    np.save('./geolife/features_segments.npy', features_segments)
    np.save('./geolife/features_segments_labels.npy', features_segments_labels)
