import math
from math import acos

import numpy as np
from geopy.distance import vincenty, geodesic
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

MAX_SEGMENT_SIZE = 48
MIN_N_POINTS = 5


def calc_single_trj_movement_features(trj, label):
    trj = np.array(trj)
    n_points = len(trj)
    distances = []
    velocities = []
    heading = []
    for i in range(n_points-1):
        p_a = (trj[i][1], trj[i][2])
        p_b = (trj[i+1][1], trj[i+1][2])
        t_a = trj[i][0]
        t_b = trj[i+1][0]
        delta_t = t_b - t_a
        if not check_lat_lng(p_a) or not check_lat_lng(p_b):
            continue
        # distance
        d = geodesic(p_a, p_b).meters
        distances.append(d)
        # velocity
        v = d/delta_t
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

        features_seg = np.array([d_seg, v_seg, h_seg])
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
        print ('invalid lat:{}'.format(p))
        return False
    if lng < -180 or lng > 180:
        print ('invalid lng:{}'.format(p))
        return False
    return True


def calc_initial_compass_bearing(pointA, pointB):
    #https://gist.github.com/jeromer/2005586
    #https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
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


if __name__ == '__main__':
    trjs = np.load('./geolife/trjs.npy', allow_pickle=True)
    labels = np.load('./geolife/labels.npy')
    n = len(trjs)
    features_segments = []
    features_segments_labels = []
    i = 0
    for trj, label in zip(trjs, labels):
        if len(trj) < MIN_N_POINTS:
            print('small size:{}'.format(len(trj)))
            continue
        print('{}/{}'.format(i, n))
        i += 1
        calc_single_trj_movement_features(trj, label)
    features_segments = np.array(features_segments)
    features_segments_labels = np.array(features_segments_labels)
    np.save('./geolife/features_segments.npy', features_segments)
    np.save('./geolife/features_segments_labels.npy', features_segments_labels)

