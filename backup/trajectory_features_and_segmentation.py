import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from params import MIN_N_POINTS, MAX_SEGMENT_SIZE, SCALE_SEGS_EACH_FEATURE, FILTER_SEGS
from backup.Time_utils import timestamp_to_hour
from trajectory_extraction import MODE_NAMES
# walk, bike, bus, driving, train/subway
# modes_to_use = [0,1,2,3,4]
from utils import scale_data, hampel_filter_forloop_numba, segment_single_series, calc_initial_compass_bearing, \
    interp_single_series

DISTANCE_LIMIT = 500
TIME_INTERVAL_LIMIT = 300  # delta time between two consecutive points
SPEED_LIMIT = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6, 5: 120 / 3.6}
# acceleration
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3, 5: 3}
# heading change rate limit
HCR_LIMIT = {0: 30, 1: 50, 2: 60, 3: 90, 4: 20}
STOP_DISTANCE_LIMIT = 2  # meters
STOP_VELOCITY_LIMIT = 2
STRAIGHT_MOVING_DEGREE_LIMIT = 30  # abs value, less than this limit mean still straight




def calc_trjs_features():
    i = 0
    for trj, label in zip(trjs, labels):
        print('{}/{}'.format(i, n))
        i += 1
        if len(trj) < MIN_N_POINTS:
            print('gps points num not enough:{}'.format(len(trj)))
            continue

        calc_single_trj_features(trj, label)


def calc_single_trj_features(trj, label):
    trj = np.array(trj)
    n_points = len(trj)

    invalid_points = []  # noise points index
    # below 3 spatial features
    distances = []
    headings = []
    heading_changes = []
    # below 3 movement features
    velocities = []
    heading_change_rates = []
    accelerations = [0]  # init acceleration

    # is stop,0~1, 0:not stop, 1:stop,  we define first point is moving
    stops = [0]
    # is turning,0~1, 0:not turning, 1:turning,  we define first point is not turning
    turnings = [0]
    # hour time interval 0~24
    hours = []
    delta_times = []

    prev_v = 0  # previous velocity
    prev_h = 0  # previous heading
    for i in range(n_points - 1):

        global N_TOTAL_POINTS
        N_TOTAL_POINTS += 1

        p_a = [trj[i][1], trj[i][2]]
        p_b = [trj[i + 1][1], trj[i + 1][2]]
        t_a = trj[i][0]
        t_b = trj[i + 1][0]
        hour = timestamp_to_hour(t_a)

        # if "point a" is invalid, using previous "point a" instead of current one
        if i in invalid_points:
            p_a = [trj[i - 1][1], trj[i - 1][2]]
            t_a = trj[i - 1][0]
            delta_t = t_b - t_a
        else:
            delta_t = t_b - t_a
        if delta_t <= 0:
            invalid_points.append(i + 1)
            print('invalid timestamp, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
            continue
        # # TODO ????
        # if delta_t >= TIME_INTERVAL_LIMIT:
        #     invalid_points.append(i + 1)
        #     print('big time interval, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
        #     continue
        if not check_lat_lng(p_a):
            invalid_points.append(i)
            continue
        if not check_lat_lng(p_b):
            invalid_points.append(i + 1)
            continue

        # distance
        d = geodesic(p_a, p_b).meters
        # # TODO ?????
        # if d > DISTANCE_LIMIT or d < 0:
        #     invalid_points.append(i + 1)
        #     print('invalid distance:{} for {}'.format(d, MODE_NAMES[label]))
        #     continue
        # velocity
        v = d / delta_t
        if v > SPEED_LIMIT[label]:  # ?? or v == 0
            invalid_points.append(i + 1)
            print('invalid speed:{} for {}'.format(v, MODE_NAMES[label]))
            continue
        # accelerations
        a = (v - prev_v) / delta_t
        if a > ACC_LIMIT[label]:
            invalid_points.append(i + 1)
            print('invalid acc:{} for {}'.format(a, MODE_NAMES[label]))
            continue

        # heading
        h = calc_initial_compass_bearing(p_a, p_b)
        # heading change
        hc = h - prev_h
        # heading change rate
        hcr = hc / delta_t
        if hcr > HCR_LIMIT[label]:  # ?? or hcr == 0
            invalid_points.append(i + 1)
            print('invalid hcr:{} for {}'.format(hcr, MODE_NAMES[label]))
            continue
        # is stop point
        s = 1 if d < STOP_DISTANCE_LIMIT else 0  # 1-(d/STOP_DISTANCE_LIMIT)
        # is turning point
        tn = 0 if abs(hc) < STRAIGHT_MOVING_DEGREE_LIMIT else 1  # 1-(abs(hc)/STRAIGHT_MOVING_DEGREE_LIMIT)

        distances.append(d)
        velocities.append(v)
        accelerations.append(a)
        headings.append(h)
        heading_changes.append(hc)
        heading_change_rates.append(hcr)
        stops.append(s)
        turnings.append(tn)
        hours.append(hour)
        delta_times.append(delta_t)

        prev_v = v
        prev_h = h

        global N_RESERVED_POINTS
        N_RESERVED_POINTS += 1

    if len(distances) < MIN_N_POINTS:
        print('feature element num not enough:{}'.format(len(distances)))
        return

    trj_filtered = np.delete(trj, invalid_points, axis=0)  # delete invalid points(rows)
    distances = np.array(distances)
    headings = np.array(headings)
    heading_changes = np.array(heading_changes)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    heading_change_rates = np.array(heading_change_rates)
    stops = np.array(stops)
    turnings = np.array(turnings)
    hours = np.array(hours)
    delta_times = np.array(delta_times)

    trj_segs = segment_single_series(trj_filtered)
    d_segs = segment_single_series(distances)
    h_segs = segment_single_series(headings)
    hc_segs = segment_single_series(heading_changes)
    v_segs = segment_single_series(velocities)
    a_segs = segment_single_series(accelerations)
    hcr_segs = segment_single_series(heading_change_rates)
    s_segs = segment_single_series(stops)
    tn_segs = segment_single_series(turnings)
    hs_segs = segment_single_series(hours)
    dt_segs = segment_single_series(delta_times)

    # no need to interp trjs
    d_segs = interp_multiple_series(d_segs)
    h_segs = interp_multiple_series(h_segs)
    hc_segs = interp_multiple_series(hc_segs)
    v_segs = interp_multiple_series(v_segs)
    a_segs = interp_multiple_series(a_segs)
    hcr_segs = interp_multiple_series(hcr_segs)
    s_segs = interp_multiple_series(s_segs)
    tn_segs = interp_multiple_series(tn_segs)
    hs_segs = interp_multiple_series(hs_segs)
    dt_segs = interp_multiple_series(dt_segs)

    for trj_seg, d_seg, h_seg, hc_seg, v_seg, a_seg, hcr_seg, s_seg, tn_seg, hs_seg, dt_seg in \
            zip(trj_segs, d_segs, h_segs, hc_segs, v_segs, a_segs, hcr_segs, s_segs, tn_segs, hs_segs, dt_segs):
        trjs_segments.append(trj_seg)

        # movement features seg
        m_features_seg = np.array(
            [[v, a, hcr] for v, a, hcr in
             zip(v_seg, a_seg, hcr_seg)]
        )
        m_features_seg = np.expand_dims(m_features_seg, axis=0)
        m_features_segments.append(m_features_seg)

        # spatial features seg
        s_features_seg = np.array(
            [[d, h, hc, s, tn, hs, dt] for d, h, hc, s, tn, hs, dt in
             zip(d_seg, h_seg, hc_seg, s_seg, tn_seg, hs_seg, dt_seg)]
        )
        # ???? for time series, no need to expand dim
        s_features_seg = np.expand_dims(s_features_seg, axis=0)
        s_features_segments.append(s_features_seg)

        segments_labels.append(label)



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



def interp_multiple_series(series, target_size=MAX_SEGMENT_SIZE, filter=FILTER_SEGS):
    interped = []
    for s in series:
        s = interp_single_series(s, target_size)
        if filter:
            s = hampel_filter_forloop_numba(s)
        interped.append(s)
    return np.array(interped)


def scale_segs(segs, scaler):
    """
    scale each feature (each column)
    """
    ori_shape = segs.shape
    n_features = ori_shape[3]
    total_points = ori_shape[0] * ori_shape[2]
    all = segs.reshape((total_points, n_features))
    all = scale_data(all, scaler=scaler)
    scaled_segs = all.reshape(ori_shape)
    return scaled_segs


if __name__ == '__main__':
    trjs = np.load('../geolife/trjs.npy', allow_pickle=True)
    labels = np.load('../geolife/labels.npy')
    trjs, labels = shuffle(trjs, labels, random_state=0)  # !!!shuffle

    n_test = 1000
    trjs = trjs[:n_test]
    labels = labels[:n_test]

    n = len(trjs)

    # filtered data segments,  below 4 array has same element num
    trjs_segments = []
    m_features_segments = []  # movement features segs
    s_features_segments = []  # spatial features segs
    segments_labels = []

    N_TOTAL_POINTS = 0
    N_RESERVED_POINTS = 0

    calc_trjs_features()

    print(N_RESERVED_POINTS)
    print(N_TOTAL_POINTS)
    print('deleted points num percentile:{}'.format(1 - N_RESERVED_POINTS / N_TOTAL_POINTS))

    trjs_segments = np.array(trjs_segments)
    m_features_segments = np.array(m_features_segments)
    s_features_segments = np.array(s_features_segments)
    segments_labels = np.array(segments_labels)

    train_trjs_segments, test_trjs_segments, \
    train_mf_segments, test_mf_segments, \
    train_sf_segments, test_sf_segments, \
    train_segments_labels, test_segments_labels = train_test_split(trjs_segments, m_features_segments,
                                                                   s_features_segments, segments_labels,
                                                                   test_size=0.20, random_state=7, shuffle=True)

    scaler = StandardScaler()
    train_sf_segments = scale_segs(train_sf_segments, scaler)
    test_sf_segments = scale_segs(test_sf_segments, scaler)

    if SCALE_SEGS_EACH_FEATURE:
        train_mf_segments, test_mf_segments = scale_segs(train_mf_segments), scale_segs(test_mf_segments)

    np.save('../geolife/train_trjs_segments.npy', train_trjs_segments)
    np.save('../geolife/test_trjs_segments.npy', test_trjs_segments)

    np.save('../geolife/train_mf_segments.npy', train_mf_segments)
    np.save('../geolife/test_mf_segments.npy', test_mf_segments)

    np.save('../geolife/train_sf_segments.npy', train_sf_segments)
    np.save('../geolife/test_sf_segments.npy', test_sf_segments)

    np.save('../geolife/train_segments_labels.npy', train_segments_labels)
    np.save('../geolife/test_segments_labels.npy', test_segments_labels)
