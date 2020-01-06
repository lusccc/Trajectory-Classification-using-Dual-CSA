import numpy as np
from geopy.distance import geodesic
from sklearn.utils import shuffle

from params import MIN_N_POINTS
from backup.Time_utils import timestamp_to_hour
from trajectory_extraction import MODE_NAMES
from utils import segment_single_series, check_lat_lng, calc_initial_compass_bearing, interp_single_series

SPEED_LIMIT = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6, 5: 120 / 3.6}
# acceleration
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3, 5: 3}
# heading change rate limit
HCR_LIMIT = {0: 30, 1: 50, 2: 60, 3: 90, 4: 20}
STOP_DISTANCE_LIMIT = 2  # meters
STOP_VELOCITY_LIMIT = 2
STRAIGHT_MOVING_DEGREE_LIMIT = 30  # abs value, less than this limit mean still straight

# 0        1     2  3  4  5  6   7    8  9
# delta_t, hour, d, v, a, h, hc, hcr, s, tn
movement_features = [3, 4, 8, 9]
other_features = [0, 2, 1, 8, 9]


def segment_trjs(trjs, labels):
    trjs_segs = []
    trjs_segs_labels = []
    for trj, label in zip(trjs, labels):
        trj_segs = segment_single_series(trj)
        trj_segs_labels = [label for _ in range(len(trj_segs))]
        trjs_segs.extend(trj_segs)
        trjs_segs_labels.extend(trj_segs_labels)
    return np.array(trjs_segs), np.array(trjs_segs_labels)


def filter_trjs_segs_gps_data(trjs_segs, trjs_segs_labels):
    new_trjs_segs = []
    new_trjs_segs_labels = []
    for trj_seg, trj_seg_label in zip(trjs_segs, trjs_segs_labels):
        n_points = len(trj_seg)
        if n_points < MIN_N_POINTS:
            print('gps points num not enough:{}'.format(n_points))
            continue
        invalid_points = []  # wrong gps data points index
        for i in range(n_points - 1):
            p_a = [trj_seg[i][1], trj_seg[i][2]]
            p_b = [trj_seg[i + 1][1], trj_seg[i + 1][2]]
            t_a = trj_seg[i][0]
            t_b = trj_seg[i + 1][0]

            # if "point a" is invalid, using previous "point a" instead of current one
            if i in invalid_points:
                p_a = [trj_seg[i - 1][1], trj_seg[i - 1][2]]
                t_a = trj_seg[i - 1][0]
                delta_t = t_b - t_a
            else:
                delta_t = t_b - t_a
            if delta_t <= 0:
                invalid_points.append(i + 1)
                print('invalid timestamp, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
                continue
            if not check_lat_lng(p_a):
                invalid_points.append(i)
                continue
            if not check_lat_lng(p_b):
                invalid_points.append(i + 1)
                continue
        new_trj_seg = np.delete(trj_seg, invalid_points, axis=0)
        if len(new_trj_seg) < MIN_N_POINTS:
            print('gps points num not enough:{}'.format(len(new_trj_seg)))
        else:
            new_trjs_segs.append(new_trj_seg)
            new_trjs_segs_labels.append(trj_seg_label)
    return np.array(new_trjs_segs), np.array(new_trjs_segs_labels)


def calc_trjs_segs_features_interp(trjs_segs, trjs_segs_labels):
    trjs_segs_features = []
    for trj_seg, trj_seg_label in zip(trjs_segs, trjs_segs_labels):
        n_points = len(trj_seg)
        hours = []  # hour of timestamp
        delta_times = []
        distances = []
        velocities = []
        accelerations = [0]  # init acceleration is 0
        headings = []
        heading_changes = []
        heading_change_rates = []
        stops = [0]  # is stop,0~1, 0:not stop, 1:stop,  we define first point is moving
        turnings = [0]  # is turning,0~1, 0:not turning, 1:turning,  we define first point is not turning

        prev_v = 0  # previous velocity
        prev_h = 0  # previous heading
        for i in range(n_points - 1):
            p_a = [trj_seg[i][1], trj_seg[i][2]]
            p_b = [trj_seg[i + 1][1], trj_seg[i + 1][2]]
            t_a = trj_seg[i][0]
            t_b = trj_seg[i + 1][0]

            delta_t = t_b - t_a
            hour = timestamp_to_hour(t_a)
            # distance
            d = geodesic(p_a, p_b).meters
            # velocity
            v = d / delta_t
            # accelerations
            a = (v - prev_v) / delta_t
            # heading
            h = calc_initial_compass_bearing(p_a, p_b)
            # heading change
            hc = h - prev_h
            # heading change rate
            hcr = hc / delta_t
            # is stop point
            s = 1 if d < STOP_DISTANCE_LIMIT else 0  # 1-(d/STOP_DISTANCE_LIMIT)
            # is turning point
            tn = 0 if abs(hc) < STRAIGHT_MOVING_DEGREE_LIMIT else 1  # 1-(abs(hc)/STRAIGHT_MOVING_DEGREE_LIMIT)

            if v > SPEED_LIMIT[trj_seg_label]:  # ?? or v == 0
                print('invalid speed:{} for {}'.format(v, MODE_NAMES[trj_seg_label]))
                continue
            if a > ACC_LIMIT[trj_seg_label]:
                print('invalid acc:{} for {}'.format(a, MODE_NAMES[trj_seg_label]))
                continue
            if hcr > HCR_LIMIT[trj_seg_label]:  # ?? or hcr == 0
                print('invalid hcr:{} for {}'.format(hcr, MODE_NAMES[trj_seg_label]))
                continue

            delta_times.append(delta_t)
            hours.append(hour)
            distances.append(d)
            velocities.append(v)
            accelerations.append(a)
            headings.append(h)
            heading_changes.append(hc)
            heading_change_rates.append(hcr)
            stops.append(s)
            turnings.append(tn)

            prev_v = v
            prev_h = h

        trj_seg_features = np.array(
            [[delta_t, hour, d, v, a, h, hc, hcr, s, tn] for delta_t, hour, d, v, a, h, hc, hcr, s, tn in
             zip(fill_series_function(delta_times),
                 fill_series_function(hours),
                 fill_series_function(distances),
                 fill_series_function(velocities),
                 fill_series_function(accelerations),
                 fill_series_function(headings),
                 fill_series_function(heading_changes),
                 fill_series_function(heading_change_rates),
                 fill_series_function(stops),
                 fill_series_function(turnings))]
        )
        trj_seg_features = np.expand_dims(trj_seg_features, axis=0)
        trjs_segs_features.append(trj_seg_features)
    return np.array(trjs_segs_features), trjs_segs_labels


def filter_trjs_segs_features_interp(trjs_segs_features, trjs_segs_labels):
    new_trjs_segs_features = []
    new_trjs_segs_labels = []
    filtered_trjs_segs_features_idx = []  # idx of segs of features whose num is not enough
    i = 0
    for trj_seg_features, trj_seg_label in zip(trjs_segs_features, trjs_segs_labels):
        invalid_values = set()  # invalid feature value index
        trj_seg_features = np.squeeze(trj_seg_features)  # for save data,  has expanded dim, now squeeze back
        j = 0
        for delta_t, hour, d, v, a, h, hc, hcr, s, tn in trj_seg_features:
            if v > SPEED_LIMIT[trj_seg_label]:  # ?? or v == 0
                invalid_values.add(j)
                print('invalid speed:{} for {}'.format(v, MODE_NAMES[trj_seg_label]))
            if a > ACC_LIMIT[trj_seg_label]:
                invalid_values.add(j)
                print('invalid acc:{} for {}'.format(a, MODE_NAMES[trj_seg_label]))
            if hcr > HCR_LIMIT[trj_seg_label]:  # ?? or hcr == 0
                invalid_values.add(j)
                print('invalid hcr:{} for {}'.format(hcr, MODE_NAMES[trj_seg_label]))
            j += 1

        new_trj_seg_features = np.delete(trj_seg_features, list(invalid_values), axis=0)
        if len(new_trj_seg_features) < MIN_N_POINTS:
            print('feature element num not enough:{}'.format(len(new_trj_seg_features)))
            filtered_trjs_segs_features_idx.append(i)
        else:
            n_features = new_trj_seg_features.shape[1]
            new_trj_seg_features_interped = np.array(
                [fill_series_function(new_trj_seg_features[:, i]) for i in range(n_features)]
            ).T
            new_trj_seg_features_interped = np.expand_dims(new_trj_seg_features_interped, axis=0)
            new_trjs_segs_features.append(new_trj_seg_features_interped)
            new_trjs_segs_labels.append(trj_seg_label)

        i += 1

    return np.array(new_trjs_segs_features), np.array(new_trjs_segs_labels), filtered_trjs_segs_features_idx


if __name__ == '__main__':
    trjs = np.load('./geolife/trjs.npy', allow_pickle=True)
    labels = np.load('./geolife/labels.npy')
    trjs, labels = shuffle(trjs, labels, random_state=0)  # !!!shuffle

    n_test = 1000
    trjs = trjs[:n_test]
    labels = labels[:n_test]
    
    fill_series_function = interp_single_series

    trjs_segs, trjs_segs_labels = segment_trjs(trjs, labels)
    trjs_segs, trjs_segs_labels = filter_trjs_segs_gps_data(trjs_segs, trjs_segs_labels)

    trjs_segs_features_ori, trjs_segs_labels_ori = calc_trjs_segs_features_interp(trjs_segs, trjs_segs_labels)
    trjs_segs_features_filtered, trjs_segs_labels_filtered, filtered_trjs_segs_features_idx = filter_trjs_segs_features_interp(
        trjs_segs_features_ori,
        trjs_segs_labels_ori)
    # make trjs_segs_features_ori have the same length as trjs_segs_features_filtered, because in trjs_segs_features_filtered,
    # segs whose features element num  is less than MIN_N_POINTS has deleted
    trjs_segs_features_ori = np.delete(trjs_segs_features_ori, filtered_trjs_segs_features_idx, axis=0)
    trjs_segs_labels_ori = np.delete(trjs_segs_labels_ori, filtered_trjs_segs_features_idx, axis=0)

    # scaler = MinMaxScaler()
    # trjs_segs_features_ori = scale_segs(trjs_segs_features_ori, scaler)
    # trjs_segs_features_filtered = scale_segs(trjs_segs_features_filtered, scaler)

    np.save('./geolife_features/trjs_segs_features_ori.npy', trjs_segs_features_ori)
    np.save('./geolife_features/trjs_segs_features_filtered.npy', trjs_segs_features_filtered)
    if np.array_equal(trjs_segs_labels_ori, trjs_segs_labels_filtered):
        print('equal label')
        np.save('./geolife_features/labels.npy', trjs_segs_labels_ori)
    else:
        print('not equal label')

    print('done')
