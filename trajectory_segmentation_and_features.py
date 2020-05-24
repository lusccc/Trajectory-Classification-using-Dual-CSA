import argparse
import multiprocessing
import time

import numpy as np
from geopy.distance import geodesic
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from params import *

from utils import timestamp_to_hour, scale_segs_each_features
from trajectory_extraction import MODE_NAMES
from utils import segment_single_series, check_lat_lng, calc_initial_compass_bearing, interp_single_series

# walk, bike, bus, driving, //or train/subway
# [0, 1, 2, 3, 4]

SPEED_LIMIT = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6, 5: 120 / 3.6}
# acceleration
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3, 5: 3}
# TODO  heading change rate limit, not sure
HCR_LIMIT = {0: 30, 1: 60, 2: 70, 3: 120, 4: 30}  # {0: 30, 1: 50, 2: 60, 3: 90, 4: 20}
# TODO  changeable !!!!!
STOP_DISTANCE_LIMIT = 3  # meters, previous is 2
STOP_VELOCITY_LIMIT = 2
STRAIGHT_MOVING_DEGREE_LIMIT = 30  # abs value, less than this limit mean still straight


def do_segment_trjs(trjs, labels):
    trjs_segs = []
    trjs_segs_labels = []
    for trj, label in zip(trjs, labels):
        trj_segs = segment_single_series(trj)
        trj_segs_labels = [label for _ in range(len(trj_segs))]
        trjs_segs.extend(trj_segs)
        trjs_segs_labels.extend(trj_segs_labels)
    return np.array(trjs_segs), np.array(trjs_segs_labels)


def segment_trjs(trjs, labels):
    tasks = []
    batch_size = int(len(trjs) / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(do_segment_trjs, (
            trjs[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])))

    res = np.array([[t.get()[0], t.get()[1]] for t in tasks])
    print(np.shape(res))
    trjs_segs = np.concatenate(res[:, 0])
    trjs_segs_labels = np.concatenate(res[:, 1])
    return trjs_segs, trjs_segs_labels


def do_filter_trjs_segs_gps_data(trjs_segs, trjs_segs_labels):
    new_trjs_segs = []
    new_trjs_segs_labels = []
    for trj_seg, trj_seg_label in zip(trjs_segs, trjs_segs_labels):
        n_points = len(trj_seg)
        if n_points < MIN_N_POINTS:
            # print('gps points num not enough:{}'.format(n_points))
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
                # print('invalid timestamp, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
                continue
            if not check_lat_lng(p_a):
                invalid_points.append(i)
                continue
            if not check_lat_lng(p_b):
                invalid_points.append(i + 1)
                continue
        new_trj_seg = np.delete(trj_seg, invalid_points, axis=0)
        if len(new_trj_seg) < MIN_N_POINTS:
            pass
            # print('gps points num not enough:{}'.format(len(new_trj_seg)))
        else:
            new_trjs_segs.append(new_trj_seg)
            new_trjs_segs_labels.append(trj_seg_label)
    return np.array(new_trjs_segs), np.array(new_trjs_segs_labels)


def filter_trjs_segs_gps_data(trjs_segs, trjs_segs_labels):
    tasks = []
    batch_size = int(len(trjs_segs) / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(do_filter_trjs_segs_gps_data,
                                      (trjs_segs[i * batch_size:(i + 1) * batch_size],
                                       trjs_segs_labels[i * batch_size:(i + 1) * batch_size])))
    res = np.array([[t.get()[0], t.get()[1]] for t in tasks])
    print(np.shape(res))
    trjs_segs = np.concatenate(res[:, 0])
    trjs_segs_labels = np.concatenate(res[:, 1])
    return trjs_segs, trjs_segs_labels


def do_calc_trjs_segs_clean_features(trjs_segs, trjs_segs_labels, fill_series_function):
    valid_trjs_segs = []

    trjs_segs_features = []
    trjs_segs_features_labels = []
    for i, (trj_seg, trj_seg_label) in enumerate(zip(trjs_segs, trjs_segs_labels)):
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
        for j in range(n_points - 1):
            p_a = [trj_seg[j][1], trj_seg[j][2]]
            p_b = [trj_seg[j + 1][1], trj_seg[j + 1][2]]
            t_a = trj_seg[j][0]
            t_b = trj_seg[j + 1][0]

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
                # print('invalid speed:{} for {}'.format(v, MODE_NAMES[trj_seg_label]))
                continue
            if abs(a) > ACC_LIMIT[trj_seg_label]:
                # print('invalid acc:{} for {}'.format(a, MODE_NAMES[trj_seg_label]))
                continue
            if abs(hcr) > HCR_LIMIT[trj_seg_label]:  # ?? or hcr == 0
                # print('invalid hcr:{} for {}'.format(hcr, MODE_NAMES[trj_seg_label]))
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

        if len(delta_times) < MIN_N_POINTS:
            # print('feature element num not enough:{}'.format(len(delta_times)))
            continue
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
        trjs_segs_features_labels.append(trj_seg_label)

        valid_trjs_segs.append(i)

    return np.array(trjs_segs_features), np.array(trjs_segs_features_labels), valid_trjs_segs


def calc_trjs_segs_clean_features(trjs_segs, trjs_segs_labels, fill_series_function):
    tasks = []
    batch_size = int(len(trjs_segs) / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(do_calc_trjs_segs_clean_features,
                                      (trjs_segs[i * batch_size:(i + 1) * batch_size],
                                       trjs_segs_labels[i * batch_size:(i + 1) * batch_size],
                                       fill_series_function)))
    res = np.array([[t.get()[0], t.get()[1]] for t in tasks])
    trjs_segs_features = np.concatenate(res[:, 0])
    trjs_segs_features_labels = np.concatenate(res[:, 1])
    return trjs_segs_features, trjs_segs_features_labels


def calc_trjs_segs_noise_features(trjs_segs, trjs_segs_labels, valid_trjs_segs):
    """
    calc features without removing noise points
    :param valid_trjs_segs: only calc segs used in the result of calc_trjs_segs_clean_features()
    """
    trjs_segs_features = []
    trjs_segs_features_labels = []
    for i, (trj_seg, trj_seg_label) in enumerate(zip(trjs_segs, trjs_segs_labels)):
        if i not in valid_trjs_segs:
            continue
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
        for j in range(n_points - 1):
            p_a = [trj_seg[j][1], trj_seg[j][2]]
            p_b = [trj_seg[j + 1][1], trj_seg[j + 1][2]]
            t_a = trj_seg[j][0]
            t_b = trj_seg[j + 1][0]

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

        if len(delta_times) < MIN_N_POINTS:
            # print('feature element num not enough:{}'.format(len(delta_times)))
            continue
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
        trjs_segs_features_labels.append(trj_seg_label)
    return np.array(trjs_segs_features), np.array(trjs_segs_features_labels)


def random_drop_points(trjs, percentage=0.1):
    new_trjs = []
    for trj in trjs:
        n = len(trj)
        n_drop = int(n * percentage)
        random_idx = np.random.choice(n, n_drop, replace=False)
        new_trj = np.delete(trj, random_idx, axis=0)
        # if len(new_trj) <= MAX_SEGMENT_SIZE:
        #     print('short seg')
        #     continue
        new_trjs.append(new_trj)
    return np.array(new_trjs)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='TRJ_SEG_FEATURE')
    parser.add_argument('--feature_set', type=str)
    parser.add_argument('--trjs_path', type=str)
    parser.add_argument('--labels_path', type=str)
    parser.add_argument('--save_file_suffix', type=str, default='train')
    # note！！！: after random drop points in trajectory,
    # the produced features series will have the different number of samples to the original features series
    parser.add_argument('--random_drop_percentage', type=float, default='0.')

    args = parser.parse_args()
    if args.feature_set is None:
        feature_set = FEATURES_SET_1
    else:
        feature_set = [int(item) for item in args.feature_set.split(',')]
    print('feature_set:{}'.format(feature_set))

    n_cpus = multiprocessing.cpu_count()
    print('n_thread:{}'.format(n_cpus))
    pool = multiprocessing.Pool(processes=n_cpus)

    trjs = np.load(args.trjs_path, allow_pickle=True)
    labels = np.load(args.labels_path, allow_pickle=True)

    if args.random_drop_percentage:
        print('random_drop_percentage:{}'.format(args.random_drop_percentage))
        trjs = random_drop_points(trjs, args.random_drop_percentage)

    fill_series_function = interp_single_series

    print('segment_trjs...')
    trjs_segs, trjs_segs_labels = segment_trjs(trjs, labels)
    print(trjs_segs.shape)
    print('filter_trjs_segs_gps_data...')
    trjs_segs, trjs_segs_labels = filter_trjs_segs_gps_data(trjs_segs, trjs_segs_labels)
    print(trjs_segs.shape)
    print('calc_trjs_segs_clean_features...')
    trjs_segs_features, trjs_segs_features_labels = calc_trjs_segs_clean_features(trjs_segs, trjs_segs_labels,
                                                                                  fill_series_function)

    end = time.time()
    print('Running time: %s Seconds' % (end - start))

    print('saving files...')
    if not os.path.exists('./data/geolife_features/'):
        os.makedirs('./data/geolife_features/')
    np.save('./data/geolife_features/trjs_segs_features_{}.npy'.format(args.save_file_suffix),
            scale_segs_each_features(trjs_segs_features[:, :, :, feature_set]))  # note scaled!
    np.save('./data/geolife_features/trjs_segs_features_labels_{}.npy'.format(args.save_file_suffix),
            to_categorical(trjs_segs_features_labels, num_classes=N_CLASS))  # labels to one-hot
