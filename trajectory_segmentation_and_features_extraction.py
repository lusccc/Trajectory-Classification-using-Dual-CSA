import argparse
import multiprocessing
import os
import time

import numpy as np
from geopy.distance import geodesic
from logzero import logger

from params import *
from utils import segment_single_series, check_lat_lng, calc_initial_compass_bearing, interp_single_seg, to_categorical
from utils import timestamp_to_hour, scale_segs_each_features

# 0,    1,    2,   3,         4
# walk, bike, bus, driving, train/subway,

# limit for different modes:

SPEED_LIMIT = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6, }
# acceleration
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3, }
#   heading change rate limit, on the basis of empirical values
# HCR_LIMIT = {0: 30, 1: 60, 2: 70, 3: 120, 4: 30}  # {0: 30, 1: 50, 2: 60, 3: 90, 4: 20}
#           walk,   bike, bus, driving, train/subway,
HCR_LIMIT = {0: 360, 1: 360, 2: 70, 3: 120, 4: 30}
#  changeable
STOP_DISTANCE_LIMIT = 3  # meters, previous is 2
STOP_VELOCITY_LIMIT = 2
STRAIGHT_MOVING_DEGREE_LIMIT = 30  # abs value, less than this limit mean still straight

MAX_STAY_TIME_INTERVAL = 300

NO_LIMIT = False
if NO_LIMIT:
    logger.info(' !not filtering values exceed limit! ')


def do_segment_trjs(trjs, labels, seg_size):
    total_trj_segs = []
    total_trj_seg_labels = []
    for trj, label in zip(trjs, labels):
        # first, split based on long stay points
        delta_ts = np.diff(trj[:, 0])
        split_idx = np.where(delta_ts > MAX_STAY_TIME_INTERVAL)
        if len(split_idx[0]) > 0:
            trj_segs = np.split(trj, split_idx[0] + 1)
            trj_segs = [seg for seg in trj_segs if seg.shape[0] > 0]
            trj_seg_labels = [label for _ in range(len(trj_segs))]
        else:
            trj_segs = [trj]
            trj_seg_labels = [label]

        # second, segment to sub_seg with max_size
        for trj_seg, trj_seg_label in zip(trj_segs, trj_seg_labels):
            trj_sub_segs = segment_single_series(trj_seg, max_size=seg_size)
            trj_sub_seg_labels = [label for _ in range(len(trj_sub_segs))]
            total_trj_segs.extend(trj_sub_segs)
            total_trj_seg_labels.extend(trj_sub_seg_labels)
    return np.array(total_trj_segs), np.array(total_trj_seg_labels)


def segment_trjs(trjs, labels, seg_size):
    logger.info('segment_trjs...')
    tasks = []
    batch_size = int(len(trjs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_segment_trjs, (
            trjs[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size], seg_size)))

    res = np.array([[t.get()[0], t.get()[1]] for t in tasks])
    trj_segs = np.concatenate(res[:, 0])
    trj_seg_labels = np.concatenate(res[:, 1])
    return trj_segs, trj_seg_labels


def do_filter_error_gps_data(trj_segs, trj_seg_labels):
    filtered_trj_segs = []
    filter_trj_seg_labels = []
    for trj_seg, trj_seg_label in zip(trj_segs, trj_seg_labels):
        n_points = len(trj_seg)
        if n_points < MIN_N_POINTS:
            # logger.info('gps points num not enough:{}'.format(n_points))
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
                # logger.info('invalid timestamp, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
                continue
            if not check_lat_lng(p_a):
                invalid_points.append(i)
                continue
            if not check_lat_lng(p_b):
                invalid_points.append(i + 1)
                continue
        filtered_trj_seg = np.delete(trj_seg, invalid_points, axis=0)
        if len(filtered_trj_seg) < MIN_N_POINTS:
            pass
            # logger.info('gps points num not enough:{}'.format(len(filtered_trj_seg)))
        else:
            filtered_trj_segs.append(filtered_trj_seg)
            filter_trj_seg_labels.append(trj_seg_label)
    return np.array(filtered_trj_segs), np.array(filter_trj_seg_labels)


def filter_error_gps_data(trj_segs, trj_seg_labels):
    logger.info('filter_error_gps_data...')
    tasks = []
    batch_size = int(len(trj_segs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_filter_error_gps_data,
                                      (trj_segs[i * batch_size:(i + 1) * batch_size],
                                       trj_seg_labels[i * batch_size:(i + 1) * batch_size])))
    res = np.array([[t.get()[0], t.get()[1]] for t in tasks])
    trj_segs = np.concatenate(res[:, 0])
    trj_seg_labels = np.concatenate(res[:, 1])
    return trj_segs, trj_seg_labels


def do_calc_trj_seg_clean_multi_features(trj_segs, trj_seg_labels, fill_seg_function, seg_size):
    """
    clean means remove noise data
    """
    valid_trj_segs = []
    multi_feature_segs = []
    multi_feature_seg_labels = []
    n_removed_points = []
    for i, (trj_seg, trj_seg_label) in enumerate(zip(trj_segs, trj_seg_labels)):
        n_points = len(trj_seg)
        hours = []  # hour of timestamp
        delta_times = []
        distances = []
        velocities = []
        accelerations = [0]  # init acceleration is 0
        headings = []
        heading_changes = []
        heading_change_rates = []
        stops = [0]  # is stop, 0~1, 0:not stop, 1:stop, we define first point is moving
        turnings = [0]  # is turning,0 ~1, 0:not turning, 1:turning, we define first point is not turning

        prev_v = 0  # previous velocity
        prev_h = 0  # previous heading
        for j in range(n_points - 1):
            p_a = [trj_seg[j][1], trj_seg[j][2]]
            p_b = [trj_seg[j + 1][1], trj_seg[j + 1][2]]
            t_a = trj_seg[j][0]
            t_b = trj_seg[j + 1][0]

            delta_t = t_b - t_a
            if delta_t > MAX_STAY_TIME_INTERVAL:
                continue
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

            if not NO_LIMIT:
                if v > SPEED_LIMIT[trj_seg_label]:  # ?? or v == 0
                    # logger.info('invalid speed:{} for {}'.format(v, MODE_NAMES[trj_seg_label]))
                    continue
                if abs(a) > ACC_LIMIT[trj_seg_label]:
                    # logger.info('invalid acc:{} for {}'.format(a, MODE_NAMES[trj_seg_label]))
                    continue
                if abs(hcr) > HCR_LIMIT[trj_seg_label]:  # ?? or hcr == 0       tn == 1 and s == 0 and
                    # logger.info('invalid hcr:{} for {}'.format(hcr, MODE_NAMES[trj_seg_label]))
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
            # logger.info('feature element num not enough:{}'.format(len(delta_times)))
            continue

        n_removed_points.append(0 if n_points == len(delta_times) + 1 else n_points - len(delta_times) + 1)

        multi_feature_seg = np.array(
            [fill_seg_function(delta_times, target_size=seg_size),
                 fill_seg_function(hours, target_size=seg_size),
                 fill_seg_function(distances, target_size=seg_size),
                 fill_seg_function(velocities, target_size=seg_size),
                 fill_seg_function(accelerations, target_size=seg_size),
                 fill_seg_function(headings, target_size=seg_size),
                 fill_seg_function(heading_changes, target_size=seg_size),
                 fill_seg_function(heading_change_rates, target_size=seg_size),
                 fill_seg_function(stops, target_size=seg_size),
                 fill_seg_function(turnings, target_size=seg_size)])
        multi_feature_segs.append(multi_feature_seg)
        multi_feature_seg_labels.append(trj_seg_label)

        valid_trj_segs.append(i)
    logger.info('* end a thread for calc_trjs_segs_clean_features')
    return np.array(multi_feature_segs), np.array(multi_feature_seg_labels), valid_trj_segs, n_removed_points


def calc_trj_seg_clean_features(trj_segs, trj_seg_labels, fill_seg_function, seg_size):
    logger.info('calc_trjs_segs_clean_features...')
    tasks = []
    batch_size = int(len(trj_segs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_calc_trj_seg_clean_multi_features,
                                      (trj_segs[i * batch_size:(i + 1) * batch_size],
                                       trj_seg_labels[i * batch_size:(i + 1) * batch_size],
                                       fill_seg_function, seg_size)))
    res = np.array([[t.get()[0], t.get()[1], t.get()[2], t.get()[3]] for t in tasks])
    logger.info('merging...')
    multi_feature_segs = np.concatenate(res[:, 0])
    multi_feature_seg_labels = np.concatenate(res[:, 1])
    valid_trj_segs = np.concatenate(res[:, 2])
    n_removed_points = np.concatenate(res[:, 3])
    return multi_feature_segs, multi_feature_seg_labels, valid_trj_segs, n_removed_points


def calc_trjs_segs_noise_features(trjs_segs, trjs_segs_labels, fill_series_function, valid_trjs_segs, seg_size):
    """
    * note: not used
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
            # logger.info('feature element num not enough:{}'.format(len(delta_times)))
            continue
        trj_seg_features = np.array(
            [[delta_t, hour, d, v, a, h, hc, hcr, s, tn] for delta_t, hour, d, v, a, h, hc, hcr, s, tn in
             zip(fill_series_function(delta_times, target_size=seg_size),
                 fill_series_function(hours, target_size=seg_size),
                 fill_series_function(distances, target_size=seg_size),
                 fill_series_function(velocities, target_size=seg_size),
                 fill_series_function(accelerations, target_size=seg_size),
                 fill_series_function(headings, target_size=seg_size),
                 fill_series_function(heading_changes, target_size=seg_size),
                 fill_series_function(heading_change_rates, target_size=seg_size),
                 fill_series_function(stops, target_size=seg_size),
                 fill_series_function(turnings, target_size=seg_size))]
        )
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
        #     logger.info('short seg')
        #     continue
        new_trjs.append(new_trj)
    return np.array(new_trjs)


if __name__ == '__main__':
    start = time.time()
    n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    logger.info(f'n_thread:{n_threads}')

    parser = argparse.ArgumentParser(description='TRJ_SEG_FEATURE')
    parser.add_argument('--trjs_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--seg_size', type=int, default=MAX_SEGMENT_SIZE)
    parser.add_argument('--feature_set', type=str)
    parser.add_argument('--data_type', type=str, required=True)  # train or test
    parser.add_argument('--save_dir', type=str, required=True)
    # note！！！: after random drop points in trajectory,
    # the produced features segs will have the different number of samples to the original features series
    parser.add_argument('--random_drop_percentage', type=float, default='0.')

    args = parser.parse_args()
    if args.feature_set is None:
        feature_set = FEATURES_SET_1
    else:
        feature_set = [int(item) for item in args.feature_set.split(',')]
    logger.info(f'feature_set:{feature_set}')

    trjs = np.load(args.trjs_path, allow_pickle=True)
    labels = np.load(args.labels_path, allow_pickle=True)

    if args.random_drop_percentage:
        logger.info(f'random_drop_percentage:{args.random_drop_percentage}')
        trjs = random_drop_points(trjs, args.random_drop_percentage)

    fill_seg_function = interp_single_seg

    trj_segs, trj_seg_labels = segment_trjs(trjs, labels, args.seg_size)
    logger.info('total n_points:{}'.format(np.sum([len(seg) for seg in trj_segs])))
    trj_segs, trj_seg_labels = filter_error_gps_data(trj_segs, trj_seg_labels)
    logger.info('total n_points after filtering invalid gps:{}'.format(np.sum([len(seg) for seg in trj_segs])))
    multi_feature_segs, multi_feature_seg_labels, _, n_removed_points = calc_trj_seg_clean_features(trj_segs,
                                                                                                    trj_seg_labels,
                                                                                                    fill_seg_function,
                                                                                                    args.seg_size)
    logger.info(f'total invalid MF n_points:{np.sum(n_removed_points)}')

    end = time.time()
    logger.info('Running time: %s Seconds' % (end - start))

    save_dir = args.save_dir
    data_type = args.data_type
    logger.info(f'saving files to {save_dir}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    multi_feature_segs = multi_feature_segs[:, feature_set, :]
    np.save(f'{save_dir}/multi_feature_segs_{data_type}.npy', multi_feature_segs)
    multi_feature_segs_normalized = scale_segs_each_features(multi_feature_segs)
    np.save(f'{save_dir}/multi_feature_segs_{data_type}_normalized.npy', multi_feature_segs_normalized)  # note: scaled!
    np.save(f'{save_dir}/multi_feature_seg_labels_{data_type}.npy',
            to_categorical(multi_feature_seg_labels, num_classes=N_CLASS))  # labels to one-hot
