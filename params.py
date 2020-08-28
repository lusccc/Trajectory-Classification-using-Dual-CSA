from sklearn.preprocessing import MinMaxScaler

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# if multiple autoencoder exist, say n, each embedding dim will be TOTAL_EMBEDDING_DIM/n
TOTAL_EMBEDDING_DIM = 48

# 0,    1,    2,   3,         4           5
# walk, bike, bus, driving, train/subway, run
# modes_to_use = [0, 1, 2, 3, 4] # geolife
modes_to_use = [0, 1, 2, 3, 4, ]  # Trajectory_Feature_Dataset

N_CLASS = len(modes_to_use)

MAX_SEGMENT_SIZE = 200
MIN_N_POINTS = 10

DIM = 3  # Embedding dimension
TAU = 8  # Embedding delay

N_VECTORS = MAX_SEGMENT_SIZE - TAU * (DIM - 1)  # RP mat size

SCALER = MinMaxScaler()

RP_MAT_SCALE_EACH_FEATURE = False
RP_MAT_SCALE_ALL = False

SCALE_SEGS_EACH_FEATURE = False

FILTER_SEGS = False

# 0        1     2  3  4  5  6   7    8  9
# delta_t, hour, d, v, a, h, hc, hcr, s, tn
# FEATURES_SET_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FEATURES_SET_1 = [ 3, 4, 7, 8, 9]
# FEATURES_SET_1 = [3,4,6]
FEATURES_SET_2 = [3, 4, 7, 8, 9]
# loss wight
ALPHA = 1
BETA = 4
GAMMA = 1

MULTI_GPU = True
