from trajectory_extraction import modes_to_use
# if multiple autoencoder exist, say n, each embedding dim will be TOTAL_EMBEDDING_DIM/n
TOTAL_EMBEDDING_DIM = 32
N_CLASS = len(modes_to_use)

MAX_SEGMENT_SIZE = 184
MIN_N_POINTS = 10

DIM = 8  # Embedding dimension
TAU = 8  # Embedding delay