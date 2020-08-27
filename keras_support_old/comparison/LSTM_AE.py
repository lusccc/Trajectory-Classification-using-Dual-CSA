from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.utils import plot_model


def LSTM_AE(timesteps, embedding_dim, n_features, name):
    lstm_ae = Sequential()
    lstm_ae.add(LSTM(128, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    lstm_ae.add(LSTM(embedding_dim, activation='relu', return_sequences=False, name='{}_embedding'.format(name)))
    lstm_ae.add(RepeatVector(timesteps))
    lstm_ae.add(LSTM(embedding_dim, activation='relu', return_sequences=True))
    lstm_ae.add(LSTM(128, activation='relu', return_sequences=True))
    lstm_ae.add(TimeDistributed(Dense(n_features), name='{}_reconstruction'.format(name)))
    lstm_ae.summary()

    plot_model(lstm_ae, to_file='./results/{}_lstm_ae.png'.format(name), show_shapes=True)
    return lstm_ae
