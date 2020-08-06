from dataset_generation import *
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report

from Geolife_trajectory_extraction import modes_to_use
from params import MAX_SEGMENT_SIZE, FEATURES_SET_1


def RNN_Softmax(timesteps, embedding_dim, n_features, n_class):
    model = Sequential()
    model.add(SimpleRNN(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, timesteps, n_features),
        output_dim=int(embedding_dim),
        unroll=True,
    ))
    model.add(Dense(n_class, activation='softmax'))
    model.summary()

    plot_model(model, to_file='./comparison_results/rnn_softmax.png', show_shapes=True)

    learning_rate = 1e-3

    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(epochs=100, batch_size=200):
    factor = 1. / np.cbrt(2)
    model_checkpoint = ModelCheckpoint("./comparison_results/rnn_softmax.model", verbose=1,
                                       monitor='val_acc', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
    callback_list = [model_checkpoint, reduce_lr, early_stopping]

    hist = model.fit(np.squeeze(x_features_series_train), y_train, epochs=epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(
                         [np.squeeze(x_features_series_test)],
                         [y_test]),
                     callbacks=callback_list)
    score = np.argmax(hist.history['val_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                             np.max(hist.history['val_acc'])))


def show_confusion_matrix():
    model = load_model('./comparison_results/rnn_softmax.model', custom_objects={'N_CLASS': N_CLASS})
    pred = model.predict([np.squeeze(x_features_series_test)])
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


if __name__ == "__main__":
    model = RNN_Softmax(MAX_SEGMENT_SIZE, 32, len(FEATURES_SET_1), N_CLASS)
    patience = 35
    train(3000)
    show_confusion_matrix()
