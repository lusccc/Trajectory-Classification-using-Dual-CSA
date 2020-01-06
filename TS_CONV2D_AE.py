from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Activation
from keras.layers import Dense
from keras.utils import plot_model


def TS_CONV2D_AE(input_shape, embedding_dim, n_features, name):
    """
    time series conv2d ae
    """
    size = input_shape[1]  # series size
    print('input_shape:', input_shape)
    activ = 'relu'
    conv_ae = Sequential()
    conv_ae.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
    conv_ae.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same'))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
    conv_ae.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same'))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))

    conv_ae.add(Flatten())
    conv_ae.add(Dense(units=embedding_dim, name='{}_embedding'.format(name)))
    intermediate_size = int(size/8)
    conv_ae.add(Dense(units=1 * intermediate_size * 128, activation=activ))
    conv_ae.add(Reshape((1, intermediate_size, 128)))

    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(64, (1, 3), strides=(1, 1), padding='same'))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(32, (1, 3), strides=(1, 1), padding='same'))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(n_features, (1, 3), strides=(1, 1), padding='same'))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ, name='{}_reconstruction'.format(name)))

    conv_ae.summary()
    plot_model(conv_ae, to_file='./results/{}_conv2d_ae.png'.format(name), show_shapes=True)
    return conv_ae
