from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Lambda, \
    BatchNormalization, Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras import backend as K, Sequential


def Conv_AE(input_shape, LATENT_VARIABLE_DIM, n_features, name):
    size = input_shape[1]  # mat size
    print('input_shape:', input_shape)
    conv_ae = Sequential()
    activ = 'relu'
    conv_ae.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
    conv_ae.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
    conv_ae.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2)))

    conv_ae.add(Flatten())
    conv_ae.add(Dense(units=LATENT_VARIABLE_DIM, name='{}_embedding'.format(name)))
    intermediate_size = int(size/8)
    conv_ae.add(Dense(units=intermediate_size * intermediate_size * 128, activation=activ))
    conv_ae.add(Reshape((intermediate_size, intermediate_size, 128)))

    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same'))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same'))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(n_features, (3, 3), strides=(1, 1), padding='same'))
    conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ, name='{}_reconstruction'.format(name)))

    conv_ae.summary()
    plot_model(conv_ae, to_file='./results/{}_conv_ae.png'.format(name), show_shapes=True)
    return conv_ae
