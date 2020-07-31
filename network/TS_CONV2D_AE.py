import os

from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Activation, \
    ZeroPadding2D, Cropping2D
from keras.layers import Dense
from keras.utils import plot_model


def TS_CONV1D_AE(input_shape, embedding_dim, n_features, name, results_path, zero_padding=0):
    """
    time series conv ae
    """
    size = input_shape[1]  # series size
    # print('input_shape:', input_shape)
    activ = 'relu'
    conv_ae = Sequential()
    conv_ae.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    # zero padding to handle odd size https://stackoverflow.com/questions/50515409/keras-shapes-while-upsampling-mismatch
    print(f'zero_padding:{zero_padding}')
    conv_ae.add(ZeroPadding2D(zero_padding))
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
    conv_ae.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same'))
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
    conv_ae.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same'))
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(1, 2)))

    conv_ae.add(Flatten())
    conv_ae.add(Dense(units=embedding_dim, name='{}_embedding'.format(name)))
    #                         mat width     + zero_padding * 2
    intermediate_size = int((size + zero_padding * 2) / 8)
    conv_ae.add(Dense(units=1 * intermediate_size * 128, activation=activ))
    conv_ae.add(Reshape((1, intermediate_size, 128)))

    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(64, (1, 3), strides=(1, 1), padding='same'))
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(32, (1, 3), strides=(1, 1), padding='same'))
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(1, 2)))
    conv_ae.add(Conv2DTranspose(n_features, (1, 3), strides=(1, 1), padding='same'))
    conv_ae.add(Activation(activ))
    conv_ae.add(Cropping2D(zero_padding, name='{}_reconstruction'.format(name)))  # crop back to input size
    print(f'###summary for {name}###')
    conv_ae.summary()
    plot_model(conv_ae, to_file=os.path.join(results_path,'{}_conv2d_ae.png'.format(name)), show_shapes=True)
    return conv_ae
