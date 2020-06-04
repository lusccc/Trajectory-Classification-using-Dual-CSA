import os

from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Activation
from keras.layers import Dense
from keras.utils import plot_model


def CONV2D_AE(input_shape, embedding_dim, n_features, name, results_path):
    size = input_shape[1]  # mat size
    # print('input_shape:', input_shape)
    activ = 'relu'
    pad = 'same'
    conv_ae = Sequential()
    conv_ae.add(Conv2D(32, (3, 3), strides=(1, 1), padding=pad, input_shape=input_shape))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
    conv_ae.add(Conv2D(64, (3, 3), strides=(1, 1), padding=pad))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
    conv_ae.add(Conv2D(128, (3, 3), strides=(1, 1), padding=pad))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(MaxPooling2D(pool_size=(2, 2), padding=pad))

    conv_ae.add(Flatten())
    conv_ae.add(Dense(units=embedding_dim, name='{}_embedding'.format(name)))
    intermediate_size = int(size/8)
    conv_ae.add(Dense(units=intermediate_size * intermediate_size * 128, activation=activ))
    conv_ae.add(Reshape((intermediate_size, intermediate_size, 128)))

    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), padding=pad))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(32, (3, 3), strides=(1, 1), padding=pad))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ))
    conv_ae.add(UpSampling2D(size=(2, 2)))
    conv_ae.add(Conv2DTranspose(n_features, (3, 3), strides=(1, 1), padding=pad))
    # conv_ae.add(BatchNormalization())
    conv_ae.add(Activation(activ, name='{}_reconstruction'.format(name)))
    print(f'###summary for {name}###')
    conv_ae.summary()
    plot_model(conv_ae, to_file=os.path.join(results_path,'{}_conv2d_ae.png'.format(name)), show_shapes=True)
    return conv_ae

def CAE(embedding_dim, n_features, name, input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    size = input_shape[1]  # mat size
    if size == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=embedding_dim, name='{}_embedding'.format(name)))
    model.add(Dense(units=filters[2]*int(size/8)*int(size/8), activation='relu'))

    model.add(Reshape((int(size/8), int(size/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='{}_reconstruction'.format(name)))
    model.summary()
    plot_model(model, to_file='./results/{}_conv2d_ae.png'.format(name), show_shapes=True)
    return model
