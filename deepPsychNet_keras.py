from keras.layers import Conv3D, Dropout, InputLayer, Dense, MaxPooling3D, BatchNormalization, Flatten, Activation
from keras.models import Sequential
import tensorflow as tf


def deep_psych_net(input_shape, conv_params, max_pool_params, fc_params, dropout_params, input_dtype=tf.float32):
    """
    :param input_shape:         (nx, ny, nz, 1)
    :param conv_params:         list of dictionaries (each dictionary a layer)
    :param max_pool_params:     list of dictionaries (each dictionary a layer)
    :param fc_params:           list of dictionaries (each dictionary a layer)
    :param dropout_params:      dictionary: {fc_layer_number: dropout_value}, starting from 1 or None
    :param input_dtype:         default: tf.float32               
    :return: 
    """
    assert len(conv_params) == len(max_pool_params), '#of conv- and max-pooling layers must match currently!'

    if dropout_params is None:
        dropout_params = {}

    n_conv = len(conv_params)
    n_fc = len(fc_params)

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape, batch_size=None, dtype=input_dtype, name='input_layer'))

    for i in xrange(n_conv):
        conv_layer_params = conv_params[i]
        max_pool_layer_params = max_pool_params[i]
        model = conv_layers(model, conv_layer_params, max_pool_layer_params, i=i+1)

    model.add(Flatten(name='flatten'))

    for i in xrange(n_fc):
        fc_layer_params = fc_params[i]
        dropout_val = dropout_params.get(i + 1, False)

        model = fully_connected(model, fc_layer_params, dropout=dropout_val, i=i+1, final_layer=(i + 1) == n_fc)
    return model


def conv_layers(model, conv_params, max_pool_params, i=1):
    # Our convolutional layers will always be comprised of
    #
    # 1. conv-layer (without non-linearity)
    # 2. batch_normalization
    # 3. non-linear activation (relu)
    # 4. max-pooling

    model.add(Conv3D(name='conv{}'.format(i), **conv_params))
    model.add(BatchNormalization(name='bn{}'.format(i)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(name='maxpool{}'.format(i), **max_pool_params))
    return model


def fully_connected(model, fc_params, dropout=False, i=1, final_layer=False):
    # Fully connected layers
    #
    # 1. Dense layer (with linear activation)
    # 2. Non-linear activation (relu if not final_layer, or softmax for final_layer)
    # 3. (optional) Dropout

    model.add(Dense(name='fc{}'.format(i), **fc_params))

    if not final_layer:
        model.add(Activation('relu'))
    else:
        model.add(Activation('softmax'))

    if dropout:
        model.add(Dropout(rate=dropout, name='dropout{}'.format(i)))

    return model


def init_network_test():
    input_shape = (91, 109, 91, 1)
    input_dtype = tf.float32

    conv_params = [
        {'filters': 64, "kernel_size": (7, 7, 7), "strides": (3, 3, 3)},
        {'filters': 32, "kernel_size": (5, 5, 5), "strides": (1, 1, 1)},
    ]
    maxpooling_params = [
        {"pool_size": (2, 2, 2), 'strides': (2, 2, 2)},
        {"pool_size": (2, 2, 2), 'strides': (2, 2, 2)}
    ]
    fc_params = [
        {'units': 1000},
        {'units': 100},
        {'units': 2}
    ]

    # dropout_params = None
    # dropout_params = {1: 0.9, 2: 0.5}
    dropout_params = {2: 0.5}

    model = deep_psych_net(input_shape, conv_params, maxpooling_params, fc_params, dropout_params, input_dtype)
    model.summary()


if __name__ == '__main__':
    init_network_test()