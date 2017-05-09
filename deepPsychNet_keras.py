from keras.layers import Conv3D, Dropout, Input, Dense, MaxPooling3D, BatchNormalization, Flatten, Activation
from keras.models import Model
import keras.backend as K


def deep_psych_net(input_shape, conv_params, max_pool_params, fc_params, dropout_params, output_params,
                   input_dtype=K.floatx()):
    """
    :param input_shape:         (nx, ny, nz, 1)
    :param conv_params:         list of dictionaries (each dictionary a layer)
    :param max_pool_params:     list of dictionaries (each dictionary a layer)
    :param fc_params:           list of dictionaries (each dictionary a layer)
    :param dropout_params:      dictionary: {fc_layer_number: dropout_value}, starting from 1 or None
    :param output_params:        list of dictionaries (each dictionary a final layer)
    :param input_dtype:         default: tf.float32               
    :return: 
    """
    assert len(conv_params) == len(max_pool_params), '#of conv- and max-pooling layers must match currently!'

    if dropout_params is None:
        dropout_params = {}

    n_conv = len(conv_params)
    n_fc = len(fc_params)
    n_output = len(output_params)

    input_to_network = Input(input_shape, batch_shape=None, dtype=input_dtype, name='input_layer')
    layer_input = input_to_network

    for i in xrange(n_conv):
        conv_layer_params = conv_params[i]
        max_pool_layer_params = max_pool_params[i]
        layer_input = conv_layers(layer_input, conv_layer_params, max_pool_layer_params, i=i+1)

    layer_input = Flatten(name='flatten')(layer_input)

    for i in xrange(n_fc):
        fc_layer_params = fc_params[i]
        dropout_val = dropout_params.get(i + 1, False)
        layer_input = fully_connected(layer_input, fc_layer_params, dropout=dropout_val, i=i+1)

    outputs_network = []
    for i in xrange(n_output):
        output_layer_params = output_params[i]
        outputs_network.append(output_layers(layer_input, output_layer_params, i=i+1))

    return Model(inputs=input_to_network, outputs=outputs_network)


def output_layers(x, layer_params, i=1):
    return Dense(name='output{}'.format(i), **layer_params)(x)


def conv_layers(x, conv_params, max_pool_params, i=1):
    # Our convolutional layers will always be comprised of
    #
    # 1. conv-layer (without non-linearity)
    # 2. batch_normalization
    # 3. non-linear activation (relu)
    # 4. max-pooling

    x = Conv3D(name='conv{}'.format(i), **conv_params)(x)
    x = BatchNormalization(name='bn{}'.format(i))(x)
    x = Activation('relu', name='conv{}_relu'.format(i))(x)
    x = MaxPooling3D(name='maxpool{}'.format(i), **max_pool_params)(x)
    return x


def fully_connected(x, fc_params, dropout=False, i=1, final_layer=False):
    # Fully connected layers
    #
    # 1. Dense layer (with linear activation)
    # 2. Non-linear activation (relu if not final_layer, or softmax for final_layer)
    # 3. (optional) Dropout

    x = Dense(name='fc{}'.format(i), **fc_params)(x)
    x = Activation('relu', name='fc{}_relu'.format(i))(x)

    if dropout:
        x = Dropout(rate=dropout, name='dropout{}'.format(i))(x)
    return x


def init_network(n_classes=2):
    input_shape = (91, 109, 91, 1)
    input_dtype = K.floatx()

    conv_params = [
        {'filters': 128, 'kernel_size': (7, 7, 7), 'strides': (2, 2, 2)},
        {'filters': 64, "kernel_size": (7, 7, 7), "strides": (1, 1, 1)},
        {'filters': 32, "kernel_size": (5, 5, 5), "strides": (1, 1, 1)},
        {'filters': 16, "kernel_size": (3, 3, 3), "strides": (1, 1, 1)}
    ]
    maxpooling_params = [
        {'pool_size': (4, 4, 4), 'strides': (2, 2, 2)},
        {"pool_size": (2, 2, 2), 'strides': (1, 1, 1)},
        {"pool_size": (2, 2, 2), 'strides': (1, 1, 1)},
        {"pool_size": (2, 2, 2), 'strides': (1, 1, 1)}

    ]
    fc_params = [
        {'units': 1000},
        {'units': 100},
        {'units': 50}
    ]

    output_params = [
        {'units': n_classes, 'activation': 'softmax'} #,
        # {'units': 20, 'activation': 'linear'}
    ]

    loss = 'categorical_crossentropy'
    loss_weights = None
    # loss = ['categorical_crossentropy', 'mean_squared_error']
    # loss_weights = [1., 0.001]

    # dropout_params = None
    # dropout_params = {1: 0.9, 2: 0.5}
    # dropout_params = {1: 0.5, 2: 0.3}
    dropout_params = {1: 0.5}

    model = deep_psych_net(input_shape, conv_params, maxpooling_params, fc_params, dropout_params, output_params,
                           input_dtype)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', balanced_accuracy], loss_weights=loss_weights)
    model.summary()
    return model


def balanced_accuracy(y_true, y_pred):
    """
    Assumes y_pred is the softmax output of the network
    
    :param y_true: 
    :param y_pred: 
    :return: 
    """
    y_pred_onehot = K.one_hot(K.argmax(y_pred, axis=1), num_classes=2)
    return K.mean(K.sum(y_true * y_pred_onehot, axis=0) / K.sum(y_true, axis=0))


def test_network():
    model = init_network()
    model.summary()


if __name__ == '__main__':
    test_network()
