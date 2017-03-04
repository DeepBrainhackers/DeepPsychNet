import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class DeepPsychNet(object):

    def __init__(self, X, y, n_classes, conv_layers_params, max_pool_layers_params, fc_layers_params,
                 init_mu=0., init_sigma=0.1):
        assert len(conv_layers_params) == len(max_pool_layers_params), 'Expects same number of ' \
                                                                       'convolutional/max-pooling layers'

        self.input = X
        self.label = y
        self.n_classes = n_classes

        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.conv_params = conv_layers_params
        self.max_pool_params = max_pool_layers_params
        self.num_layers_conv = len(self.conv_params)

        self.fc_params = fc_layers_params
        self.num_layers_fc = len(fc_layers_params)
        self.num_layers = self.num_layers_conv + self.num_layers_fc
        self.variable_list = []

        self.network = self.initialize_network()

        self.one_hot_y_encoding = self.get_hot_encoding()
        self.cost_function = self.get_cost_function()

    def initialize_network(self):

        input_to_layer = self.input

        for id_layer in xrange(self.num_layers_conv):
            conv_params_layer = self.conv_params[id_layer]
            max_pool_layer = self.max_pool_params[id_layer]

            shape_conv_layer = conv_params_layer['shape']
            stride_conv_layer = conv_params_layer['strides']

            ksize = max_pool_layer['ksize']
            strides = max_pool_layer['strides']

            conv_W = tf.Variable(tf.truncated_normal(shape=shape_conv_layer,
                                                     mean=self.init_mu,
                                                     stddev=self.init_sigma))
            conv_b = tf.Variable(tf.zeros(shape_conv_layer[-1]))
            self.variable_list += [conv_W]
            self.variable_list += [conv_b]

            conv = tf.nn.conv3d(input_to_layer, conv_W, strides=stride_conv_layer, padding='VALID') + conv_b

            # Activation.
            conv = tf.nn.relu(conv)

            # Pooling
            input_to_layer = tf.nn.max_pool3d(conv, ksize=ksize, strides=strides, padding='VALID')

        return self.create_fully_connected(flatten(input_to_layer))

    def create_fully_connected(self, input_to_layer):
        output_network = np.nan

        for id_layer in xrange(self.num_layers_fc):
            fc_params_layer = self.fc_params[id_layer]
            shape_layer = fc_params_layer['shape']

            # Layer 3: Fully Connected. Input = previous. Output = 120.
            fc_W = tf.Variable(tf.truncated_normal(shape=shape_layer, mean=self.init_mu, stddev=self.init_sigma))
            fc_b = tf.Variable(tf.zeros(shape_layer[-1]))
            self.variable_list += [fc_W]
            self.variable_list += [fc_b]

            fc = tf.matmul(input_to_layer, fc_W) + fc_b

            # Activation.
            if id_layer < (self.num_layers_fc - 1):
                input_to_layer = tf.nn.relu(fc)
            else:
                output_network = fc

        return output_network

    def get_cost_function(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.one_hot_y_encoding)
        return tf.reduce_mean(cross_entropy)

    def get_hot_encoding(self):
        return tf.one_hot(self.label, self.n_classes)

    def get_training_function(self, learning_rate=0.001):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost_function)

    def get_performance(self):
        correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.one_hot_y_encoding, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def run():
    n_classes = 2
    batch_size = 64
    nx, ny, nz = 91, 109, 91
    conv_params = [{'shape': (8, 8, 8, 1, 6),
                    'strides': (1, 2, 2, 2, 1)},
                   {'shape': (5, 5, 5, 6, 16),
                    'strides': (1, 2, 2, 2, 1)}
                   ]
    max_pool_params = [{'ksize': (1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)},
                       {'ksize': (1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)}
                       ]
    fc_params = [{'shape': (1280, 120)},
                 {'shape': (120, 84)},
                 {'shape': (84, n_classes)}]
    X = tf.placeholder(tf.float32, (batch_size, nx, ny, nz, 1))
    y = tf.placeholder(tf.int32, (batch_size))
    network = DeepPsychNet(X, y, n_classes=n_classes, conv_layers_params=conv_params,
                           max_pool_layers_params=max_pool_params, fc_layers_params=fc_params)


if __name__ == '__main__':
    run()
