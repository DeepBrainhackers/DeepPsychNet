import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class DeepPsychNet(object):

    def __init__(self, X, y, n_classes, conv_layers_params, max_pool_layers_params, fc_layers_params,
                 init_mu=0., init_sigma=0.1, dropout=True, use_prior=True):
        assert len(conv_layers_params) == len(max_pool_layers_params), 'Expects same number of ' \
                                                                       'convolutional/max-pooling layers'
        self.input = X
        self.label = y
        self.prior = use_prior
        self.coordinates = tf.placeholder(tf.float32)

        self.n_classes = n_classes

        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.conv_params = conv_layers_params
        self.max_pool_params = max_pool_layers_params
        self.num_layers_conv = len(self.conv_params)

        self.fc_params = fc_layers_params
        self.num_layers_fc = len(fc_layers_params)
        self.num_layers = self.num_layers_conv + self.num_layers_fc
        self.dropout = dropout
        self.keep_dims = tf.placeholder(tf.float32)

        self.network, self.network2 = self.initialize_network()

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

            conv = tf.nn.conv3d(input_to_layer, conv_W, strides=stride_conv_layer, padding='VALID') + conv_b

            # Activation.
            conv = tf.nn.relu(conv)

            if self.dropout and (id_layer == (self.num_layers_conv - 1)):
                conv = tf.nn.dropout(conv, self.keep_dims)

            # Pooling
            input_to_layer = tf.nn.max_pool3d(conv, ksize=ksize, strides=strides, padding='VALID')

        return self.create_fully_connected(flatten(input_to_layer))

    def create_fully_connected(self, input_to_layer):
        output_network = np.nan
        output_network2 = None

        for id_layer in xrange(self.num_layers_fc):
            fc_params_layer = self.fc_params[id_layer]
            shape_layer = fc_params_layer['shape']

            # Fully Connected Layers. Input = previous
            if isinstance(shape_layer, list) and len(shape_layer) == 2:
                fc_W1 = tf.Variable(tf.truncated_normal(shape=shape_layer[0], mean=self.init_mu, stddev=self.init_sigma))
                fc_b1 = tf.Variable(tf.zeros(shape_layer[0][-1]))

                fc1 = tf.matmul(input_to_layer, fc_W1) + fc_b1

                fc_W2 = tf.Variable(tf.truncated_normal(shape=shape_layer[1], mean=self.init_mu, stddev=self.init_sigma))
                fc_b2 = tf.Variable(tf.zeros(shape_layer[1][-1]))

                fc2 = tf.matmul(input_to_layer, fc_W2) + fc_b2
            else:
                fc_W = tf.Variable(tf.truncated_normal(shape=shape_layer, mean=self.init_mu, stddev=self.init_sigma))
                fc_b = tf.Variable(tf.zeros(shape_layer[-1]))

                fc = tf.matmul(input_to_layer, fc_W) + fc_b


            # Activation.
            if id_layer < (self.num_layers_fc - 1):
                input_to_layer = tf.nn.relu(fc)
            else:
                if self.prior and (len(shape_layer) == 2):
                    output_network = fc1
                    output_network2 = fc2
                else:
                    output_network = fc

            if self.dropout and (id_layer == 1):
                input_to_layer = tf.nn.dropout(input_to_layer, self.keep_dims)

        return output_network, output_network2

    def get_cost_function(self):
        cost_function = tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.one_hot_y_encoding)

        if self.prior:
            mse = tf.losses.mean_squared_error(labels=self.coordinates, predictions=self.network2)
            cost_function = tf.add(tf.multiply(cost_function, 1.), tf.multiply(mse, 0.001))
        return tf.reduce_mean(cost_function)

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
