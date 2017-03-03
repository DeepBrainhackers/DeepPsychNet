from sklearn.model_selection import train_test_split
import tensorflow as tf
from deepPsychNet import DeepPsychNet
import h5py
import numpy as np


def create_train_validation_test_set(data, y, num_test=100, num_valid=100):
    """
    Define training, validation, test set
    :param data:
    :param y:
    :return:
    """
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=num_test, stratify=y)
    data_train, data_valid, y_train, y_valid = train_test_split(data_train, y_train, test_size=num_valid,
                                                                stratify=y_train)
    return data_train, data_valid, data_test


def evaluate(data, y_data, id_to_take, network, batch_size=25):
    num_examples = id_to_take.size
    sess = tf.get_default_session()
    num_batches = int(np.ceil(num_examples/float(batch_size)))
    accuracy_batches = np.zeros(num_batches)

    for i_batch, offset in enumerate(xrange(0, num_examples, batch_size)):
        end = np.minimum(offset+batch_size, num_examples)
        id_minibatch = id_to_take[offset:end]
        batch_x, batch_y = data[id_minibatch, :, :, :, :], y_data[id_minibatch]
        accuracy = sess.run(network.get_performance(), feed_dict={network.input: batch_x, network.label: batch_y})
        accuracy_batches[i_batch] = accuracy
    return accuracy_batches


def train_network(data, y, id_train, id_valid, id_test, network, save_path, batch_size=25,
                  num_epochs=20):
    saver = tf.train.Saver()
    num_train_examples = id_train.size

    num_batches_train = int(np.ceil(id_train.size/float(batch_size)))
    num_batches_test = int(np.ceil(id_test.size/float(batch_size)))
    num_batches_valid = int(np.ceil(id_valid.size/float(batch_size)))

    accuracy_train = np.zeros((num_batches_train, num_epochs))
    accuracy_test = np.zeros((num_batches_test, num_epochs))
    accuracy_valid = np.zeros((num_batches_valid, num_epochs))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        print "Training..."
        print

        for id_epoch in xrange(num_epochs):

            # will shuffle the indices of the training data INPLACE (i.e. id_train changed)
            np.random.shuffle(id_train)

            for offset in xrange(0, num_train_examples, batch_size):
                end = np.minimum(offset + batch_size, num_train_examples)
                id_minibatch = id_train[offset:end]
                id_minibatch = np.sort(id_minibatch)

                batch_x, batch_y = data[id_minibatch, :, :, :, :], y[id_minibatch]
                sess.run(network.get_training_function(), feed_dict={network.input: batch_x, network.label: batch_y})

            accuracy_valid[:, id_epoch] = evaluate(data, y, id_valid, network, batch_size=batch_size)
            accuracy_train[:, id_epoch] = evaluate(data, y, id_train, network, batch_size=batch_size)
            accuracy_test[:, id_epoch] = evaluate(data, y, id_test, network, batch_size=batch_size)

            # print("EPOCH {} ...".format(i+1))
            print "EPOCH {}/{}: Training Acc: {:.3f}; Validation Acc = {:.3f}; Test Acc = {:.3f}".format(id_epoch + 1,
                                                                                                         num_epochs,
                                                                                                         accuracy_train[:, id_epoch].mean(),
                                                                                                         accuracy_valid[:, id_epoch].mean(),
                                                                                                         accuracy_test[:, id_epoch].mean())
        print
        saver.save(sess, save_path)
        print 'Model saved!'


def iterate_and_train(hdf5_file_path, save_path):
    network = init_network()

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        y_labels = dataT1.attrs['labels_subj'].astype(np.int32)
        id_subj = np.arange(dataT1.shape[0])
        id_train, id_valid, id_test = create_train_validation_test_set(id_subj, y_labels, num_test=100, num_valid=100)
        train_network(dataT1, y_labels, id_train, id_valid, id_test, network, save_path,
                      batch_size=25, num_epochs=20)


def init_network():
    n_classes = 2
    batch_size = 25
    nx, ny, nz = 91, 109, 91

    conv_params = [{'shape': (8, 8, 8, 1, 6),
                    'strides': (1, 2, 2, 2, 1)},
                   {'shape': (5, 5, 5, 6, 16),
                    'strides': (1, 2, 2, 2, 1)}
                   ]
    max_pool_params = [{'ksize':(1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)},
                       {'ksize': (1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)}
                       ]
    fc_params = [{'shape': (1280, 120)},
                 {'shape': (120, 84)},
                 {'shape': (84, n_classes)}]

    X = tf.placeholder(tf.float32, (batch_size, nx, ny, nz, 1))
    y = tf.placeholder(tf.int32, (batch_size))

    return DeepPsychNet(X, y, n_classes=n_classes, conv_layers_params=conv_params,
                        max_pool_layers_params=max_pool_params, fc_layers_params=fc_params)


if __name__ == '__main__':
    hdf5_file = '/media/paul/kaggle/dataHDF5/abide.hdf5'
    save_path = '/media/paul/kaggle/dataHDF5/DeepPsychNet'
    iterate_and_train(hdf5_file_path=hdf5_file, save_path=save_path)