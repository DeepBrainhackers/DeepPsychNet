from sys import stdout
import os.path as osp

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from deepPsychNet import DeepPsychNet
from imageTransformer3d import ImageTransformer3d


def run():
    hdf5_file = '/home/rthomas/BrainHack/dataHDF5/abide.hdf5'
    save_folder = '/home/rthomas/BrainHack/dataHDF5/DeepPsychNet'
    model_name = 'lenet3d.ckpt'
    batch_size = 25
    num_epochs = 20
    # if 1 no augmentation will be performed. Has to be a multiple of batch_size otherwise
    num_augmentations = 5

    iterate_and_train(hdf5_file_path=hdf5_file, save_path=save_folder, model_name=model_name, batch_size=batch_size,
                      num_augmentation=num_augmentations, num_epochs=num_epochs)


def init_network(batch_size=None, n_classes=2):
    nx, ny, nz = 91, 109, 91

    conv_params = [{'shape': (8, 8, 8, 1, 6),
                    'strides': (1, 5, 5, 5, 1)},
                   {'shape': (5, 5, 5, 6, 16),
                    'strides': (1, 2, 2, 2, 1)}
                   ]
    max_pool_params = [{'ksize': (1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)},
                       {'ksize': (1, 2, 2, 2, 1),
                        'strides': (1, 2, 2, 2, 1)}
                       ]
    fc_params = [{'shape': (16, 120)},
                 {'shape': (120, 84)},
                 {'shape': (84, n_classes)}]

    X = tf.placeholder(tf.float32, (batch_size, nx, ny, nz, 1))
    y = tf.placeholder(tf.int32, (None))

    return DeepPsychNet(X, y, n_classes=n_classes, conv_layers_params=conv_params,
                        max_pool_layers_params=max_pool_params, fc_layers_params=fc_params)


def create_train_validation_test_set(data, y, num_test=100, num_valid=100):
    """
    Define training, validation, test set
    :param data:
    :param y:
    :param num_test:
    :param num_valid:
    :return:
    """
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=num_test, stratify=y)
    data_train, data_valid, y_train, y_valid = train_test_split(data_train, y_train, test_size=num_valid,
                                                                stratify=y_train)
    return data_train, data_valid, data_test


def evaluate(data, y_data, id_to_take, network, affine, batch_size=25):
    num_examples = id_to_take.size
    sess = tf.get_default_session()
    num_batches = int(np.ceil(num_examples/float(batch_size)))
    accuracy_batches = np.zeros(num_batches)

    image_iterator_transformer = ImageTransformer3d(data_obj=data, affine=affine, y_label=y_data,
                                                    id_data=id_to_take, batch_size=batch_size,
                                                    num_augmentation_set=1, shuffle=False)
    image_generator = image_iterator_transformer.iter()

    for i_batch, (batch_x, batch_y, _) in enumerate(image_generator):
        stdout.write('\r {}/{}'.format(i_batch + 1, num_batches))
        stdout.flush()

        accuracy = sess.run(network.get_performance(), feed_dict={network.input: batch_x, network.label: batch_y})
        accuracy_batches[i_batch] = accuracy
    print
    return accuracy_batches


def train_network(data, y, affine, id_train, id_valid, id_test, network, save_path, model_name, batch_size=25,
                  num_epochs=20, num_augmentation=1):
    saver = tf.train.Saver()

    num_batches_train = int(np.ceil(id_train.size/((float(batch_size)/num_augmentation))))
    num_batches_train_validation = int(np.ceil(id_train.size/(float(batch_size))))
    num_batches_test = int(np.ceil(id_test.size/float(batch_size)))
    num_batches_valid = int(np.ceil(id_valid.size/float(batch_size)))

    accuracy_train = np.zeros((num_batches_train_validation, num_epochs))
    accuracy_test = np.zeros((num_batches_test, num_epochs))
    accuracy_valid = np.zeros((num_batches_valid, num_epochs))

    train_op = network.get_training_function()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print "Training..."
        print

        for id_epoch in xrange(num_epochs):

            image_iterator_transformer = ImageTransformer3d(data_obj=data, affine=affine, y_label=y,
                                                            id_data=id_train, batch_size=batch_size,
                                                            num_augmentation_set=num_augmentation, shuffle=True)
            image_generator = image_iterator_transformer.iter()

            for id_batch, (batch_x, batch_y, affine_train) in enumerate(image_generator):
                stdout.write('\r {}/{}'.format(id_batch + 1, num_batches_train))
                stdout.flush()

                sess.run(train_op, feed_dict={network.input: batch_x, network.label: batch_y})

            print
            print 'Validation...'
            accuracy_valid[:, id_epoch] = evaluate(data, y, id_valid, network, affine, batch_size=batch_size)
            accuracy_train[:, id_epoch] = evaluate(data, y, id_train, network, affine, batch_size=batch_size)
            accuracy_test[:, id_epoch] = evaluate(data, y, id_test, network, affine, batch_size=batch_size)

            print
            print "EPOCH {}/{}: Training Acc: {:.3f}; Validation Acc = {:.3f}; Test Acc = {:.3f}".format(id_epoch + 1,
                                                                                                         num_epochs,
                                                                                                         accuracy_train[:, id_epoch].mean(),
                                                                                                         accuracy_valid[:, id_epoch].mean(),
                                                                                                         accuracy_test[:, id_epoch].mean())
        print
        saver.save(sess, osp.join(save_path, model_name))
        np.savez_compressed(osp.join(save_path, 'accuracies_data.npz'), accuracy_train=accuracy_train,
                            accuracy_valid=accuracy_valid, accuracy_test=accuracy_test)
        print 'Model saved!'


def iterate_and_train(hdf5_file_path, save_path, model_name='model.ckpt', batch_size=25, num_epochs=20,
                      num_augmentation=1):
    network = init_network()

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        affine = hdf5_file['dataAffine']
        y_labels = dataT1.attrs['labels_subj'].astype(np.int32)
        id_subj = np.arange(dataT1.shape[0])
        id_train, id_valid, id_test = create_train_validation_test_set(id_subj, y_labels, num_test=100, num_valid=100)
        train_network(dataT1, y_labels, affine, id_train, id_valid, id_test, network, save_path, model_name,
                      batch_size=batch_size, num_epochs=num_epochs, num_augmentation=num_augmentation)


if __name__ == '__main__':
    run()
