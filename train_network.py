import os
from sys import stdout
import os.path as osp

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# from deepPsychNet import DeepPsychNet
from deepPsychNet_keras import init_network as get_keras_network
from imageTransformer3d import ImageTransformer3d


def run():
    hdf5_file = '/home/rthomas/BrainHack/dataHDF5/abide.hdf5'
    save_folder = '/home/rthomas/BrainHack/dataHDF5/DeepPsychNet'
    model_folder = 'lenet3d'
    model_name = 'lenet3d'
    batch_size = 25
    num_epochs = 20
    # if 1 no augmentation will be performed. Has to be a multiple of batch_size otherwise
    num_augmentations = 5

    iterate_and_train(hdf5_file_path=hdf5_file, save_path=save_folder, model_folder=model_folder, model_name=model_name,
                      batch_size=batch_size, num_augmentation=num_augmentations, num_epochs=num_epochs)


def init_network(batch_size=None, n_classes=2):
    return get_keras_network(n_classes=n_classes)


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
    num_performance_metrics = len(network.metrics_names)

    metrics_batches = np.zeros((num_batches, num_performance_metrics))

    image_iterator_transformer = ImageTransformer3d(data_obj=data, affine=affine, y_label=y_data,
                                                    id_data=id_to_take, batch_size=batch_size,
                                                    num_augmentation_set=1, shuffle=False)
    image_generator = image_iterator_transformer.iter()

    for i_batch, (batch_x, batch_y, _) in enumerate(image_generator):
        stdout.write('\r {}/{}'.format(i_batch + 1, num_batches))
        stdout.flush()

        metrics = network.test_on_batch(batch_x, batch_y)
        metrics_batches[i_batch, :] = metrics

    print
    return metrics_batches


def train_network(data, y, affine, id_train, id_valid, id_test, network, save_path, model_folder, model_name,
                  batch_size=25, num_epochs=20, num_augmentation=1):
    saver = tf.train.Saver()

    num_batches_train = int(np.ceil(id_train.size/((float(batch_size)/num_augmentation))))
    num_batches_train_validation = int(np.ceil(id_train.size/(float(batch_size))))
    num_batches_test = int(np.ceil(id_test.size/float(batch_size)))
    num_batches_valid = int(np.ceil(id_valid.size/float(batch_size)))

    num_metrics = len(network.metrics_names)

    metrics_train = np.zeros((num_batches_train_validation, num_epochs, num_metrics))
    metrics_test = np.zeros((num_batches_test, num_epochs, num_metrics))
    metrics_valid = np.zeros((num_batches_valid, num_epochs, num_metrics))

    model_save = osp.join(save_path, model_folder)

    if not osp.exists(model_save):
        os.makedirs(model_save)


    print "Training..."
    print

    for id_epoch in xrange(num_epochs):
        print 'EPOCH {}/{}:'.format(id_epoch + 1, num_epochs)

        image_iterator_transformer = ImageTransformer3d(data_obj=data, affine=affine, y_label=y,
                                                        id_data=id_train, batch_size=batch_size,
                                                        num_augmentation_set=num_augmentation, shuffle=True)
        image_generator = image_iterator_transformer.iter()

        for id_batch, (batch_x, batch_y, affine_train) in enumerate(image_generator):
            stdout.write('\r {}/{}'.format(id_batch + 1, num_batches_train))
            stdout.flush()

            network.train_on_batch(batch_x, batch_y)

        print
        print 'Validation...'
        metrics_valid[:, id_epoch, :] = evaluate(data, y, id_valid, network, affine, batch_size=batch_size)
        metrics_train[:, id_epoch, :] = evaluate(data, y, id_train, network, affine, batch_size=batch_size)
        metrics_test[:, id_epoch, :] = evaluate(data, y, id_test, network, affine, batch_size=batch_size)

        print
        print 'Training: '
        print_metrics(metrics_train[:, id_epoch, :].mean(axis=0), network.metrics_names)
        print 'Validation: '
        print_metrics(metrics_valid[:, id_epoch, :].mean(axis=0), network.metrics_names)
        print 'Test: '
        print_metrics(metrics_test[:, id_epoch, :].mean(axis=0), network.metrics_names)

        network.save(osp.join(model_save, model_name + '_epoch_{}.h5'.format(id_epoch + 1)))

    np.savez_compressed(osp.join(model_save, 'metrics_model.npz'), metrics_train=metrics_train,
                        metrics_valid=metrics_valid, metrics_test=metrics_test, metrics_names=network.metrics_names)
    print 'Model saved!'


def print_metrics(metrics_array, metric_names):
    print_str = ''
    for i in xrange(len(metric_names)):
        print_str += '{}: {:.2f} '.format(metric_names[i], metrics_array[i])
    print print_str


def iterate_and_train(hdf5_file_path, save_path, model_folder='model', model_name='model.ckpt', batch_size=25,
                      num_epochs=20, num_augmentation=1):
    network = init_network()

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        affine = hdf5_file['dataAffine']
        y_labels = dataT1.attrs['labels_subj'].astype(np.int32)
        id_subj = np.arange(dataT1.shape[0])
        id_train, id_valid, id_test = create_train_validation_test_set(id_subj, y_labels, num_test=100, num_valid=100)
        train_network(dataT1, y_labels, affine, id_train, id_valid, id_test, network, save_path, model_folder,
                      model_name, batch_size=batch_size, num_epochs=num_epochs, num_augmentation=num_augmentation)


if __name__ == '__main__':
    run()
