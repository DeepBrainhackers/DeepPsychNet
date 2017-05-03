import os
import json
from sys import stdout
import os.path as osp

import h5py
import numpy as np
from keras.utils import to_categorical

from deepPsychNet_keras import init_network as LeNet3D
from imageTransformer3d import ImageTransformer3d
from time import time
from resnet_architechture import ResNet as ResNet3D


def run():
    hdf5_file = '/home/paulgpu/git/DeepPsychNet/dataHDF5/abide.hdf5'
    save_folder = '/home/paulgpu/git/DeepPsychNet'

    model_name = 'LeNet3D'  # currently only available: 'ResNet3D' or 'LeNet3D'
    model_folder = model_name
    save_folder = osp.join(save_folder, model_folder)

    batch_size = 25
    num_epochs = 500
    # if 1 no augmentation will be performed. Has to be a multiple of batch_size otherwise
    num_augmentations = 5
    type_augmentation = 'translation'

    print 'Meta-Parameters: '
    print 'Data: {}; SaveFolder: {}; ModelName: {}'.format(hdf5_file, save_folder, model_name)
    print 'BatchSize: {}; NumEpochs: {}; NumAugmentation {}; TypeAugmentation {}'.format(batch_size, num_epochs,
                                                                                         num_augmentations,
                                                                                         type_augmentation)
    print

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    save_meta_params(data=hdf5_file, save_folder=save_folder, model_name=model_name, model_folder=model_folder,
                     batch_size=batch_size, num_epochs=num_epochs, num_augmentations=num_augmentations,
                     type_augmentation=type_augmentation)

    iterate_and_train(hdf5_file_path=hdf5_file, save_path=save_folder, model_name=model_name, batch_size=batch_size,
                      num_augmentation=num_augmentations, type_augmentation=type_augmentation, num_epochs=num_epochs)


def save_meta_params(**kwargs):
    save_folder = kwargs.get('save_folder', '.')

    with open(osp.join(save_folder, 'meta_params.json'), 'wb') as json_file:
        json.dump(kwargs, json_file)


def init_network(model_name, save_path, n_classes=2):
    if model_name == 'LeNet3D':
        model = LeNet3D(n_classes=n_classes)
    elif model_name == 'ResNet3D':
        model = ResNet3D()
    else:
        raise RuntimeError("Currently only 'LeNet3D' and 'ResNet3D' are implemented. You chose {}".format(model_name))

    with open(osp.join(save_path, 'model_architecture.json'), 'wb') as json_file:
        json.dump(model.to_json(), json_file)

    return model


def evaluate(data, y_data, id_to_take, network, affine, batch_size=25):
    num_examples = id_to_take.size
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

        batch_y = to_categorical(batch_y, num_classes=2)
        metrics = network.test_on_batch(batch_x, batch_y)
        metrics_batches[i_batch, :] = metrics

    print
    return metrics_batches


def train_network(data, y, affine, id_train, id_valid, id_test, network, save_path, model_name,
                  batch_size=25, num_epochs=20, num_augmentation=1, type_augmentation=None):

    num_batches_train = int(np.ceil(id_train.size/(float(batch_size)/num_augmentation)))
    num_batches_train_validation = int(np.ceil(id_train.size/(float(batch_size))))
    num_batches_test = int(np.ceil(id_test.size/float(batch_size)))
    num_batches_valid = int(np.ceil(id_valid.size/float(batch_size)))

    num_metrics = len(network.metrics_names)

    metrics_train = np.zeros((num_batches_train_validation, num_epochs, num_metrics))
    metrics_test = np.zeros((num_batches_test, num_epochs, num_metrics))
    metrics_valid = np.zeros((num_batches_valid, num_epochs, num_metrics))

    print "Training..."
    print

    for id_epoch in xrange(num_epochs):
        t1_epoch = time()
        print 'EPOCH {}/{}:'.format(id_epoch + 1, num_epochs)

        image_iterator_transformer = ImageTransformer3d(data_obj=data, affine=affine, y_label=y,
                                                        id_data=id_train, batch_size=batch_size,
                                                        type_augmentation=type_augmentation,
                                                        num_augmentation_set=num_augmentation, shuffle=True)
        image_generator = image_iterator_transformer.iter()

        t1_train = time()
        for id_batch, (batch_x, batch_y, affine_train) in enumerate(image_generator):
            stdout.write('\r {}/{}'.format(id_batch + 1, num_batches_train))
            stdout.flush()

            batch_y = to_categorical(batch_y, num_classes=2)
            network.train_on_batch(batch_x, batch_y)
        t2_train = time()
        print
        print 'Training time epoch: {:.02f}m'.format((t2_train - t1_train)/60.)
        print
        print 'Validation...'
        print '... for validation set'
        metrics_valid[:, id_epoch, :] = evaluate(data, y, id_valid, network, affine, batch_size=batch_size)
        print '... for test set'
        metrics_test[:, id_epoch, :] = evaluate(data, y, id_test, network, affine, batch_size=batch_size)
        print '... for training set'
        metrics_train[:, id_epoch, :] = evaluate(data, y, id_train, network, affine, batch_size=batch_size)

        print
        print 'Training: '
        print_metrics(np.nanmean(metrics_train[:, id_epoch, :], axis=0), network.metrics_names)
        print 'Validation: '
        print_metrics(np.nanmean(metrics_valid[:, id_epoch, :], axis=0), network.metrics_names)
        print 'Test: '
        print_metrics(np.nanmean(metrics_test[:, id_epoch, :], axis=0), network.metrics_names)

        test_acc = np.nanmean(metrics_test[:, id_epoch, -1], axis=0) * 100
        network.save(osp.join(save_path, model_name + '_epoch_{:04}_test_{}.h5'.format(id_epoch + 1, int(test_acc))))

        t2_epoch = time()

        print 'Epoch: time-taken {:.2f}m'.format((t2_epoch - t1_epoch)/60.)
        print

    np.savez_compressed(osp.join(save_path, 'metrics_model.npz'), metrics_train=metrics_train,
                        metrics_valid=metrics_valid, metrics_test=metrics_test, metrics_names=network.metrics_names)
    print 'Model saved!'


def print_metrics(metrics_array, metric_names):
    print_str = ''
    for i in xrange(len(metric_names)):
        print_str += '{}: {:.4f} '.format(metric_names[i], metrics_array[i])
    print print_str


def iterate_and_train(hdf5_file_path, save_path, model_name='LeNet3D', batch_size=25, num_epochs=20,
                      num_augmentation=1, type_augmentation=None):
    network = init_network(model_name, save_path)

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        affine = hdf5_file['dataAffine']
        y_labels = dataT1.attrs['labels_subj'].astype(np.int32)
        id_train, id_valid, id_test = dataT1.attrs['id_train'], dataT1.attrs['id_valid'], dataT1.attrs['id_test']
        train_network(dataT1, y_labels, affine, id_train, id_valid, id_test, network, save_path, model_name,
                      batch_size=batch_size, num_epochs=num_epochs, num_augmentation=num_augmentation,
                      type_augmentation=type_augmentation)


if __name__ == '__main__':
    run()
