"""
Usage:
    data_handling.py PATH

Arguments:
    PATH    Path to hdf5 data
"""

from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from docopt import docopt


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


def compute_mean_std_online(X, id_X, batch_size=50):
    n = id_X.size
    mean_data = np.zeros(X.shape[1:], dtype=np.float64)
    var_data = np.zeros_like(mean_data)

    for i in xrange(0, n, batch_size):
        id_batch = np.sort(id_X[i:i+batch_size])
        mean_data += X[id_batch, ...].sum(axis=0)/float(n)

    for i in xrange(0, n, batch_size):
        id_batch = np.sort(id_X[i:i+batch_size])
        var_data += np.sum((X[id_batch, ...] - mean_data)**2., axis=0)/(float(n) - 1)

    return mean_data, var_data, np.sqrt(var_data)


def standardize_data(X, mean, std, batch_size=50):
    for i in xrange(0, X.shape[0], batch_size):
        end_id = np.minimum(i + batch_size, X.shape[0])
        x_batch = X[i:end_id, ...]
        # some std might be zero because nothing changes, i.e. all voxels are zero (ensures numerical stability)
        X[i:end_id, ...] = (x_batch - mean)/(std + 0.0000000001)


def run(PATH):
    with h5py.File(PATH, 'r+') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        mean_var_std = hdf5_file.create_dataset('mean_var_std_train', shape=(3, ) + dataT1.shape[1:], dtype=np.float64)
        n = dataT1.shape[0]
        y = dataT1.attrs['labels_subj'].astype(np.int)
        id_train, id_valid, id_test = create_train_validation_test_set(np.arange(n), y)

        dataT1.attrs['id_train'] = id_train
        dataT1.attrs['id_valid'] = id_valid
        dataT1.attrs['id_test'] = id_test

        mean, var, std = compute_mean_std_online(dataT1, id_train)
        mean_var_std[0] = mean
        mean_var_std[1] = var
        mean_var_std[2] = std

        standardize_data(dataT1, mean, std)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(**args)
