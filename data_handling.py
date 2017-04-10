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
    """
    Computes the overall mean and variance (std) over the dataset: i.e. the mean across all the subjects AND voxels
    This is done instead of computing the mean/var/std across the subjects per voxel since the image is not aligned 
    
    :param X: 
    :param id_X: 
    :param batch_size: 
    :return: 
    """
    print 'Computing Mean/Var/Std of the NONZERO voxels'
    print
    
    n_all = np.product(X.shape[1:]) * id_X.size
    mean_data = np.float(0.)
    var_data = np.float64(0.)

    # it's stupid but we need to find out how many NON-ZERO voxels we will have (otherwise the mean becomes tiny)
    # the only way I can think of right now how to do that is to iterate through everything
    n_nonzero = 0
    for i in xrange(0, id_X.size, batch_size):
        id_batch = np.sort(id_X[i:i+batch_size])
        batch = X[id_batch, ...]
        n_nonzero += (batch != 0).sum(dtype=np.int64)
    print 'Without zeros/with zeros: {}/{}'.format(n_nonzero, n_all)

    # all sums, etc. will be done in float128 to prevent numerical overflow
    for i in xrange(0, id_X.size, batch_size):
        id_batch = np.sort(id_X[i:i+batch_size])
        mean_data += X[id_batch, ...].sum(dtype=np.float128)/np.float128(n_nonzero)
    print 'Mean: {}'.format(mean_data)

    for i in xrange(0, id_X.size, batch_size):
        id_batch = np.sort(id_X[i:i+batch_size])
        var_data += np.sum((X[id_batch, ...] - mean_data)**2., dtype=np.float128)/(np.float128(n_nonzero) - 1)
    print 'Variance: {}'.format(var_data)

    return mean_data, var_data, np.sqrt(var_data), n_nonzero


def standardize_data(X, mean, std, batch_size=50):
    print 'Start z-scaling'
    for i in xrange(0, X.shape[0], batch_size):
        end_id = np.minimum(i + batch_size, X.shape[0])
        x_batch = X[i:end_id, ...]
        # epsilon is added for numerical stability
        X[i:end_id, ...] = (x_batch - mean)/(std + 0.0000000001)


def run(PATH):
    with h5py.File(PATH, 'r+') as hdf5_file:
        dataT1 = hdf5_file['dataT1']
        n = dataT1.shape[0]
        y = dataT1.attrs['labels_subj'].astype(np.int)
        id_train, id_valid, id_test = create_train_validation_test_set(np.arange(n), y)

        dataT1.attrs['id_train'] = id_train
        dataT1.attrs['id_valid'] = id_valid
        dataT1.attrs['id_test'] = id_test

        mean, var, std, n = compute_mean_std_online(dataT1, id_train)
        dataT1.attrs['mean_train'] = mean
        dataT1.attrs['var_train'] = var
        dataT1.attrs['std_train'] = std
        dataT1.attrs['n_nonzero_train'] = n

        standardize_data(dataT1, mean, std)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(**args)
