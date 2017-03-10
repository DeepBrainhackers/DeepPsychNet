import os
import os.path as osp
from glob import glob
from sys import stdout
from time import time

import h5py
import nibabel as nib
import numpy as np


def ensure_folder(folder_dir):
    if not osp.exists(folder_dir):
        os.makedirs(folder_dir)


def create_hdf5_file(directory_data, sub_folders=('ASD', 'CON'), save_directory='hdf5_data', save_path='data.hdf5',
                     nifit_ending='*T1_shft_res.nii'):
    ensure_folder(save_directory)

    path_niftis, n_subj, subj_id = get_niftis(directory_data, sub_folders, nifti_ending=nifit_ending)
    labels_subj = np.concatenate((np.ones(n_subj[0]), np.zeros(n_subj[1])))

    affine_dtype, affine_shape, data_shape = get_affine_shape(path_niftis)
    n_total_subj = n_subj.sum()

    with h5py.File(osp.join(save_directory, save_path), 'w') as hdf5_file:
        dataT1 = hdf5_file.create_dataset('dataT1', shape=(n_total_subj, ) + data_shape + (1,), dtype=np.float32)
        dataAffine = hdf5_file.create_dataset('dataAffine', shape=(n_total_subj, ) + affine_shape, dtype=affine_dtype)

        for id_subj in xrange(len(path_niftis)):
            t1 = time()
            nifti_img = nib.load(path_niftis[id_subj])
            affine = nifti_img.affine
            nifti_img = np.array(nifti_img.get_data(), dtype=np.float32)[:, :, :, np.newaxis]
            dataT1[id_subj, :, :, :] = nifti_img
            dataAffine[id_subj, :, :] = affine
            t2 = time()
            stdout.write('\r{}/{}: {:0.2}sec'.format(id_subj + 1, len(path_niftis), t2-t1))
            stdout.flush()

        print
        dataT1.attrs['labels_subj'] = labels_subj
        dataT1.attrs['subj_id'] = subj_id


def get_affine_shape(path_niftis):
    tmp = nib.load(path_niftis[0])
    data_shape = tmp.shape
    affine = tmp.affine
    affine_shape = affine.shape
    affine_dtype = affine.dtype
    return affine_dtype, affine_shape, data_shape


def get_niftis(directory_data, sub_folders, nifti_ending='*T1_shft_res.nii'):
    if not isinstance(sub_folders, tuple):
        sub_folders = (sub_folders)

    num_subj = np.zeros(len(sub_folders), dtype=np.int)
    path_def = []
    subj_id = []
    for i_sub_folder, sub_folder in enumerate(sub_folders):
        tmp = sorted(glob(osp.join(directory_data, sub_folder, nifti_ending)))
        subj_id += [osp.basename(sub).split('_')[0] for sub in tmp]
        num_subj[i_sub_folder] = len(tmp)
        path_def += tmp


    return path_def, num_subj, np.array(subj_id)


def run():
    directory_data = '/media/paul/kaggle/dataProcessed'
    sub_folders = ('ASD', 'CON')
    save_folder = '/media/paul/kaggle/dataHDF5'
    save_file = 'abide.hdf5'
    img_pattern = '*T1_shft_res.nii'
    create_hdf5_file(directory_data, sub_folders, save_directory=save_folder, save_path=save_file,
                     nifit_ending=img_pattern)


if __name__ == '__main__':
    run()
