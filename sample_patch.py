import os.path as osp
from glob import glob
from itertools import product

import nibabel as nib
import numpy as np


class PatchSampler(object):

    def __init__(self, folder_data, patch_size=(50, 50, 50), patches_per_img=1, max_amount_zeros=0.4):
        self.folder_data = folder_data
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.max_zeros = max_amount_zeros

    def sample(self, subject_ids, shuffle=True):
        if shuffle:
            np.random.shuffle(subject_ids)

        patches = np.zeros(((subject_ids.size * self.patches_per_img, ) + self.patch_size), dtype=np.float)

        for i_all, (i_img, i_patch) in enumerate(product(xrange(subject_ids.size), xrange(self.patches_per_img))):
            img = nib.load(glob(osp.join(self.folder_data, '*', '*_{}_*.nii.gz'.format(subject_ids[i_img])))[0])
            patches[i_all, :] = self._sample_patch(img.get_data())
        return patches

    def _sample_patch(self, img):
        img_shape = np.array(img.shape)
        perc_zeros = 1.
        while perc_zeros > self.max_zeros:
            x_patch, y_patch, z_patch = self._create_patch_center(img_shape)
            patch = img[(x_patch - self.patch_size[0] // 2): (x_patch + self.patch_size[0] // 2),
                        (y_patch - self.patch_size[1] // 2): (y_patch + self.patch_size[1] // 2),
                        (z_patch - self.patch_size[2] // 2): (z_patch + self.patch_size[2] // 2)]
            perc_zeros = (patch == 0).sum()/float(patch.size)
        return patch

    def _create_patch_center(self, img_shape):
        patch_size = np.array(self.patch_size)
        center_xyz = img_shape // 2
        cov_xyz = np.diag(2 * (img_shape - patch_size))
        patch_coord = np.random.multivariate_normal(mean=center_xyz, cov=cov_xyz).astype(np.int)
        patch_coord[patch_coord <= (patch_size // 2)] = patch_size[patch_coord <= (patch_size // 2)] // 2
        patch_coord[patch_coord > (img_shape - patch_size // 2)] = (img_shape - patch_size // 2)[patch_coord >
                                                                                                 (img_shape -
                                                                                                  patch_size // 2)]
        return patch_coord
