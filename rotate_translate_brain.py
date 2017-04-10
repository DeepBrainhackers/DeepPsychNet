# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.ndimage.interpolation import affine_transform


def rotate_brain(image_data, angle, axis, affine_matrix):
    """
    image_data is a 3dimensionl numpy array
    angle is the rotation angle in degrees
    axis is the rotation axis 0,1 or 2"""

    angle = np.deg2rad(angle)

    if axis == 1:
        rotation = np.array([[1, 0, 0, 0],
                             [0, np.cos(angle), -1*np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0, 0, 0, 1]])
    elif axis==2:
        rotation = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                             [0, 1, 0, 0],
                             [-1*np.sin(angle), 0, np.cos(angle), 0],
                             [0, 0, 0, 1]])
    elif axis==3:
        rotation = np.array([[np.cos(angle), -1*np.sin(angle), 0, 0],
                             [np.sin(angle), np.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    else:
        raise RuntimeError('axis has to be 1, 2 or 3. {} was given'.format(axis))

    new_affine = affine_matrix.dot(rotation)

    #offset in the center
    center = 0.5*np.array(image_data.shape)
    offset = center - center.dot(new_affine[:3,:3])
    rotated_data = affine_transform(image_data, new_affine[0:3,0:3:].T, offset=offset, order=1)

    return rotated_data, new_affine


def shift_brain(image_data, shift, axis, affine_matrix):
    """
    image_data:     3d array
    shift:          shift amount in voxels
    axis:           which axis to shift (0, 1 or 2)
    """

    translated_data = np.roll(image_data, shift, axis)

    return translated_data
