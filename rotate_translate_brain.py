# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as nib
import numpy as np
import math
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
    rotated_data = affine_transform(image_data, new_affine[0:3,0:3:].T, offset=offset, order=0)
    
    return rotated_data, new_affine


def shift_brain(image_data, shift, axis,affine_matrix):
    
    if axis==1:
        x = shift
        y = 0
        z =0
    elif axis==2:
        x = 0
        y = shift
        z =0
    elif axis ==3:
        x = 0
        y = 0
        z = shift
    
    translation = np.array([[1,0,0,-x],[0,1,0,-y],[0,0,1,-z],[0,0,0,1]])

    new_affine = np.dot(affine_matrix,translation)
    
    #offset in the center
    center = 0.5*np.array(image_data.shape)
    trans = [-x,-y,-z]
    offset = trans+center - center.dot(new_affine[:3,:3])  
    
    translation_data = affine_transform(image_data,new_affine[0:3,0:3:].T,
                                    offset=offset,order=0)
    
    return translation_data, new_affine


if __name__ == "main":
    
    fname = "/home/egill/Dropbox/Brainhack/data/ASD/0050182_T1.nii"
    
    image = nib.load(fname)
    
    image_data = image.get_data()

    angle =50
    axis =1
    shift = -50
    translated_brain, new_affine = shift_brain(image_data,shift,axis,
                                             affine_matrix = image.affine)
    
    
    plt.figure()
    plt.imshow(image_data[100,:,:],interpolation='none')
    plt.show()
    
    plt.figure()
    plt.imshow(translated_brain[100,:,:],interpolation='none')
    plt.show()