# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as nib
import numpy as np
import scipy

def rotate_brain(image_data, angle, axis):
    """
    image_data is a 3dimensionl numpy array
    angle is the rotation angle in degrees
    axis is the rotation axis 0,1 or 2"""
    
    
    
    all_axes = {0: (1,2), 1: (0,2), 2: (0,1)}    
    
    rotated_data = scipy.ndimage.interpolation.rotate(image_data,angle,
                                                      all_axes[axis],
                                             reshape=False, order=0)

    
    return rotated_data


def shift_brain(3d_image_data, angle, axis):
    
    
    
    pass


if __name__ == "main":
    
    fname = "/home/egill/Dropbox/Brainhack/data/ASD/0050182_T1.nii"
    
    image = nib.load(fname)
    
    image_data = image.get_data()


    rotated_brain = rotate_brain(image_data,angle=30,axis=0)
    
    
    plt.figure()
    plt.imshow(image_data[100,:,:],interpolation='none')
    plt.show()
    
    plt.figure()
    plt.imshow(rotated_brain[100,:,:],interpolation='none')
    plt.show()