import numpy as np
import nibabel as nib
import os


def to_native_coords(y_file, affine_rand_rot, mm_mni):
    """
     :input: y_file (filename of the warped file
            affine_rand_rot (the affine matrix used for random rotation)
            mm_mni (the MNI coordinate desired)

     /// Code adapted from John Ashburner (private comm)
     :return: coordinates in native space.
    """
    if mm_mni.ndim == 1:
        mm_mni = mm_mni[:, np.newaxis]

    V = nib.load(y_file)
    iM = np.linalg.inv(V.affine.dot(affine_rand_rot))
    warp_field = np.array(V.get_data(), dtype=np.float)

    vx_mni = np.dot(iM[:3, :], np.vstack((mm_mni, np.ones(mm_mni.shape[1])))).squeeze()

    x, y, z = vx_mni[0], vx_mni[1], vx_mni[2]

    X = warp_field[..., 0]
    Y = warp_field[..., 1]
    Z = warp_field[..., 2]

    mm_native = np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]])
    if mm_native.ndim == 1:
        mm_native = mm_native[:, np.newaxis]

    return mm_native

if __name__ == '__main__':

    datapath = '/home/rajat/Dropbox/Brainhack/data/CON'
    subject_id = '0051134'

    # Py -> warp field y_XXX.nii file
    Py = 'y_' + subject_id + '_T1.nii'
    Py = os.path.join(datapath, Py)

    mni = np.array([0, 0, 0])
    native_coords = to_native_coords(Py, np.eye(4), mni)

    print(native_coords)
