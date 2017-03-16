import numpy as np
import nibabel as ni
import os


def to_native_coords(y_file, affine_rand_rot, mm_mni):
    """
     :input: y_file (filename of the warped file
            affine_rand_rot (the affine matrix used for random rotation)
            mm_mni (the MNI coordinate desired)

     /// Code adapted from John Ashburner (private comm)
     :return: coordinates in native space.
    """

    V = ni.load(y_file)
    iM = np.linalg.inv(V.affine).dot(affine_rand_rot)
    warp_field = V.get_data()

    mm_mni.shape = (3,1) # Ensure column vector
    vx_mni = np.dot(iM[:3, :], np.vstack((mm_mni, 1)))

    x, y, z = list(vx_mni.flatten().astype(int))

    X = warp_field[..., 0]
    Y = warp_field[..., 1]
    Z = warp_field[..., 2]

    # float converts from memmap object
    mm_native = [float(X[x, y, z]), float(Y[x, y, z]), float(Z[x, y, z])]

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
