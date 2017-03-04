import numpy as np
from rotate_translate_brain import rotate_brain

class ImageTransformer3d(object):

    def __init__(self, data_obj, affine, y_label, id_data, batch_size, num_augment_per_image=5, augment=False,
                 shuffle=True):

        assert batch_size % num_augment_per_image != 0, 'The number of augmentations per image has to be '

        self.batch_size = batch_size
        self.augment = augment
        self.data = data_obj
        self.y = y_label
        self.id_data = id_data
        self.n_data = self.id_data.size
        self.shuffle = shuffle
        self.num_augmentations = num_augment_per_image
        self.affine = affine

    def iter(self):
        id_data = self.id_data

        if self.shuffle:
            np.random.shuffle(self.id_data)

        if self.augment:
            yield self.__iter_augment(id_data)
        else:
            yield self.__iter_no_augment(id_data)

    def __iter_no_augment(self, id_data):
        for offset_batch in xrange(self.n_data, self.batch_size):
            end_batch = np.minimum(offset_batch, self.n_data)
            id_minibatch = np.sort(id_data[offset_batch:end_batch])
            yield (self.data[id_minibatch, ...], np.array(self.y[id_minibatch]))

    def __iter_augment(self, id_data):

        for offset_batch in xrange(self.n_data, self.batch_size/self.num_augmentations):
            end_batch = np.minimum(offset_batch, self.n_data)
            id_minibatch = np.sort(id_data[offset_batch:end_batch])
            yield self.data_augmentation(id_minibatch)

    def data_augmentation(self, id_minibatch):
        images_used = self.batch_size/self.num_augmentations
        images_to_load = id_minibatch[:images_used]
        y = self.y[images_to_load]
        y = np.concatenate([y for _ in xrange(images_used)])

        data_to_augment = self.data[images_to_load, ...]
        affine_for_augment = self.affine[images_to_load]

        axis_to_rotate = np.random.randint(1, 4, self.batch_size)
        angle_to_rotate = np.random.randint(0, 181, self.batch_size)

        data = np.zeros(((self.batch_size, ) + data_to_augment.shape[1:]), dtype=data_to_augment.dtype)

        # TODO: How to handle the indices? How to handle the affine matrices
        for i in xrange(images_to_load.size):
            data[i] = rotate_brain(data_to_augment)

        # data = [rotate_brain(data_to_augment[i], angle_to_rotate[i], axis_to_rotate[i],
        #                      affine_matrix=affine_for_augment[i]) ]

        return data, y
