import numpy as np
from itertools import product
from rotate_translate_brain import rotate_brain, shift_brain
import warnings

class ImageTransformer3d(object):

    def __init__(self, data_obj, affine, y_label, id_data, batch_size, type_augmentation=None, num_augmentation_set=1,
                 shuffle=True):

        assert batch_size % num_augmentation_set == 0, 'The number of augmentations per image has to be a multiple of batch_size'

        self.batch_size = batch_size
        self.augment = num_augmentation_set > 1
        self.data = data_obj
        self.y = y_label
        self.id_data = id_data
        self.n_data = self.id_data.size
        self.shuffle = shuffle
        self.num_augmentations = num_augmentation_set
        self.affine = affine
        self.type_augmentation = self.check_augmentation(type_augmentation)

    def check_augmentation(self, type_augmentation):
        if (self.num_augmentations == 1) and type_augmentation:
            warnings.warn('WARNING: Number of augmentations is 1 but type of augmentation is not None. '
                          'It will be set to None!')
            return None
        elif type_augmentation in ['translation', 'rotation', None]:
            return type_augmentation
        else:
            raise RuntimeError("Type of augmentation has to be either None, 'translation' or 'rotation'. "
                               "You chose {}".format(type_augmentation))

    def iter(self):
        id_data = self.id_data

        if self.shuffle:
            np.random.shuffle(id_data)

        if self.augment:
            gen = self.__iter_augment(id_data)
        else:
            gen = self.__iter_no_augment(id_data)

        for data, y, affine in gen:
            yield self.atleast_5d(data), np.array(y), affine

    def __iter_no_augment(self, id_data):
        for offset_batch in xrange(0, self.n_data, self.batch_size):
            end_batch = np.minimum(offset_batch + self.batch_size, self.n_data)
            id_minibatch = np.sort(id_data[offset_batch:end_batch])
            yield (self.data[id_minibatch, ...], np.array(self.y[id_minibatch]), self.affine[id_minibatch, ...])

    def __iter_augment(self, id_data):

        num_orig_images = self.batch_size/self.num_augmentations

        for offset_batch in xrange(0, self.n_data, num_orig_images):
            end_batch = np.minimum(offset_batch + num_orig_images, self.n_data)
            id_minibatch = np.sort(id_data[offset_batch:end_batch])
            data, y, affine = self.data_augmentation(id_minibatch, num_orig_images)

            yield data, y, affine

    def data_augmentation(self, id_minibatch, num_orig_images):
        y = self.y[id_minibatch]
        y = np.concatenate((y, np.repeat(y, self.num_augmentations - 1)))

        data_to_augment = self.atleast_5d(self.data[id_minibatch, ...])
        affine_for_augment = self.affine[id_minibatch, ...]

        if id_minibatch.size != num_orig_images:
            num_orig_images = id_minibatch.size

        batch_size = num_orig_images * self.num_augmentations

        axis_to_transform = np.random.randint(0, 3, size=(num_orig_images, self.num_augmentations - 1))
        
        if self.type_augmentation == 'rotation':        
            angle_to_rotate = np.random.randint(1, 11, size=(num_orig_images, self.num_augmentations - 1))
        elif self.type_augmentation == 'translation':
            shift_to_translate = np.random.randint(1, 20, size=(num_orig_images, self.num_augmentations -1))
        else:
            raise RuntimeError('type_augmentation has to be rotations or translation but {} was given'.format(self.type_augmentation))

        data_new = np.zeros(((batch_size, ) + self.data.shape[1:]), dtype=data_to_augment.dtype)
        affine_new = np.zeros(((batch_size, ) + self.affine.shape[1:]), dtype=affine_for_augment.dtype)

        data_new[:num_orig_images] = data_to_augment
        affine_new[:num_orig_images] = affine_for_augment

        for id_new, (id_img, id_augment) in enumerate(product(xrange(num_orig_images), xrange(self.num_augmentations - 1))):
            id_new += num_orig_images
            max_orig, min_orig = data_to_augment[id_img, ...].max(), data_to_augment[id_img, ...].min()

            if self.type_augmentation == 'rotation':
                data_new[id_new, ..., 0], affine_new[id_new, ...] = rotate_brain(data_to_augment[id_img, ...].squeeze(),
                                                                          angle_to_rotate[id_img, id_augment],
                                                                          axis_to_transform[id_img, id_augment],
                                                                          affine_for_augment[id_img, ...])
            elif self.type_augmentation == 'translation':
                data_new[id_new, ..., 0], affine_new[id_new, ...] = shift_brain(data_to_augment[id_img, ...].squeeze(),
                                                                            shift_to_translate[id_img, id_augment],
                                                                            axis_to_transform[id_img, id_augment],
                                                                            affine_for_augment[id_img, ...])
                
            # http://stackoverflow.com/a/11121189 for the scaling
            # scales data_new to range min_orig, max_orig
            data_new[id_new, ...] = min_orig + (max_orig - min_orig)/\
                                               (data_new[id_new].max() - data_new[id_new].min()) * \
                                               (data_new[id_new] - data_new[id_new].min())

        return data_new, y, affine_new

    @staticmethod
    def atleast_5d(data):
        if len(data.shape) != 5:
            return data[np.newaxis, :, :, :, :]
        return data

if __name__ == '__main__':
    X = np.random.rand(50, 5, 5, 5, 1)
    y = np.random.randint(0, 2, 50)
    affine = np.random.rand(50, 4, 4)
    idx = np.arange(50)
    batch_size = 10
    num_augment = 5

    transformer = ImageTransformer3d(X, affine, y, idx, batch_size, num_augmentation_set=num_augment)