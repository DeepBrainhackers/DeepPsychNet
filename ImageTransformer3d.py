import numpy as np


class ImageTransformer3d(object):

    def __init__(self, data_obj, y_label, id_data, batch_size, num_augment_per_image=5, augment=False, shuffle=True):

        assert batch_size % num_augment_per_image != 0, 'The number of augmentations per image has to be '

        self.batch_size = batch_size
        self.augment = augment
        self.data = data_obj
        self.y = y_label
        self.id_data = id_data
        self.n_data = self.id_data.size
        self.shuffle = shuffle
        self.num_augmentations = num_augment_per_image

    def iter(self):
        id_data = self.id_data

        if self.shuffle:
            np.random.shuffle(self.id_data)

        if self.augment:
            self.__iter_augment(id_data)
        else:
            self.__iter_no_augment(id_data)

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

        #TODO: augment

        return data, y
