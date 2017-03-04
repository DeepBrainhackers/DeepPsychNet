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
            id_data = np.random.shuffle(self.id_data)

        num_iter = self.batch_size
        if self.augment:
            num_iter /= self.num_augmentations

        for offset_batch in xrange(self.n_data, num_iter):
            end_batch = np.minimum(offset_batch, self.n_data)
            id_minibatch = np.sort(id_data[offset_batch:end_batch])

            if self.augment:
                data, y = self.data_augmentation(id_minibatch)
            else:
                data, y = self.data[id_minibatch, ...], self.y[id_minibatch]

            yield (data, y)

    def data_augmentation(self, id_minibatch):
        images_to_load = id_minibatch[:self.batch_size/self.num_augmentations]
        y = self.y[images_to_load]
        data_to_augment = self.data[images_to_load, ...]

        #TODO: augment

        return data, y
