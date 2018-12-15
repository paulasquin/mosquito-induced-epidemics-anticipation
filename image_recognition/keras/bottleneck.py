import numpy as np


class Bottleneck:
    def __init__(self, model, batch_size, train_generator, validation_generator):
        self.bottleneck_train_npy = model.bottleneck_features_train_npy
        self.bottleneck_validation_npy = model.bottleneck_features_validation_npy
        self.batch_size = batch_size

        bottleneck_features_train = model.predict_generator(train_generator)
        np.save(str(open(self.bottleneck_train_npy, 'w')), bottleneck_features_train)

        bottleneck_features_validation = model.predict_generator(validation_generator)
        np.save(str(open(self.bottleneck_validation_npy, 'w')), bottleneck_features_validation)
