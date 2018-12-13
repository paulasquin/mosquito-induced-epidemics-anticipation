import numpy as np


class Bottleneck:
    def __init__(self, datagen, model, batch_size=16):
        self.batch_size = batch_size
        generator = datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode=None,  # this means our generator will only yield batches of data, no labels
            shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
        # the predict_generator method returns the output of a model, given
        # a generator that yields batches of numpy data
        nb_train = len(generator.filenames)
        bottleneck_features_train = model.predict_generator(generator, int(nb_train * 0.8))
        # save the output as a Numpy array
        np.save(str(open('bottleneck_features_train.npy', 'w')), bottleneck_features_train)

        generator = datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_validation = model.predict_generator(generator, int(nb_train * 0.2))
        np.save(str(open('bottleneck_features_validation.npy', 'w')), bottleneck_features_validation)
