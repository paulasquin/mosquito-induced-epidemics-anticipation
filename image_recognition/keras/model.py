import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class Model:

    def __init__(self):
        self.train_data = np.load(open('bottleneck_features_train.npy'))
        # the features were saved in order, so recreating the labels is easy
        self.train_labels = np.array([0] * 1000 + [1] * 1000)

        self.validation_data = np.load(open('bottleneck_features_validation.npy'))
        self.validation_labels = np.array([0] * 400 + [1] * 400)
        self.__model = Sequential()
        self.__model.add(Flatten(input_shape=self.train_data.shape[1:]))
        self.__model.add(Dense(256, activation='relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(1, activation='sigmoid'))

    @property
    def model(self):
        return self.__model

    def train_model(self, batch_size):
        if self.__model is None:
            raise NameError("Model have to be created first")

        print("Compiling model...")
        self.__model.compile(optimizer='rmsprop',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

        print("Fitting model...")
        self.__model.fit(self.train_data, self.train_labels,
                         epochs=50,
                         batch_size=batch_size,
                         validation_data=(self.validation_data, self.validation_labels))

        print("Saving model...")
        self.__model.save_weights('bottleneck_fc_model.h5')
        print("Done!")
