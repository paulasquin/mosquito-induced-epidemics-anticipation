import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras import optimizers


class Model:

    def __init__(self, img_height, img_width):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(img_height, img_width, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

        model.add(Conv2D(64, (3, 3), dim_ordering="th"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self.model = model

    def train_model(self, batch_size, train_generator, validation_generator):
        self.model.fit_generator(
            train_generator,
            validation_data=validation_generator
        )
        self.model.save_weights('first_try.h5')  # always save your weights after training or during training
