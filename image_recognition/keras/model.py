import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras import optimizers

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'


class Model:

    def __init__(self, bottleneck_features_train_npy, bottleneck_features_validation_npy):
        self.bottleneck_features_train_npy = bottleneck_features_train_npy
        self.bottleneck_features_validation_npy = bottleneck_features_validation_npy

        self.model = applications.VGG16(weights='imagenet', include_top=False)
        print('Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=self.model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(top_model_weights_path)

        # add the model on top of the convolutional base
        self.model.add(top_model)

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in self.model.layers[:25]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                           metrics=['accuracy'])

    def train_model(self, batch_size):
        train_data = np.load(open(self.bottleneck_features_train_npy))
        validation_data = np.load(open(self.bottleneck_features_validation_npy))

        self.model.fit(train_data,
                       epochs=50,
                       batch_size=batch_size,
                       validation_data=validation_data)
        self.model.save_weights('bottleneck_fc_model.h5')
