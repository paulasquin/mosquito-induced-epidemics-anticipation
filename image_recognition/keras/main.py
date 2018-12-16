from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import image_recognition.keras.model as keras_model
import image_recognition.keras.bottleneck as keras_bottleneck
from keras.preprocessing.image import ImageDataGenerator
import os

BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = '/'.join(FOLDER_PATH.split("/")[:-1]) + "/preprocessed_dataset_augmented/train"
IMG_HEIGHT = 150
IMG_WIDTH = IMG_HEIGHT
BOTTLENECK_FEATURES_TRAIN_NPY = FOLDER_PATH + "/bottleneck_features_train.npy"
BOTTLENECK_FEATURES_VALIDATION_NPY = FOLDER_PATH + "/bottleneck_features_validation.npy"


def main():
    #  Initialing data generator and defining a validation split
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)

    # Split dataset in train and validation sets
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,  # same directory as training data
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation')

    # Initialing the model
    model = keras_model.Model(IMG_HEIGHT, IMG_WIDTH)
    # Training the model
    model.train_model(BATCH_SIZE, train_generator, validation_generator)


if __name__ == "__main__":
    main()
