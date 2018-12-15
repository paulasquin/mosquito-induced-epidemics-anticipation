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
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,  # same directory as training data
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation')  # set as validation data

    model = keras_model.Model(BOTTLENECK_FEATURES_TRAIN_NPY, BOTTLENECK_FEATURES_VALIDATION_NPY)
    bottleneck = keras_bottleneck.Bottleneck(model, BATCH_SIZE, train_generator, validation_generator)
    model.train_model(batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
