import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
# import matplotlib.pyplot as plt
import math
import cv2
import os

BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = '/'.join(FOLDER_PATH.split("/")[:-1]) + "/preprocessed_dataset_augmented/train"
BOTTLENECK_FEATURES_TRAIN_NPY = FOLDER_PATH + "/bottleneck_features_train.npy"
BOTTLENECK_FEATURES_VALIDATION_NPY = FOLDER_PATH + "/bottleneck_features_validation.npy"
CLASS_INDICES_NPY = FOLDER_PATH + "/class_indices.npy"
TOP_MODEL_WEIGHTS_PATH = FOLDER_PATH + '/bottleneck_fc_model.h5'

IMG_HEIGHT = 150
IMG_WIDTH = IMG_HEIGHT

img_width, img_height = 150, 150

# number of epochs to train top model
epochs = 50


def save_bottlebeck_features():
    print("Saving bottleneck features...")
    # build the VGG16 network
    print("Downloading VGG16")
    model = applications.VGG16(include_top=False, weights='imagenet')
    print("Splitting dataset in train and validation with a " + str(VALIDATION_SPLIT) + " ratio for validation")
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)

    # Split dataset in train and validation sets
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        subset='training')

    print(len(train_generator.filenames))
    print(train_generator.class_indices)
    print(len(train_generator.class_indices))

    nb_train_samples = len(train_generator.filenames)
    num_classes = len(train_generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / BATCH_SIZE))
    print("Computing bottleneck features train")
    bottleneck_features_train = model.predict_generator(train_generator, predict_size_train)
    print("Saving bottleneck features train")
    np.save(BOTTLENECK_FEATURES_TRAIN_NPY, bottleneck_features_train)

    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,  # same directory as training data
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        subset='validation')

    nb_validation_samples = len(validation_generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / BATCH_SIZE))
    print("Computing bottleneck features validation")
    bottleneck_features_validation = model.predict_generator(validation_generator, predict_size_validation)
    print("Saving bottleneck features validation")
    np.save(BOTTLENECK_FEATURES_VALIDATION_NPY, bottleneck_features_validation)


def train_top_model():
    print("Train top model")
    datagen_top = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)
    generator_top = datagen_top.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='training')

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save(CLASS_INDICES_NPY, generator_top.class_indices)

    # load the bottleneck features saved earlier
    print("Load bottleneck features train")
    train_data = np.load(BOTTLENECK_FEATURES_TRAIN_NPY)

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        subset='validation')

    nb_validation_samples = len(generator_top.filenames)
    print("Load bottleneck features validation")
    validation_data = np.load(BOTTLENECK_FEATURES_VALIDATION_NPY)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    print("Building the model")
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    print("Compiling the model")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Fitting the model")
    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(validation_data, validation_labels)
    )

    print("Saving the model")
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)

    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=BATCH_SIZE, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    # plt.figure(1)
    #
    # # summarize history for accuracy
    #
    # plt.subplot(211)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    #
    # # summarize history for loss
    #
    # plt.subplot(212)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def predict(image_path):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load(CLASS_INDICES_NPY).item()
    num_classes = len(class_dictionary)

    orig = cv2.imread(image_path)
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255
    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(TOP_MODEL_WEIGHTS_PATH)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))

    # # display the predictions with the image
    # cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
    #             cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
    #
    # cv2.imshow("Classification", orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    save_bottlebeck_features()
    train_top_model()
    # predict()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
