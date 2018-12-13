import image_recognition.keras.model as keras_model
import image_recognition.keras.bottleneck as keras_bottleneck
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

BATCH_SIZE = 16


def main():
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = keras_model.Model()
    bottleneck = keras_bottleneck.Bottleneck(datagen=datagen, model=model.model, batch_size=BATCH_SIZE)
    model.train_model(batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
