from image_recognition.preprocessing import Preprocessing
import os

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

les_image_path = [FOLDER_PATH + '/anopheles.jpg',
                  FOLDER_PATH + '/culex.jpg']
for image_path in les_image_path:
    image_framed = image_path.replace(".jpg", "_framed.jpg")
    image_cropped = image_path.replace(".jpg", "_cropped.jpg")
    preprocessing = Preprocessing(image_path)

    framed = preprocessing.save_framed_img(image_framed)
    crop_resized = preprocessing.save_crop_img(image_cropped)

    print("Images have been written to \n" + image_framed + "\n" + image_cropped)
