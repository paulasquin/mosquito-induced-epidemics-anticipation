from image_recognition.preprocessing import Preprocessing
import os

image_path = FOLDER_PATH = os.path.dirname(os.path.abspath(__file__)) + '/pic_014.jpg'
image_framed = image_path.replace(".jpg", "_framed.jpg")
image_croped = image_path.replace(".jpg", "_cropped.jpg")
preprocessing = Preprocessing(image_path)

framed = preprocessing.save_framed_img(image_framed)
crop_resized = preprocessing.save_crop_img(image_croped)

print("Images have been written to \n" + image_framed + "\n" + image_croped)
