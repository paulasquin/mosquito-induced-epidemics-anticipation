from image_recognition.inception_classification.preprocessing import Preprocessing

image_path = 'image_recognition/dataset/training/anopheles/pic_014.jpg'
image_framed = "tests/pic_014_framed.jpg"
image_croped = "tests/pic_014_cropped.jpg"
preprocessing = Preprocessing(image_path)

framed = preprocessing.save_framed_img(image_framed)
crop_resized = preprocessing.save_crop_img(image_croped)

print("Images have been written to \n" + image_framed + "\n" + image_croped)
