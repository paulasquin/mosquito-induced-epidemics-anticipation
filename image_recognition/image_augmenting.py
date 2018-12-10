#!/usr/bin/python3
# Process the dataset to clean it and augment it
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018

from image_recognition.from_scratch_neural_network.tools import *
import PIL
import sys
import os
import glob

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
LES_AUGMENTATION = ['width-flip', 'height-flip', 'cwRotate', 'ccwRotate', 'inverse']
DATASET_FOLDER = FOLDER_PATH + "/preprocessed_dataset"
AUGMENTED_FOLDER = DATASET_FOLDER + "_augmented"
create_folder(AUGMENTED_FOLDER)


def get_augmentation_path(img_path, augmentation):
    """ Generate the augmented image path, with given original path and augmentation """
    augmentation_path = \
        AUGMENTED_FOLDER \
        + "/" + img_path.split("/")[-2] + "/" \
        + img_path.split("/")[-1].replace(".jpg", "-" + augmentation + ".jpg")
    return augmentation_path


def not_already_augmented(img_path, augmentation):
    """ Return False if asked augmentation already exists or if the file is already an augmentation"""
    aug_path = get_augmentation_path(img_path=img_path, augmentation=augmentation)
    augmented = False
    for aug in LES_AUGMENTATION:
        if "-" + aug in img_path:
            augmented = True
    return not (os.path.isfile(aug_path) or augmented)


def augment_image(les_img_path):
    """ Apply augmentation operations defined by LES_AUGMENTATION corresponding to PIL transformations"""
    global LES_AUGMENTATION
    print(', '.join(LES_AUGMENTATION))
    for i, img_path in enumerate(les_img_path):
        with PIL.Image.open(img_path) as img:
            print(str(i + 1) + "/" + str(len(les_img_path)) + " : Augmenting " + img_path.split("/")[-1] + " "*30, end="\r")
            for augmentation in LES_AUGMENTATION:
                if augmentation == 'width-flip' and not_already_augmented(img_path=img_path, augmentation=augmentation):
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT).save(
                        get_augmentation_path(
                            img_path=img_path,
                            augmentation=augmentation)
                    )
                elif augmentation == 'height-flip' and not_already_augmented(img_path=img_path,
                                                                             augmentation=augmentation):
                    img.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(
                        get_augmentation_path(
                            img_path=img_path,
                            augmentation=augmentation)
                    )
                elif augmentation == 'cwRotate' and not_already_augmented(img_path=img_path, augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_270).save(
                        get_augmentation_path(
                            img_path=img_path,
                            augmentation=augmentation)
                    )
                elif augmentation == 'ccwRotate' and not_already_augmented(img_path=img_path,
                                                                           augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_90).save(
                        get_augmentation_path(
                            img_path=img_path,
                            augmentation=augmentation)
                    )
                elif augmentation == 'inverse' and not_already_augmented(img_path=img_path, augmentation=augmentation):
                    img.transpose(PIL.Image.ROTATE_180).save(
                        get_augmentation_path(
                            img_path=img_path,
                            augmentation=augmentation)
                    )


def main():
    les_img_path = glob.glob(DATASET_FOLDER + '/**/*.jpg', recursive=True)
    print("Augmenting images from " + DATASET_FOLDER + "\n\tto " + AUGMENTED_FOLDER)
    label_folders = glob.glob(DATASET_FOLDER + "/*/")
    for label_folder in label_folders:
        label = label_folder.split("/")[-2]
        create_folder(AUGMENTED_FOLDER + "/" + label)
    augment_image(les_img_path)
    print(" "*50 + "\nDone!")


if __name__ == "__main__":
    main()
