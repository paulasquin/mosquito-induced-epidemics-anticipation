#!/usr/bin/python3
# Process the dataset to clean it and augment it
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018

from image_recognition.from_scratch_neural_network.tools import *
import PIL
import sys
import os
import glob

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
LES_AUGMENTATION = ['original', 'width-flip', 'height-flip', 'cwRotate', 'ccwRotate', 'inverse']
DATASET_PATH = FOLDER_PATH + "/preprocessed_dataset"
AUGMENTED_FOLDER = DATASET_PATH + "_augmented"


class Augmentation:
    @staticmethod
    def get_augmentation_path(img_path, augmentation, sub_augmented_path):
        """ Generate the augmented image path, with given original path and augmentation """
        label = img_path.split("/")[-2]

        # computing augmentation path name
        if ".jpg" in img_path:
            augmentation_name = img_path.split("/")[-1].replace(".jpg", "-" + augmentation + ".jpg")
        elif ".jpeg" in img_path:
            augmentation_name = img_path.split("/")[-1].replace(".jpeg", "-" + augmentation + ".jpeg")
        else:
            print("Error with " + img_path)
            raise TypeError("image should only be jpg or jpeg")

        augmentation_path = sub_augmented_path + "/" + label + "/" + augmentation_name
        return augmentation_path

    @staticmethod
    def not_already_augmented(img_path, augmentation, sub_augmented_path):
        """ Return False if asked augmentation already exists or if the file is already an augmentation"""
        aug_path = Augmentation.get_augmentation_path(
            img_path=img_path,
            augmentation=augmentation,
            sub_augmented_path=sub_augmented_path
        )
        augmented = False
        for aug in LES_AUGMENTATION:
            if "-" + aug in img_path:
                augmented = True
        return not (os.path.isfile(aug_path) or augmented)

    @staticmethod
    def augment_image(les_img_path, sub_augmented_path):
        """ Apply augmentation operations defined by LES_AUGMENTATION corresponding to PIL transformations"""
        global LES_AUGMENTATION
        print(', '.join(LES_AUGMENTATION))
        for i, img_path in enumerate(les_img_path):
            with PIL.Image.open(img_path) as img:
                print(str(i + 1) + "/" + str(len(les_img_path)) + " : Augmenting " + img_path.split("/")[-1] + " " * 30)
                for augmentation in LES_AUGMENTATION:
                    if augmentation == 'original' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                         augmentation=augmentation,
                                                                                         sub_augmented_path=sub_augmented_path):
                        img.save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )
                    if augmentation == 'width-flip' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                           augmentation=augmentation,
                                                                                           sub_augmented_path=sub_augmented_path):
                        img.transpose(PIL.Image.FLIP_LEFT_RIGHT).save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )
                    elif augmentation == 'height-flip' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                              augmentation=augmentation,
                                                                                              sub_augmented_path=sub_augmented_path):
                        img.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )
                    elif augmentation == 'cwRotate' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                           augmentation=augmentation,
                                                                                           sub_augmented_path=sub_augmented_path):
                        img.transpose(PIL.Image.ROTATE_270).save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )
                    elif augmentation == 'ccwRotate' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                            augmentation=augmentation,
                                                                                            sub_augmented_path=sub_augmented_path):
                        img.transpose(PIL.Image.ROTATE_90).save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )
                    elif augmentation == 'inverse' and Augmentation.not_already_augmented(img_path=img_path,
                                                                                          augmentation=augmentation,
                                                                                          sub_augmented_path=sub_augmented_path):
                        img.transpose(PIL.Image.ROTATE_180).save(
                            Augmentation.get_augmentation_path(
                                img_path=img_path,
                                augmentation=augmentation,
                                sub_augmented_path=sub_augmented_path)
                        )

    @staticmethod
    def create_augmented_dataset():
        """
        Command augmented dataset creation and subfolders management
        as train, validation and test to create their augmented dataset
        :return:
        """
        create_folder(AUGMENTED_FOLDER)
        les_img_path = []

        if len(glob.glob(DATASET_PATH + "/*/*.jpg")) != 0:
            print("Found label image folders in " + DATASET_PATH)
            print("Augmenting without looking for potential subfolders as train, validation or test")
            label_folders = glob.glob(DATASET_PATH + "/*")
            for label_folder in label_folders:
                print("Augmenting " + label_folder)
                les_img_path = glob.glob(label_folder + "/*.jp*", recursive=True)
                label = label_folder.split("/")[-1]
                create_folder(AUGMENTED_FOLDER + "/" + label)
                Augmentation.augment_image(les_img_path, DATASET_PATH)

        else:
            print("No label folders found in " + DATASET_PATH)
            print("Looking at subfolders")
            for sub_dataset_path in glob.glob(DATASET_PATH + "/*"):
                sub_folder_name = sub_dataset_path.split("/")[-1]
                sub_augmented_path = AUGMENTED_FOLDER + "/" + sub_folder_name
                create_folder(sub_augmented_path)

                # getting images in subfolder
                label_folders = glob.glob(sub_dataset_path + "/*")
                for label_folder in label_folders:
                    print("Augmenting " + label_folder)
                    les_img_path = glob.glob(label_folder + "/*.jp*", recursive=True)
                    label = label_folder.split("/")[-1]
                    create_folder(sub_augmented_path + "/" + label)
                    Augmentation.augment_image(les_img_path, sub_augmented_path)
                print("Done!\n\n")


if __name__ == "__main__":
    Augmentation.create_augmented_dataset()
