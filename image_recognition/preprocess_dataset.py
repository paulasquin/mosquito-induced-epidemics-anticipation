# preprocess_dataset.py by Mosquito boys project

import glob
import os
from image_recognition.preprocessing import Preprocessing
from image_recognition.inception_classification.utilities import Errors
import sys

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = FOLDER_PATH + "/dataset"


class InitProject():
    def __init__(self):
        print("Initialization of the dataset")
        self.create_preprocessed_dataset()

    @staticmethod
    def check_create_folder(path):
        if not os.path.exists(path):
            print("Creating folder " + path)
            os.mkdir(path)

    def subfolder_preprocessing(self, sub_dataset_path):
        sub_preprocessed_dataset_path = sub_dataset_path.replace("/dataset/", "/preprocessed_dataset/")
        print(sub_preprocessed_dataset_path)
        self.check_create_folder(sub_preprocessed_dataset_path)
        print("\n")
        print("Preprocessing " + str(sub_dataset_path.split("/")[-2:-1]) + " to " + sub_preprocessed_dataset_path)

        #  Getting labels
        labels = [folder.split("/")[-2] for folder in glob.glob(sub_dataset_path + "/*/")]
        print("Found labels: " + str(labels))

        for label in labels:
            print("=== " + label + " ===")
            # Creating label dataset folder
            self.check_create_folder("/".join([sub_preprocessed_dataset_path, label]))

            # getting image names from the folder
            image_names = [file.split("/")[-1] for file in glob.glob(sub_dataset_path + "/" + label + "/*.jp*")]
            for image_name in image_names:
                # generate path for original and preprocessed image
                preprocessed_image_name = "crop_" + image_name
                path_image = "/".join([sub_dataset_path, label, image_name])
                path_preprocessed_image = "/".join([sub_preprocessed_dataset_path, label, preprocessed_image_name])

                # Check if preprocessed was not already done
                if not os.path.exists(path_preprocessed_image):
                    # ask for computing the preprocessed image and to write it at the desired path
                    try:
                        print("+ Saving preprocessed " + path_preprocessed_image)
                        preprocessing = Preprocessing(path_image)
                        preprocessing.save_crop_img(path_preprocessed_image)
                    except Errors.InsectNotFound:
                        print("\tCan't crop the image")
                    except KeyError:
                        print("- not vector found for " + path_preprocessed_image)
                else:
                    print("o Already preprocessed " + image_name)

    def create_preprocessed_dataset(self):
        """
        Command preprocessed dataset creation and subfolders management
        as train, validation and test to create their preprocessed dataset
        :return:
        """
        #  Creating destination folder of preprocessed dataset
        print("Starting creation of the preprocessed dataset")
        preprocessed_dataset_path = FOLDER_PATH + "/preprocessed_dataset"
        self.check_create_folder(preprocessed_dataset_path)

        # Manage if we have plain dataset dataset/[label]/*.jpg] or subfolders dataset as dataset/train/[label]/*.jpg

        if len(glob.glob(DATASET_PATH + "/*/*.jpg")) != 0:
            print("Found label image folders in " + DATASET_PATH)
            print("Augmenting without looking for potential subfolders as train, validation or test")
            self.subfolder_preprocessing(sub_dataset_path=DATASET_PATH)
        else:
            print("No label folders found in " + DATASET_PATH)
            print("Looking at subfolders")
            #   Running preprocessed to train, validation and test folders:
            for sub_dataset_path in glob.glob(DATASET_PATH + "/*"):
                #  Create path name to preprocessed sub folder
                print(sub_dataset_path)
                self.subfolder_preprocessing(sub_dataset_path=sub_dataset_path)


InitProject()
