#!/usr/bin/python3
# General toolbox for Room Classification Project
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018

import os
import subprocess
from PIL import Image
import numpy as np

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

def relative_to_absolute_path(path=""):
    """ Send back absolute path if relative was given """
    # Check if not already absolute path
    if path == "":
        path = os.getcwd()
    elif path[0] != "/":
        path = os.getcwd() + "/" + path
    return path


def locate_files(extension, path=FOLDER_PATH, dbName="locate"):
    """ Locate files using .db database. May need sudo to write the database"""
    print("\nCreating the database \"" + dbName + ".db\" for the \"local\" command")
    print("Searching " + extension + " in " + path)
    try:
        cmd = "updatedb -l 0 -o " + dbName + ".db -U " + path + "/"
        print(cmd)
        subprocess.call(["echo 'You may need sudo access' &" + cmd], shell=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print("Might be an error in permission to write the locate database."
              " Try to launch the script with \"sudo python\"")
    cmd = "locate -d " + dbName + ".db " + relative_to_absolute_path(path) + "*" + extension
    print(cmd)
    p = subprocess.Popen([cmd],
                         stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    paths = out.decode('utf-8')
    paths = paths.split('\n')
    paths.pop()  # We delete the last, empty, element
    print("Found " + str(len(paths)) + " elements")
    return paths


def create_folder(path):
    """ Create the folder label if not already exists """
    if not os.path.isdir(path):
        print("Creating folder " + str(path))
        os.mkdir(path)
    return 0


def filepath_image(pathToPly, label, imgFolder, prefix="", suffix="", folderName="", extension="jpg", ):
    """ Generate the path name for the image, default folder name is label.title()
    A suffix can be indicated to indicate, for example, the altitude level """
    if folderName == "":
        folderName = label.title()
    if suffix != "":
        suffix = "-" + suffix
    if prefix != "":
        prefix = "-" + prefix
    sceneId = pathToPly.split("/")[-2] + "_" + pathToPly.split("/")[-3]
    path = os.getcwd() + "/" + imgFolder + "/" + folderName + "/" + sceneId + prefix + suffix + "." + extension
    return path


def get_export_number(modelFolder):
    """ Get the number of the export folder looking at already existing folders
    Handle the presence of '_precisions' at the end of the folder name """

    lesDir = os.listdir(modelFolder)
    lesExport = []
    lesNum = []
    num = 0
    for dir in lesDir:
        if "export_" in dir:
            lesExport.append(dir)
    for i in range(len(lesExport)):
        # Get number of export and add 1 to it
        # If we have an extension in the name
        if lesExport[i][7:].find("_") != -1:
            lesNum.append(int(lesExport[i][7:7 + lesExport[i][7:].find("_")]))
        # If there is not extension
        else:
            lesNum.append(int(lesExport[i][7:]))

    if len(lesNum) != 0:
        num = max(lesNum) + 1

    return num
