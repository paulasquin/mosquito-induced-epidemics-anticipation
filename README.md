# mosquito-induced-epidemics-anticipation
Use Machine Learning to recognise mosquito species from pictures and anticipate risk areas with geolocalised data.

# Introduction
**This project** 
* is led by [Robin Schowb](https://www.linkedin.com/in/robin-schwob-608934100/) and [Paul Asquin](https://www.linkedin.com/in/paulasquin) from the french engineering school CentraleSupélec.  
* is led autonomously and takes place in Machine Learning courses teached by Pr [Fragkiskos D. Malliaros](http://fragkiskos.me/).  
* complete another project, led by Marc Than Van Con, Victor Aubin and Paul Asquin on developing a Web Platform to share field data of encountered mosquitoes : https://github.com/mosquito-boys/mosquito-monitoring 


This project is aim to achieve two objectives:  
  
**1. Use Convolutional Neural Networks to recognise mosquito species.**  
We will use pictures downloaded via Google and study the recognition accuracy differences between Inception retraining and From Scratch Neural Networks.  
  
**2. Use Machine Learning models to anticipate if an area have risks to develop mosquito induced epidemics.**  
We will use Kaggle Datasets to achieve this objective :  
a. [Malarial Mosquito Database](https://www.kaggle.com/jboysen/malaria-mosquito/)  
*Geo-coded Inventory of Anophelines in the Sub-Saharan Africa*  
b. [The fight against malaria](https://www.kaggle.com/teajay/the-fight-against-malaria)  
*Who is dying and being saved from this destructive disease?*

# Installation
## Without Docker
First of all, be sure to have python3 and pip3 installed.
If not, run
```bash
sudo apt-get install python3 python3-pip
```
Then install the project dependencies
```bash
git clone https://github.com/paulasquin/mosquito-induced-epidemics-anticipation.git
cd mosquito-induced-epidemics-anticipation
pip3 install -r requirements.txt
```
If you encounter an issue with the installation of Tensorflow 1.5, it may be linked to Python 3.7. Try to use Python 3.6, or choose to use Docker!

## With Docker
Be sure to have docker and docker-compose installed. 
If you haven't them already, you can follow those tutorials for [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [docker-compose](https://docs.docker.com/compose/install/).  
Note for incoming commands : depending of your installation, you may not need ```sudo```
   
### Build the docker image:
```bash
sudo docker-compose build
```
   
### Up the docker   
```bash
sudo docker-compose up -d
```
This will run the docker container in background and display its name.
Your container's name should be of format ```mosquito-induced-epidemics-anticipation_mosquito_1_######```.   
To make our life easier, run the command:
```bash
export mosquito=YOUR_CONTAINER_NAME
```

### Optional : Run the project with a Google API Key
You may want to run the project already knowing that you don't want to use preprocessing features.
Thus, you don't need the ```.env``` file with its Google API Key.
You can run:
```bash
sudo docker-compose -f docker-compose-without-env.yml build
sudo docker-compose -f docker-compose-without-env.yml up -d
```
   
### Optional : Check for docker container names   
```bash
sudo docker ps --format "{{.Names}}"
```

### Enter in the inner bash   
Open a new terminal and run:
```bash
sudo docker exec -it $mosquito bash
``` 
You can now run commands in the docker container!
If you want to run multiples command at the same time, you can re-run this command in new terminal.   

### Stop the docker container
```bash
sudo docker stop $mosquito
```
You may have to wait up to ~8 seconds before the container stops. 
This is because of the ```sleep infinity``` command that is keeping the container alive.


## Get your .env file to use Google API
You will have to use a Google Developer Account to create a .env file and use the Google Image API.
We only use this API in order to crop our images as a part of the preprocessing. 
We identified an accuracy improvement by using insect-cropped images. 
This way, we eliminate useless information in the mosquito species identification process.

Thus, create a ```.env``` file at the root of the project and write ```GOOGLE_APPLICATION_CREDENTIALS=YOUR_KEY```


# Initialise the project
## Preprocess the dataset

In order to avoid re-preprocessing the whole dataset, we provide the [image_recognition/preprocessed_dataset](image_recognition/preprocessed_dataset) folder. 
Still, if you want to reprocessed the dataset, because you want to have a try or because you add new images, just run:
```bash
python3 -m image_recognition.preprocess_dataset
```
If an image is detected to have already been preprocessed, it will be passed. 
Remove the [image_recognition/preprocessed_dataset](image_recognition/preprocessed_dataset) folder if you want to perform a whole re-preprocessing.

## Augment the dataset
In order to improve our models accuracy, a good way to augment the data is to propose rotated pictures.
Thus, to perform augmentatin as ```width-flip, height-flip, cwRotate, ccwRotate, inverse``` run:
```bash
python3 -m image_recognition.image_augmenting
```


# Train models   
## Run an Inception retraining  
```bash
python3 -m image_recognition.inception_classification.command_classification
```
You will be able to monitor the retraining at ```127.0.0.1:6006```

## Run a From Scratch Neural Network training  
### Run the training  
```bash
python3 -m image_recognition.from_scratch_neural_network.train
```
Please note that you can change the Neural Network parameters, that we call Hyperparameters. More information bellow.  

### Customize hyperparameters
 You can choose your own Neural Networks parameters by editing [image_recognition/from_scratch_neural_network/hyperparams.txt](image_recognition/from_scratch_neural_network/hyperparams.txt)
 
Here are the influences of each parameter: 

**NUM_ITERATION**: number of training iterations.  
\+ If too tall, the model still works but in the end will not learn anymore and performs unnecessary calculations  
\- If it is too small, the model does not have time to reach its actual performance  

**BATCH_SIZE**: The size of the image subpacket used for each train iteration.  
\+ If too large, the necessary calculations and memory explode and the performance of the model decreases by loss of the generalization capacity.  
\- If too small, gradient descents are less representative and performance calculations become noisy.  

**LEARNING_RATE**: learning speed, speed coefficient of the gradient descent.  
\+ If too large, the gradient descent can lead to a divergence.  
\- If too low greatly slows the speed of calculation.  

**SHORTER_DATASET_VALUE** optional: limit the number of images per categories.  
\+ If the number of files used is too large, the demand in memory and calculation explodes.  
\- If this number is too low, the model is lacking data to learn in a representative way.  

**IMG_SIZE**: size in pixels of images, with a native maximum of 500px.  
\+ If too big, the resolution of the images explodes the request in memory and calculation. Similarly, this feature may not be representative of the user application.  
\- If too small, the resolution of the images no longer makes it possible to identify features on the cards.  

**LES_CONV_FILTER_SIZE**: list of the size of the convolution filters, that is to say size of the local area to study. See Figures 4 & 5 of [this page](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)  
\+ If values are too large or if the list is too big, features will become invisible to the model.  
\- If values are too small or the list to small, the model will not be able to clear features effectively.  

**LES_NUM_FILTERS_CONV**: list of the number of filters per convolution layer, that is to say number of neurons per layer.  
\+ If the values are too large, the memory and the necessary computing capacity grow enormously.  
\- If the values are too small, the model is not complex enough and can not learn data.  

**FC_LAYER_SIZE**: size of the last Fully Connected layer (cf figure 9 in [this page](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)).  
\+ If the value is too large, the memory charge explodes.  
\- If the value is too low, the accuracy of the model falls considerably.  

For instance you can have:
```
NUM_ITERATION = 500
BATCH_SIZE = 30
LEARNING_RATE = 0.001
SHORTER_DATASET_VALUE = 0
IMG_SIZE = 256
LES_CONV_FILTER_SIZE = [5, 5, 5, 3, 3, 3]
LES_NUM_FILTERS_CONV = [128, 128, 128, 64, 64, 64]
FC_LAYER_SIZE = 128
```
Note : be sure that **LES_CONV_FILTER_SIZE** and **LES_NUM_FILTERS_CONV** lists have the same lengths.

# Run tests
## Test .env access
Go to the root of the project and run
```bash
python3 -m tests.test_env
```
You should get ```Success!```.


## Test image preprocessing
As explained before, you know that for improving our models accuracy, we have to preprocessed images and crop them to the insect they contains.
To test this features you can run:
```bash
python3 -m tests.test_preprocessing
```
  
  
## Test inception retraining and labelling
You can test inception retraining and inception image labelling:
```bash
python3 -m tests.test_inception_classification [command]
```
```[command]``` can be 
```
--retrain : retrain the inception model
--label [optional path to one or more images to label]
```


# Code explanation
## .env
This file contain the API key used for preprocessing the dataset, for the insect-cropping process.

## Dockerfile
Contains instructions for docker.
 * Use python3 build
 * Link the application folder to the docker container
 * Keep the container alive with sleep command
 
## docker-compose.yml and docker-compose-without-env.yml 
Simplify docker commands by mounting .env and app folders in a modifiable way.

## requirements.txt
Contains the required pip3 modules to install.

## tests
Run tests to check the project function

### test_env.py
Test the .env existence and operation 
 
### test_inception_classification.py
Test image recognition on the inception retraining side.
You can test inception retraining and inception image labelling:
```bash
python3 -m tests.test_inception_classification [command]
```
```[command]``` can be 
```
--retrain : retrain the inception model
--label [optional path to one or more images to label]
```

### test_preprocessing
Test a one image preprocessing : use pic_014 in the tests folder to generate framed and crop pictures.
The output images will also be stored in the tests folder for you to verify them.

## image_recognition/
Folder containing image recognition techniques and dataset processing.

### preprocessing.py
Use Google Vision API to crop images to the "insect" box.

* mosquito_position: Send the image to the API and retrieve mosquito position. 
Return the insect coordinates on a 0-1 scale.
* Compute which pixel form the insect boundaries
* mosquito_cropping: crop the mosquito image to insect boundaries
* mosquito_framing: frame the mosquito in a squared to visualize the identification
* save_crop_img: command and save the image cropping
* save_framed_img: command and save the image framing

### preprocess_dataset.py
Command the dataset preprocessing, and avoid to re-preprocessed already preprocessed pictures.

* check_create_folder: If destination folder doesn't exists, create it.
* create_preprocessed_dataset: command dataset preprocessing for not already preprocessed pictures.

### image_augmenting.py
Augment the dataset for better image recognition performance (in particular for from scratch models).
Perform image rotation augmentations.
* get_augmentation_path: Generate the augmented image path, with given original path and augmentation
* not_already_augmented: Return False if asked augmentation already exists or if the file is already an augmentation
* augment_image: perform augmentations.

### dataset/
The raw dataset

### preprocessed_dataset/
The dataset after having perform the preprocessing

### preprocessed_dataset_augmented/
The dataset after having perform augmentation on the preprocessed_dataset

### inception_classification/
Inception retraining to perform picture classification

#### command_classification.py
Perform retrain, monitoring and predict commands.

* Retrain: Command the retraining agent with indicated parameters and chosen model.
* Tools: Arrange the models in different files for monitoring the model versions 
and get the number of the export folder looking at already existing folders
* Tensorboard: Command Tensorboard monitoring
* Predict: perform label prediction for given images. Make the user able to chose the model or let it automatic. 
* train_and_monitor: run both Retrain and Tensorboard
* label_automatic: run prediction on a file using automatic model folder 

#### retrain.py
Retrain agent provided by Google

#### label_image.py
Model using agent provided by Google

### from_scratch_neural_network/
From scratch neural network creation and training to perform picture classification
