# mosquito-induced-epidemics-anticipation
Use Machine Learning to recognise mosquito species from pictures and anticipate risk areas with geolocalised data.

# Introduction
**This project** 
* is led by Robin Schowb and Paul Asquin form the french engineering school CentraleSupélec.  
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

## With Docker
*Working on it*

## Get your .env file to use Google API
You will have to use a Google Developer Account to create a .env file and use the Google Image API.
We only use this API in order to crop our images as a part of the preprocessing. 
We identified an accuracy improvement by using insect-cropped images. 
This way, we eliminate useless information in the mosquito species identification process.

Thus, create a ```.env``` file at the root of the project and write ```GOOGLE_APPLICATION_CREDENTIALS=YOUR_KEY```

## Test
### Test .env access
Go to the root of the project and run
```bash
python3 -m tests.test_env
```
You should get ```Success!```.


### Test image preprocessing
As explained before, you know that for improving our models accuracy, we have to preprocessed images and crop them to the insect they contains.
To test this features you can run:
```bash
python3 -m tests.test_preprocessing
```

## Initialise the project
### Preprocess the dataset

In order to avoid re-preprocessing the whole dataset, we provide the [image_recognition/preprocessed_dataset](image_recognition/preprocessed_dataset) folder. 
Still, if you want to reprocessed the dataset, because you want to have a try or because you add new images, just run:
```bash
python3 -m image_recognition.preprocess_dataset
```
If an image is detected to have already been preprocessed, it will be passed. 
Remove the [image_recognition/preprocessed_dataset](image_recognition/preprocessed_dataset) folder if you want to perform a whole re-preprocessing.

### Augment the dataset
In order to improve our models accuracy, a good way to augment the data is to propose rotated pictures.
Thus, to perform augmentatin as ```width-flip, height-flip, cwRotate, ccwRotate, inverse``` run:
```bash
python3 -m image_recognition.image_augmenting
```
## Train models   
### Run an Inception retraining  
```bash
python3 -m tests.test_command_classification --retrain
```
 
### Run a From Scratch Neural Network training  
#### Run the training  
```bash
python3 -m image_recognition.from_scratch_neural_network.train
```
Please note that you can change the Neural Network parameters, that we call Hyperparameters. More information bellow.  

#### Customize hyperparameters
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
