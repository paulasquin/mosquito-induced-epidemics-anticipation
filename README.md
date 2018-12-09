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

## Without Docker
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

## With Docker
*Working on it*

## Get your .env file to use Google API
You will have to use a Google Developer Account to create a .env file and use the Google Image API.
We only use this API in order to crop our images as a part of the preprocessing. 
We identified an accuracy improvement by using insect-cropped images. 
This way, we eliminate useless information in the mosquito species identification process.

Thus, create a ```.env``` file at the root of the project and write ```GOOGLE_APPLICATION_CREDENTIALS=YOUR_KEY```

## Test
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


## Train models
### Launch an Inception retraining
 ```bash
 python3 -m tests.test_command_classification --retrain
 ```
 
### Launch a From Scratch Neural Network training
```bash
python3 -m image_recognition.from_scratch_neural_network.train
```