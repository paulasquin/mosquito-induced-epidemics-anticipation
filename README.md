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
We use this API in order to crop our images. 
We identified an accuracy improvement by using insect-cropped images. 
This way, we eliminate useless information in identifying the mosquito species.

Thus, create a ```.env``` file at the root of the project and write ```GOOGLE_APPLICATION_CREDENTIALS=YOUR_KEY```

 