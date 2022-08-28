# Transcription_factor_predict
## Abstract
This project provides the core code for the transcription factor prediction project.As you can see, the project contains three py files and three folders.The following describes the meaning of each file one by one.  
1. Dataset folder: This folder contains the original data set.  
2. Dataset_preprocess folder: This folder holds the pre-processed data and the corresponding labels, which are generated by the data_generate.py program.  
3. weight folder: This folder holds our model weights.    
4. Data_generate.py file: This file is used to process the raw data.  
5. Framework_model.py file: This file shows our model construction framework and demonstrates a training example.  
6. Predict_model.py file: This file shows the predictions of our trained model on the test set data.  

## Requirements
* Python3
* Tensorflow>=2.0
* numpy==1.18.5
* sklearn

## Guidance
### Predicting Test Set  
*Step1:  
run the Data_generate.py (Since the Dataset_preprocess folder already holds preprocessed data, this step can be skipped):   python Data_generate.py  
*Step2:  
run the Predict_model.py:  python Predict_model.py  

## Training the model
*Step1:  
run the Data_generate.py (Since the Dataset_preprocess folder already holds preprocessed data, this step can be skipped):   python Data_generate.py  
*Step2:
run the Framework_model.py (You can change the code as appropriate to meet your training requirements):  python Framework_model.py  
