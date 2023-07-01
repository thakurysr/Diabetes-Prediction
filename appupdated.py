# -*- coding: utf-8 -*-
"""
Created on Thu April 24 13:01:17 2023

"""

import pandas as pd
import numpy as np
import streamlit as st
from pickle import dump
from pickle import load
import pickle
from sklearn.preprocessing import MinMaxScaler




scaler= load(open('scaler.sav', 'rb'))

loaded_model = load(open('knn.sav', 'rb'))




# creating a function for Prediction

def db_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    input_data_reshaped=scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return 'Positive '
    else:
      return 'Negative '
  
    
  
def main():
    
    
    # giving a title
    st.title('Model Deployment: KNN Model')
    
    
    # getting the input data from the user
    
    
    number1 = st.number_input('Insert  Number of times pregnant')
    number2 = st.number_input('Insert  Plasma glucose concentration')
    number3 = st.number_input('Insert  Blood Pressure')
    number4 = st.number_input('Insert  Skin Thickness')
    number5 = st.number_input('Insert  Insulin')
    number6 = st.number_input('Insert  BMI')
    number7 = st.number_input('Insert  Diabetes Pedigree Function')
    number8 = st.number_input('Insert  Age')
    
    
#     # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = db_prediction([number1, number2, number3,
                                   number4,number5,number6,number7,number8])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    


