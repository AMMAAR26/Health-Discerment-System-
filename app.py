
import streamlit as st
from streamlit_extras import add_vertical_space
import streamlit.components.v1 as components
from annotated_text import annotated_text

import tensorflow as tf
from tensorflow import keras

from keras.models import load_model
from PIL import Image
import numpy as np


from prediction_pipeline import diabetes_prediction, breast_cancer_prediction, heart_disease_prediction



#st.set_page_config(layout='wide')

import pandas as pd

import json



with st.sidebar:
    st.title("Health Discerment System")
    

    diseases = ["Diabetes Prediction","Breast Cancer","Heart Disease Prediction","Malaria Detection", "Pneumonia Detection", "Brain Tumour Detection"]

    


    selected_diseases = st.selectbox("Select Diseases to Predict", diseases)


    
if selected_diseases == "Diabetes Prediction":
    
    st.title("DIABETES PREDICTION")
    
    # Input fields for user to input data
    pregnancies = st.number_input("Number of Pregnancies", 0, 17, 1)
    glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", 0, 200, 100)
    blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 0, 122, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
    insulin = st.number_input("Insulin Level (mu U/mL)", 0, 846, 79)
    bmi = st.number_input("Body Mass Index (BMI)", 0.0, 67.1, 30.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
    age = st.number_input("Age (years)", 21, 81, 25)
    
    if st.button("Predict"):
        prediction = diabetes_prediction(data=[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age])
        
        if prediction==1:
            st.error("The patient has diabetes")
        else:
            st.success("The patient does not have diabetes")


if selected_diseases == "Breast Cancer":
    
    st.title("BREAST CANCER PREDICTION")
    
    # Input fields for user to input data
    radius_mean = st.number_input("Radius Mean", 6.981, 28.11, 14.127)
    area_mean = st.number_input("Area Mean", 143.5, 2501.0, 654.889)
    compactness_mean = st.number_input("Compactness Mean", 0.019, 0.345, 0.104)
    concavity_mean = st.number_input("Concavity Mean", 0.0, 0.427, 0.089)
    concave_points_mean = st.number_input("Concave Points Mean", 0.0, 0.201, 0.049)
    area_worst = st.number_input("Area Worst", 185.200000, value=686.500000)
    compactness_worst = st.number_input("Compactness Worst",0.027290, value=0.211900)
    concavity_worst = st.number_input("Concavity Worst",0.000000, value=0.226700)
    area_se = st.number_input("Area Se", 6.802000, value=24.530000)
    fractal_dimension_se = st.number_input("Fractal Dimension Mean", 0.05, 0.097, 0.062)
    symmetry_worst = st.number_input("Symmetry Worst", 0.106, 0.304, 0.181)
    fractal_dimension_worst = st.number_input("Fractal_Dimension_Worst", 0.055040, value=0.080040)

    if st.button("Predict"):
        prediction = breast_cancer_prediction(data=[radius_mean,area_mean,compactness_mean,concavity_mean,concave_points_mean,area_worst,compactness_worst,concavity_worst,area_se,fractal_dimension_se,symmetry_worst,fractal_dimension_worst])
        
        if prediction==1:
            st.error("The patient has Breast Cancer")
        else:
            st.success("The patient does not have Breast Cancer")



if selected_diseases == "Heart Disease Prediction":
    
    st.title("HEART DISEASE PREDICTION")
    
    # Input fields for user to input data
    age = st.number_input("Age", 29, 77, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    Cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 126, 564, 240)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    RestingECG = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Probable or Definite Left Ventricular Hypertrophy"])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", 71, 202, 150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    Oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.2, 2.0)
    ST_Slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    #converting categorical into numerical
    sex = 1 if sex == "Male" else 0
    
    if ChestPainType == "Typical Angina":
        ChestPainType = 0
    elif ChestPainType == "Atypical Angina":
        ChestPainType = 1
    elif ChestPainType == "Non-anginal Pain":
        ChestPainType = 2
    else:
        ChestPainType = 3
        
        
    if FastingBS == "True":
        FastingBS = 1
    else:
        FastingBS = 0
        
    if RestingECG == "Normal":
        RestingECG = 0
    elif RestingECG == "ST-T wave abnormality":
        RestingECG = 1
    else:
        RestingECG = 2
        
    if ExerciseAngina == "Yes":
        ExerciseAngina = 1
    else:
        ExerciseAngina = 0
        
    if ST_Slope == "Upsloping":
        ST_Slope = 0
    elif ST_Slope == "Flat":
        ST_Slope = 1
    else:
        ST_Slope = 2
        
        
    if st.button("Predict"):
        prediction = heart_disease_prediction(data=[age,sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
        
        if prediction==1:
            st.error("The patient has Heart Disease")
        else:
            st.success("The patient does not have Heart Disease")
    
    

if selected_diseases == "Malaria Detection":
    
    st.title("MALARIA DISEASE DETECTION")
    
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    model = load_model('src/Malaria-Detection/malaria.h5')

    
    def preprocess_image(image_file):
        img = Image.open(image_file)
        img = img.resize((128, 128))  # Resize the image to match the input size of the model
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    
    def predict_malaria(image_file):
        img_array = preprocess_image(image_file)
        prediction = model.predict(img_array)
        return prediction
        

        # When the user clicks the predict button
    if st.button("Predict"):
        # Make prediction
        prediction = predict_malaria(uploaded_file)
        # Display prediction
        if prediction[0][0] > 0.5:
            st.success("The image does not contain malaria parasites.")
        else:
            st.error("The image contains malaria parasites.")
                
                
                
 
                
                
if selected_diseases == "Pneumonia Detection":
    
    st.title("PNEUMONIA DISEASE DETECTION")
    

    # Load the pre-trained model
    model = load_model('src/Pneumonia-Detection/pneumonia_detection.h5')

    # Function to preprocess the image
    def preprocess_image(image_file):
        img = Image.open(image_file)
        img = img.resize((150, 150))  # Resize the image to match the input size of the model
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Function to make prediction
    def predict_pneumonia(image_file):
        img_array = preprocess_image(image_file)
        prediction = model.predict(img_array)
        return prediction

   

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # When the user uploads an image and clicks the predict button
    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # When the user clicks the predict button
        if st.button("Predict"):
            # Make prediction
            prediction = predict_pneumonia(uploaded_file)
            # Display prediction
            if prediction[0][0] > 0.5:
                st.error("The image indicates pneumonia.")
            else:
                st.success("The image is normal.")



if selected_diseases == "Brain Tumour Detection":
    
    st.title("BRAIN TUMOUR DETECTION")
    
    st.write("Working on it, coming soon!")