
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# Diabetes prediction

def diabetes_prediction(data):
    
    with open("src/Diabetes-Detection/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    data = np.array(data).reshape(1,-1)
    scaled_data = scaler.transform(data)
    
    with open("src/Diabetes-Detection/model.pkl", "rb") as f:
        model = pickle.load(f)
        
    pred = model.predict(scaled_data)
    
    
    return pred

# new_data = np.array([6,148,72,35,0,33.6,0.627,50]).reshape(1, -1)

# pred = diabetes_prediction(new_data)

# if pred==1:
#     print("The patient has diabetes")



#breast cancer prediction
def breast_cancer_prediction(data):
    
    with open("src/Breast-Cancer/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    data = np.array(data).reshape(1,-1)
    scaled_data = scaler.transform(data)
    
    with open("src/Breast-Cancer/model.pkl", "rb") as f:
        model = pickle.load(f)
        
    pred = model.predict(scaled_data)
    
    
    return pred


# new_data = np.array([17.99,	1001.0,	0.262779,	0.3001,	0.14710,	2019.0,	0.665600,	0.7119,	153.40,	0.006193,	0.460100,	0.11890]).reshape(1, -1)

# pred = breast_cancer_prediction(new_data)

# if pred==1:
#     print("The patient has Breast cancer")
# else:
#     print("The patient does not have Breast cancer")

def heart_disease_prediction(data):
    
    with open("src/Heart-Disease/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    data = np.array(data).reshape(1,-1)
    scaled_data = scaler.transform(data)
    
    with open("src/Heart-Disease/heart_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    pred = model.predict(scaled_data)
    
    return pred





def malaria_detection():
    pass