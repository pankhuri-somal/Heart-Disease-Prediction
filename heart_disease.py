import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import joblib



st.header('Heart Disease Prediction Using Machine Learning')

data = '''Heart Disease Prediction using Machine Learning Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**
'''

st.markdown(data)


st.image('https://hinduja-prod-assets.s3.ap-south-1.amazonaws.com/s3fs-public/2024-03/Heart%20Failure%20and%20Symptoms.jpg')




# Load data
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)

model = joblib.load("heart_disease_pred.pkl")

st.sidebar.header('Select Features to Predict Heart Disease')
st.sidebar.image('https://media.sciencephoto.com/f0/06/12/19/f0061219-800px-wm.jpg')

all_values = []

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

final_value = [all_values]

ans = model.predict(final_value)[0]

import time
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://i.pinimg.com/originals/6d/e1/7f/6de17f9493638040838c12f4c947365b.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)



