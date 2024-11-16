import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#1 loading the trained model
model = tf.keras.models.load_model('model.h5')

#load encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


#streamlit app

st.title(" Customer Churn Prediction")

#user input
Geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age',18,92)
Balance = st.number_input('Balance')
Credit_score = st.number_input('Credit Score')
Estimated_salary = st.number_input('Estimated Salary')
Tenure = st.slider('Tenure',0,10)
Num_of_products = st.slider("Number of Products",1,4)
Has_cr_card = st.selectbox('Has Credit Card',[0,1])
Is_active_member = st.selectbox('Is Active Member',[0,1])

#preparing input data

input_data = pd.DataFrame({
    'CreditScore': [Credit_score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [Num_of_products],
    'HasCrCard': [Has_cr_card],
    'IsActiveMember': [Is_active_member],
    'EstimatedSalary': [Estimated_salary]
})

#encoding geography using one hot encoder

geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded,columns= onehot_encoder_geo.get_feature_names_out(['Geography']))

#appending encoded geo to input data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

#scale the data

input_scaled = scaler.transform(input_data)

#prediction

prediction = model.predict(input_scaled)

prediction_prob = prediction[0][0]

st.write(f'Churn-probablity : {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write(" customer is likely to churn")
else:
     st.write("customer is not likely to churn")