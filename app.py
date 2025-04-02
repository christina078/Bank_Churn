#importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# Load Model and Scaler
with open('xgmodel.pkl', 'rb') as file:
    xgmodel = pickle.load(file)

with open('scl.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.header('BANK CUSTOMER CHURN PREDICTION')
with st.sidebar:
    st.image("https://dezyre.gumlet.io/images/blog/machine-learning-case-studies/Customer_Churn_Prediction_Machine_Learning_Case_Study.png?w=940&dpr=1.3")
    st.subheader('About this project')
    st.write('This application predicts whether a bank customer churns or not.')

# User inputs
CreditScore = st.number_input('Enter Credit Score:', min_value=0, max_value=1000, value=600)
Geography = st.selectbox('Choose the region:', ('France', 'Spain', 'Germany'))
Gender = st.selectbox('Gender:', ('Male', 'Female'))
Age = st.slider('Enter the Age:', 18, 100, 30)
Tenure = st.number_input('Enter Tenure:', min_value=0, max_value=10, value=5)
NumOfProducts = st.slider('No of Products purchased:', 1, 5, 2)
HasCrCard = st.selectbox('Does the customer have Credit card?', ('Yes', 'No'))
IsActiveMember = st.selectbox('Is the customer an active member?', ('Yes', 'No'))
EstimatedSalary_f = st.selectbox('Estimated salary', ('0-50K', '50K-100K', '100K-150K', '150K-200K'))
Complain = st.selectbox('Has complained?', ('Yes', 'No'))
Satisfaction_Score = st.selectbox("Satisfaction Score", (1, 2, 3, 4, 5))
Cardtype = st.selectbox('Card Type:', ('DIAMOND', 'SILVER', 'GOLD', 'PLATINUM'))
Point_earned = st.number_input('Points Earned:', min_value=0, value=1000)
Balance_cat = st.selectbox('Balance Category', ('Zero', '0-50K', '50K+'))

# Encoding categorical variables
EstimatedSalary_0to50K, EstimatedSalary_50Kto100K, EstimatedSalary_100Kto150K, EstimatedSalary_150Kto200K = 0, 0, 0, 0
if EstimatedSalary_f == '0-50K':
    EstimatedSalary_0to50K = 1
elif EstimatedSalary_f == '50K-100K':
    EstimatedSalary_50Kto100K = 1
elif EstimatedSalary_f == '100K-150K':
    EstimatedSalary_100Kto150K = 1
else:
    EstimatedSalary_150Kto200K = 1

Gender = 0 if Gender == 'Female' else 1
HasCrCard = 1 if HasCrCard == 'Yes' else 0
Complain = 1 if Complain == 'Yes' else 0
IsActiveMember = 1 if IsActiveMember == 'Yes' else 0

Geography_France, Geography_Germany, Geography_Spain = 0, 0, 0
if Geography == 'France':
    Geography_France = 1
elif Geography == 'Germany':
    Geography_Germany = 1
else:
    Geography_Spain = 1

Card_Type_Gold, Card_Type_Platinum, Card_Type_Silver, Card_Type_Diamond = 0, 0, 0, 0
if Cardtype == 'GOLD':
    Card_Type_Gold = 1
elif Cardtype == 'DIAMOND':
    Card_Type_Diamond = 1
elif Cardtype == 'PLATINUM':
    Card_Type_Platinum = 1
else:
    Card_Type_Silver = 1

Balance_Category_Zero, Balance_Category_0to50K, Balance_Category_50Kabove = 0, 0, 0
if Balance_cat == 'Zero':
    Balance_Category_Zero = 1
elif Balance_cat == '0-50K':
    Balance_Category_0to50K = 1
else:
    Balance_Category_50Kabove = 1

#scaling
numerical_features = np.array([[CreditScore, Age, Tenure, NumOfProducts, Satisfaction_Score, Point_earned]])
scaled_features = scaler.transform(numerical_features)[0] 

CreditScore, Age, Tenure, NumOfProducts, Satisfaction_Score, Point_earned = scaled_features


features = [CreditScore, Gender, Age, Tenure, NumOfProducts, HasCrCard, IsActiveMember, Complain, 
            Satisfaction_Score, Point_earned, Balance_Category_0to50K, Balance_Category_50Kabove, 
            Balance_Category_Zero, Geography_France, Geography_Germany, Geography_Spain, 
            Card_Type_Gold, Card_Type_Platinum, Card_Type_Silver, Card_Type_Diamond, 
            EstimatedSalary_0to50K, EstimatedSalary_100Kto150K, EstimatedSalary_150Kto200K, 
            EstimatedSalary_50Kto100K]

# Prediction
if st.button('Predict'):
    input_data = np.array(features).reshape(1, -1)
    prediction = xgmodel.predict(input_data)[0]
    
    # Displaying the Result
    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")

