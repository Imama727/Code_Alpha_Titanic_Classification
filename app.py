import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl



# Load the pre-trained model
model = pkl.load(open('model.pkl', 'rb'))

# Define the Streamlit app
st.title('Titanic Survival Prediction')

# Create input fields for user input
st.sidebar.header('User Input Features')

def user_input_features():
    pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    sex = st.sidebar.selectbox('Gender', ['male', 'female'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    sibsp = st.sidebar.slider('SibSp', 0, 10, 0)
    parch = st.sidebar.slider('Parch', 0, 10, 0)
    fare = st.sidebar.text_input('Fare', '10.0')  # Moved Fare to the sidebar
    embarked = st.sidebar.selectbox('Embarked', ['S', 'C', 'Q'])
    
    # Convert Fare to float
    try:
        fare = float(fare)
    except ValueError:
        fare = 10.00  # Default value in case of invalid input
    
    data = {'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Convert categorical features
df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Ensure the input DataFrame matches the training feature order
expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[expected_columns]

# Add Predict button
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        
        # Display results
        st.write('### Prediction')
        if prediction[0] == 1:
            st.write('passenger survived:)')
        else:
            st.write('Unfortunately, passenger did not survive:(')

    except Exception as e:
        st.write(f'Error: {e}')
