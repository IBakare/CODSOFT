import streamlit as st
import joblib

# Load the saved model from the file
loaded_model = joblib.load('knn_model.pkl')

# Define a function to make predictions
def predict_survival(features):
    prediction = loaded_model.predict([features])
    return prediction[0]

# Streamlit interface code
st.title('Titanic Survival Predictor')

# Input fields for user to enter data
pclass = st.number_input('Enter Pclass:', min_value=1, max_value=3, value=1)
sex = st.selectbox('Select Sex:', ['male', 'female'])
age = st.number_input('Enter Age:', min_value=0, max_value=100, value=30)
fare = st.number_input('Enter Fare:', min_value=0.0, value=50.0)

# Make prediction on user input
if st.button('Predict'):
    gender = 1 if sex == 'female' else 0
    prediction_features = [pclass, gender, age, fare]
    result = predict_survival(prediction_features)
    if result == 0:
        st.error('Prediction: Did not survive')
    else:
        st.success('Prediction: Survived')
