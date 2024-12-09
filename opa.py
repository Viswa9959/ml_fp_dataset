import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing objects
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Title and description of the web app.
st.title("OBESITY PREDICTION APP")
st.write("This app will predicts the obesity level's based on the user inputs.")

# Input form for the new user data.
def user_input_form():
    Gender = st.selectbox("Gender", ["Male", "Female"], key="Gender")
    Age = st.number_input("Age", min_value=1, max_value=120, step=1, key="Age")
    Height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, step=0.01, key="Height")
    Weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1, key="Weight")
    family_history_with_overweight = st.selectbox("Family History of Overweight", ["yes", "no"], key="family_history")
    FAVC = st.selectbox("Frequent High Caloric Food Consumption", ["yes", "no"], key="FAVC")
    FCVC = st.number_input("Frequency of Vegetable Consumption (1-3)", min_value=1, max_value=3, step=1, key="FCVC")
    NCP = st.number_input("Number of Main Meals", min_value=1, max_value=7, step=1, key="NCP")
    CAEC = st.selectbox("Eating Between Meals", ["No", "Sometimes", "Frequently", "Always"], key="CAEC")
    SMOKE = st.selectbox("Smoker", ["yes", "no"], key="SMOKE")
    CH2O = st.number_input("Water Intake (in liters)", min_value=1.0, max_value=5.0, step=0.1, key="CH2O")
    SCC = st.selectbox("Calorie Monitoring", ["yes", "no"], key="SCC")
    FAF = st.number_input("Physical Activity Frequency (hours per week)", min_value=0.0, max_value=7.0, step=0.25, key="FAF")
    TUE = st.number_input("Time Using Technology (hours per day)", min_value=0.0, max_value=24.0, step=0.25, key="TUE")
    CALC = st.selectbox("Alcohol Consumption", ["No", "Sometimes", "Frequently", "Always"], key="CALC")
    MTRANS = st.selectbox("Transportation Mode", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"], key="MTRANS")

    # Return user inputs as a dictionary.
    return {
        "Gender": [Gender],
        "Age": [Age],
        "Height": [Height],
        "Weight": [Weight],
        "family_history_with_overweight": [family_history_with_overweight],
        "FAVC": [FAVC],
        "FCVC": [FCVC],
        "NCP": [NCP],
        "CAEC": [CAEC],
        "SMOKE": [SMOKE],
        "CH2O": [CH2O],
        "SCC": [SCC],
        "FAF": [FAF],
        "TUE": [TUE],
        "CALC": [CALC],
        "MTRANS": [MTRANS]
    }


# Processing the input data.
user_data = user_input_form()
if st.button("Predict"):
    # Converting user input to a DataFrame.
    input_df = pd.DataFrame(user_data)
    
    # Preprocess categorical features.
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                        'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_cols:
        encoder = label_encoders[col]
        input_df[col] = encoder.transform(input_df[col])
    
    # Scaling the numerical features.
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Making the predictions.
    prediction = model.predict(input_df)
    decoded_prediction = label_encoders['NObeyesdad'].inverse_transform(prediction)
    
    # Displaying the result.
    st.success(f"The predicted obesity level is: {decoded_prediction[0]}")
