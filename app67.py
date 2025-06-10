import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
try:
    model = joblib.load('heart_disease_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run the Colab notebook to save the model and scaler.")
    st.stop()

st.title("Heart Disease Prediction App")

st.sidebar.header("User Input Features")

def user_input_features():
    male = st.sidebar.radio('Sex (Male)', [0, 1])
    age = st.sidebar.slider('Age', 20, 80, 50)
    education = st.sidebar.selectbox('Education', [1, 2, 3, 4], format_func=lambda x: {1:'Some High School', 2:'High School/GED', 3:'Some College/Vocational', 4:'College'}[x])
    currentSmoker = st.sidebar.radio('Current Smoker', [0, 1])
    cigsPerDay = st.sidebar.slider('Cigarettes per Day', 0, 70, 10)
    BPMeds = st.sidebar.radio('on BP Meds', [0, 1])
    prevalentStroke = st.sidebar.radio('Prevalent Stroke', [0, 1])
    prevalentHyp = st.sidebar.radio('Prevalent Hypertensive', [0, 1])
    diabetes = st.sidebar.radio('Diabetes', [0, 1])
    totChol = st.sidebar.slider('Total Cholesterol', 100, 700, 200)
    sysBP = st.sidebar.slider('Systolic BP', 80, 300, 130)
    diaBP = st.sidebar.slider('Diastolic BP', 50, 200, 80)
    BMI = st.sidebar.slider('BMI', 15.0, 50.0, 25.0)
    heartRate = st.sidebar.slider('Heart Rate', 40, 150, 75)
    glucose = st.sidebar.slider('Glucose', 40, 400, 80)

    data = {'male': male,
            'age': age,
            'education': education,
            'currentSmoker': currentSmoker,
            'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds,
            'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose
            }
    features = pd.DataFrame(data, index=[0])

    # Add engineered features (must match the training data)
    features['meanBP'] = (2 * features['diaBP'] + features['sysBP']) / 3

    # BMI category encoding (must match the training data's label encoding)
    def bmi_category(bmi):
        if bmi < 18.5: return "Underweight"
        elif bmi < 25: return "Normal"
        elif bmi < 30: return "Overweight"
        else: return "Obese"

    features['bmi_cat'] = features['BMI'].apply(bmi_category)

    # This part is tricky - the original notebook uses LabelEncoder on the full dataset *before* the split.
    # For a robust app, you would need to save the LabelEncoder fitted on the training data
    # and load it here. Since we don't have the saved encoder, we'll hardcode the mapping
    # based on the common ordering of categories after LabelEncoding.
    # Ensure this mapping matches the order in which LabelEncoder assigned values during training.
    bmi_mapping = {"Underweight": 2, "Normal": 1, "Overweight": 3, "Obese": 0} # Verify this mapping
    features['bmi_cat_encoded'] = features['bmi_cat'].map(bmi_mapping)
    features.drop('bmi_cat', axis=1, inplace=True)


    # Reorder columns to match the training data columns *exactly*
    # Get the list of columns from the original training data
    # This requires knowing the exact column order used for training
    # Assuming the column order was based on the original DataFrame after feature engineering and encoding
    # You might need to inspect X_train's columns after scaling in your notebook
    original_columns_after_engineering = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                                          'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
                                          'diaBP', 'BMI', 'heartRate', 'glucose', 'meanBP', 'bmi_cat_encoded'] # Verify this order

    # Ensure the input DataFrame has these columns in this order
    # Add missing columns if any (though unlikely for a single input row)
    # Reindex the input features DataFrame
    features = features.reindex(columns=original_columns_after_engineering, fill_value=0)


    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Apply the same scaling as used during training
# Ensure the scaler was fitted on the *correct* features (all X columns after engineering)
try:
    input_df_scaled = scaler.transform(input_df)
except ValueError as e:
     st.error(f"Error during scaling. Ensure the input features match the features the scaler was trained on. Error: {e}")
     st.stop()


# Make prediction
if st.sidebar.button('Predict'):
    try:
        prediction = model.predict(input_df_scaled)
        prediction_proba = model.predict_proba(input_df_scaled)

        st.subheader('Prediction')
        chd_prediction = 'Yes (High Risk)' if prediction[0] == 1 else 'No (Low Risk)'
        st.write(f'Predicted Heart Disease Risk within 10 Years: **{chd_prediction}**')

        st.subheader('Prediction Probability')
        # Display probabilities for both classes
        prob_no_chd = prediction_proba[0][0]
        prob_yes_chd = prediction_proba[0][1]

        st.write(f'Probability of No CHD: **{prob_no_chd:.4f}**')
        st.write(f'Probability of Yes CHD: **{prob_yes_chd:.4f}**')

        # Interpretation based on probability
        if prob_yes_chd > 0.5: # Using 0.5 as threshold based on default classification
            st.warning("Based on the model, this individual is predicted to be at high risk of Coronary Heart Disease within 10 years.")
        else:
            st.success("Based on the model, this individual is predicted to be at low risk of Coronary Heart Disease within 10 years.")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


st.markdown("""
<br>
<div style="text-align: center">
    <p>This app uses a Random Forest model trained on the Framingham Heart Study dataset to predict the 10-year risk of Coronary Heart Disease.
    <br>
    <b>Disclaimer:</b> This is for informational purposes only and not a substitute for professional medical advice.
    </p>
</div>
""", unsafe_allow_html=True)