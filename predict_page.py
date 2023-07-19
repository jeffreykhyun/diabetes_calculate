import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

def load_model(): 
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
gender = data['gender']
age = data['age']
hypertension = data['hypertension']
heart_disease = data['heart_disease']
smoking_history = data['smoking_history']
bmi = data['bmi']
HbA1c_level = data['HbA1c_level']
blood_glucose_level = data['blood_glucose_level']

def show_predict_page():
    st.title("Diabetes Risk Predictor")

    st.write("""### We need some information to predict diabetes """)

    gender = (
        "Male",
        "Female"
    )

    smoking_history = (
        "non-smoker",
        "current",
        "past_smoker"
    )

    gender = st.selectbox("gender", gender)
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
    hypertension = st.checkbox("Do you have hypertension? Check for Yes.")
    hypertension_value = 1 if hypertension else 0
    heart_disease = st.checkbox("Do you have heart disease? Check for Yes.")
    heart_disease_value = 1 if hypertension else 0
    smoking_history = st.selectbox("smoking_history", smoking_history)
    bmi = st.number_input("Enter your bmi", min_value=0.0)
    HbA1c_level = st.number_input("Enter your HbA1c level", min_value=0.0)
    blood_glucose_level = st.number_input("Enter your blood glucose level", min_value=0.0)

    ok = st.button("Calculate Diabetes Risk")
    if ok:
        sample = {'gender': gender, 'age': age, 'hypertension': hypertension_value, 'heart_disease': heart_disease_value, 'smoking_history': smoking_history, 'bmi': bmi, 'HbA1c_level': HbA1c_level, 'blood_glucose_level': blood_glucose_level}
        sample_df = pd.DataFrame(sample, index=[0])
        diabetes_risk = model.predict(sample_df)
        diagnosis = "Positive" if diabetes_risk[0] == 1 else "Negative"
        st.subheader(f"The diagnosis for diabetes is {diagnosis}")
