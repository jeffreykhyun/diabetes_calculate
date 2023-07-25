import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    df = df.drop_duplicates()
    df = df[df['gender'] != 'Other']
    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Variables of Diabetes Risk")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='diabetes', y='age', data=df, ax=ax1, palette=['green', 'red'])
    ax1.set_title('Age and Diabetes')
    st.write("""#### Age and Diabetes """)
    st.pyplot(fig1)


    # Plot 2: Gender and Diabetes
    fig2, ax2 = plt.subplots()
    sns.countplot(x='gender', hue='diabetes', data=df, ax=ax2, palette=['green', 'red'])
    ax2.set_title('Gender and Diabetes')
    st.write("""#### Gender and Diabetes """)
    st.pyplot(fig2)

    # Plot 3: HbA1c levels and Diabetes
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='diabetes', y='HbA1c_level', data=df, ax=ax3, palette=['green', 'red'])
    ax3.set_title('HbA1c levels and Diabetes')
    st.write("""#### HbA1c and Diabetes """)
    st.pyplot(fig3)

    # Plot 4: Blood Glucose Levels and Diabetes
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='diabetes', y='blood_glucose_level', data=df, ax=ax4, palette=['green', 'red'])
    ax4.set_title('Blood Glucose Levels and Diabetes')
    st.write("""#### Blood Glucose Levels and Diabetes """)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.boxplot(x='diabetes', y='bmi', hue='gender', data=df, ax=ax5, palette={'Female': 'pink', 'Male': 'aqua'})
    ax5.set_title('BMI Distribution by Diabetes Diagnosis and Gender')
    st.write("""#### BMI Distribution by Diabetes Diagnosis and Gender """)
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots()
    sns.boxplot(x='diabetes', y='age', hue='gender', data=df, ax=ax6, palette={'Female': 'pink', 'Male': 'aqua'})
    ax6.set_title('Age Distribution by Diabetes Diagnosis and Gender')
    st.write("""#### Age Distribution by Diabetes Diagnosis and Gender """)
    st.pyplot(fig6)

   
