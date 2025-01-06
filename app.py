import streamlit as st
import pickle
import numpy as np
import pandas as pd

# page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="centered"
)

# loading the model
@st.cache_resource
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
    """
    Preprocess the input data (matching training data)
    """
    
    data = {
        'gender_Female': [0],
        'gender_Male': [0],
        'gender_Other': [0],
        'smoking_history_No Info': [0],
        'smoking_history_current': [0],
        'smoking_history_ever': [0],
        'smoking_history_former': [0],
        'smoking_history_never': [0],
        'smoking_history_not current': [0],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    }
    
    # gender
    gender_map = {0:'Female', 1:'Male'}
    data[f'gender_{gender_map[gender]}'] = [1]
    
    # smoking history
    smoking_map = {
        0: 'never',
        1: 'former',
        2: 'current',
        3: 'not current',
        4: 'ever',
        5: 'No Info'
    }
    data[f'smoking_history_{smoking_map[smoking_history]}'] = [1]
    
    # dataFrame 
    df = pd.DataFrame(data)
    
    # Ensure exact column order as seen in the training data
    expected_columns = [
        'gender_Female', 'gender_Male', 'gender_Other',
        'smoking_history_No Info', 'smoking_history_current',
        'smoking_history_ever', 'smoking_history_former',
        'smoking_history_never', 'smoking_history_not current', 'age',
        'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
        'blood_glucose_level'
    ]
    
    df = df.reindex(columns=expected_columns,fill_value=0)
    return df

def main():
    st.title("Diabetes Prediction System 🏥")
    st.markdown("""
    This app predicts the likelihood of diabetes based on various health parameters.
    Please fill in the information below to get a prediction.
    """)
    
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male"
            )
            
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=40
            )
            
            hypertension = st.selectbox(
                "Hypertension",
                options=[0, 1],
                format_func= lambda x:  "No" if x == 0 else "Yes"
            )
            
            heart_disease = st.selectbox(
                "Heart Disease",
                options=[0, 1],
                format_func= lambda x: "No" if x == 0 else  "Yes"
            )
        
        with col2:
            smoking_history = st.selectbox(
                "Smoking History",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: {
                    0: "Never",
                    1: "Former",
                    2: "Current",
                    3: "Not Current",
                    4: "Ever",
                    5: "No Info"
                }[x]
            )
            
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=0.1
            )
            
            hba1c_level = st.number_input(
                "HbA1c Level",
                min_value=3.0,
                max_value=15.0,
                value=5.5,
                step=0.1
            )
            
            blood_glucose_level = st.number_input(
                "Blood Glucose Level",
                min_value=50,
                max_value=500,
                value=120
            )

        submit_button = st.form_submit_button("Predict")

    model = load_model()

    if submit_button and model is not None:
        try:
            # Preprocess the input data
            input_df = preprocess_input(
                gender, age, hypertension, heart_disease, 
                smoking_history, bmi, hba1c_level, blood_glucose_level
            )
            
            # Debug information
            with st.expander("Show preprocessed features"):
                st.write(input_df)

            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]

            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.error("High Risk of Diabetes")
                else:
                    st.success("Low Risk of Diabetes")
            
            with col2:
                st.metric(
                    label="Risk Probability",
                    value=f"{probability:.1%}"
                )

            # Display input summary
            st.subheader("Input Summary")
            summary_df = pd.DataFrame({
                'Feature': [
                    'Gender', 'Age', 'Hypertension', 'Heart Disease',
                    'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose Level'
                ],
                'Value': [
                    'Male' if gender == 1 else 'Female',
                    f"{age} years",
                    'Yes' if hypertension == 1 else 'No',
                    'Yes' if heart_disease == 1 else 'No',
                    {0: "Never", 1: "Former", 2: "Current", 3: "Not Current", 4: "Ever", 5: "No Info"}[smoking_history],
                    f"{bmi:.1f}",
                    f"{hba1c_level:.1f}",
                    f"{blood_glucose_level:.0f}"
                ]
            })
            st.dataframe(summary_df, hide_index=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check the model compatibility with the input features.")

if __name__ == "__main__":
    main()