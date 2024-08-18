import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import streamlit as st

# Custom CSS for professional look with a box style
st.markdown(
    """
    <style>
    .approved {
        color: green;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid green;
    }
    .rejected {
        color: red;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid red;
    }
    .error {
        color: orange;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid orange;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and encoders
filename = "claimAssit_cts.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
jobtitle_le = pickle.load(open("le_job_title_Assist.pkl", 'rb'))
hereditary_disease_le = pickle.load(open("le_hereditary_diseases_Assist.pkl", 'rb'))
city_le = pickle.load(open("le_city_Assist.pkl", 'rb'))
scaler = pickle.load(open("scalerAssit.pkl", 'rb'))

# Sample list of claim IDs (this can be replaced by actual database or stored data)
existing_claim_ids = []  # Replace with real data

def claim_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents', 'smoker', 'city', 'bloodpressure', 'diabetes', 'regular_ex', 'job_title', 'claim'])

    # Encode categorical data
    input_df['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    input_df['smoker'].replace([0, 1], inplace=True)
    input_df['diabetes'].replace([0, 1], inplace=True)
    input_df['regular_ex'].replace([0, 1], inplace=True)

    # Safe label transformation
    def safe_label_transform(le, column):
        known_labels = le.classes_
        input_df[column] = input_df[column].apply(lambda x: x if x in known_labels else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        input_df[column] = le.transform(input_df[column])

    safe_label_transform(hereditary_disease_le, 'hereditary_diseases')
    safe_label_transform(city_le, 'city')
    safe_label_transform(jobtitle_le, 'job_title')

    input_data_scaled = scaler.transform(input_df)
    prediction = loaded_model.predict(input_data_scaled)

    return prediction[0]

def main():
    st.title('ClaimAssist Web Page')

    # Input fields
    claim_id = st.text_input('Enter Claim ID')
    name = st.text_input('Enter Name')
    age = st.text_input('Enter Age')
    sex = st.selectbox('Select Sex', ('female', 'male'))
    weight = st.text_input('Enter Weight (in kg)')
    bmi = st.text_input('Enter Body Mass Index (BMI)')
    hereditary_diseases = st.text_input('List any hereditary diseases')
    no_of_dependents = st.text_input('No of Dependents')
    smoker = st.selectbox('Do you smoke?', ('no', 'yes'))  # Dropdown for smoker
    city = st.text_input('Enter City')
    bloodpressure = st.text_input('Blood Pressure')
    diabetes = st.selectbox('Do you have diabetes?', ('no', 'yes'))  # Dropdown for diabetes
    regular_ex = st.selectbox('Do you engage in regular exercise?', ('no', 'yes'))  # Dropdown for regular exercise
    job_title = st.text_input('Enter Job Title')
    claim = st.text_input('Enter Claim Amount')

    outcome = ''
    if st.button('Claim Prediction'):
        # Convert yes/no inputs to 1/0
        smoker = 1 if smoker == 'yes' else 0
        diabetes = 1 if diabetes == 'yes' else 0
        regular_ex = 1 if regular_ex == 'yes' else 0
        
        # Check if the claim ID already exists
        if claim_id in existing_claim_ids:
            outcome = f'<div class="error">Claim ID "{claim_id}" already exists. Please enter a unique Claim ID.</div>'
        else:
            existing_claim_ids.append(claim_id)
            prediction = claim_prediction([age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, city, bloodpressure, diabetes, regular_ex, job_title, claim])
            if prediction > 0.5:
                outcome = f'<div class="approved">Congratulations {name}, your claim is approved! You will receive your payment shortly!</div>'
            else:
                outcome = f'<div class="rejected">Sorry {name} your claim has been reviewed and unfortunately, it has been rejected.</div>'
        
        st.markdown(outcome, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
