import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import streamlit as st

# Custom CSS for background and text styling
st.markdown(
"""
    <style>
    .main {
        background-color: #f5f5f5; /* Light grey background */
    }
    .approved {
        color: green;
        font-size: 24px;
        font-weight: bold;
    }
    .rejected {
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

filename = "claimAssit_cts.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
jobtitle_le = pickle.load(open("le_job_title_Assist.pkl", 'rb'))
hereditary_disease_le = pickle.load(open("le_hereditary_diseases_Assist.pkl", 'rb'))
city_le = pickle.load(open("le_city_Assist.pkl", 'rb'))
scaler = pickle.load(open("scalerAssit.pkl", 'rb'))

def claim_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents', 'smoker', 'city', 'bloodpressure', 'diabetes', 'regular_ex', 'job_title', 'claim'])

    input_df['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    input_df['smoker'].replace(['no', 'yes'], [0, 1], inplace=True)

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
    st.text_input('Enter Claim ID')
    st.text_input('Enter Name')
    age = st.text_input('Enter Age')
    sex = st.text_input('Enter Sex')
    weight = st.text_input('Enter Weight (in kg)')
    bmi = st.text_input('Enter Body Mass Index (BMI)')
    hereditary_diseases = st.text_input('List any hereditary diseases')
    no_of_dependents = st.text_input('No of Dependents')
    smoker = st.text_input('Do you smoke? (yes/no)')
    city = st.text_input('Enter City')
    bloodpressure = st.text_input('Blood Pressure')
    diabetes = st.text_input('Do you have diabetes? (yes/no)')
    regular_ex = st.text_input('Do you engage in regular exercise? (yes/no)')
    job_title = st.text_input('Enter Job Title')
    claim = st.text_input('Enter Claim Amount')

    outcome = ''
    if st.button('Claim Prediction'):
        prediction = claim_prediction([age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, city, bloodpressure, diabetes, regular_ex, job_title, claim])
        if prediction > 0.5:
            outcome = '<div class="approved">The claim is approved. You will receive your payment shortly!</div>'
        else:
            outcome = '<div class="rejected">The claim is rejected. Please contact support for further details.</div>'
        st.markdown(outcome, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
