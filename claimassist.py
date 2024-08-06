import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import streamlit as st

# Load models and scalers
filename = "claimAssit_cts.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
jobtitle_le = pickle.load(open("le_job_title_Assist.pkl", 'rb'))
hereditary_disease_le = pickle.load(open("le_hereditary_diseases_Assist.pkl", 'rb'))
city_le = pickle.load(open("le_city_Assist.pkl", 'rb'))
scaler = pickle.load(open("scalerAssit.pkl", 'rb'))

def claim_prediction(input_data):
    # Convert input data into a DataFrame for processing
    input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents', 'smoker', 'city', 'bloodpressure', 'diabetes', 'regular_ex', 'job_title', 'claim'])

    # Replace categorical variables with numerical ones
    input_df['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    input_df['smoker'].replace(['no', 'yes'], [0, 1], inplace=True)
    input_df['diabetes'].replace(['no', 'yes'], [0, 1], inplace=True)
    input_df['regular_ex'].replace(['no', 'yes'], [0, 1], inplace=True)

    def safe_label_transform(le, column):
        known_labels = le.classes_
        input_df[column] = input_df[column].apply(lambda x: x if x in known_labels else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        input_df[column] = le.transform(input_df[column])

    safe_label_transform(hereditary_disease_le, 'hereditary_diseases')
    safe_label_transform(city_le, 'city')
    safe_label_transform(jobtitle_le, 'job_title')

    # Ensure the input data is scaled correctly
    input_data_scaled = scaler.transform(input_df)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_scaled)

    return 'The claim is likely to be Approved.' if prediction[0] > 0.5 else 'The claim is likely to be Rejected.'

def main():
    st.title('ClaimAssist Web Page')

    # Input fields
    name = st.text_input('Enter Name')
    age = st.text_input('Enter Age')
    sex = st.selectbox('Select Sex', ['female', 'male'])
    weight = st.text_input('Enter Weight (in kg)')
    bmi = st.text_input('Enter Body Mass Index (BMI)')
    hereditary_diseases = st.text_input('List any hereditary diseases')
    no_of_dependents = st.text_input('No of Dependents')
    smoker = st.selectbox('Do you smoke?', ['no', 'yes'])
    city = st.text_input('Enter City')
    bloodpressure = st.text_input('Blood Pressure')
    diabetes = st.selectbox('Do you have diabetes?', ['no', 'yes'])
    regular_ex = st.selectbox('Do you engage in regular exercise?', ['no', 'yes'])
    job_title = st.text_input('Enter Job Title')
    claim = st.text_input('Enter Claim Amount')

    input_data = [int(age), sex, float(weight), float(bmi), hereditary_diseases, int(no_of_dependents), smoker, city, bloodpressure, diabetes, regular_ex, job_title, float(claim)]
    if st.button('Claim Prediction'):
        outcome = claim_prediction(input_data)
        st.success(outcome)


if __name__ == '__main__':
    main()
