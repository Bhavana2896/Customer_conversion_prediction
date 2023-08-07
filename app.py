import streamlit as st
import pickle
import pandas as pd

# Load the pickled model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapping Dictionaries
job_mapping = {'management': 8, 'technician': 4, 'entrepreneur': 1, 'blue-collar': 0, 'unknown': 5, 'retired': 10,
               'admin.': 7, 'services': 3, 'self-employed': 6, 'unemployed': 9, 'housemaid': 2, 'student': 11}
marital_mapping = {'married': 0, 'single': 2, 'divorced': 1}
education_qual_mapping = {'tertiary': 3, 'secondary': 1, 'unknown': 2, 'primary': 0}
call_type_mapping = {'unknown': 0, 'cellular': 2, 'telephone': 1}
mon_mapping = {'may': 0, 'jun': 4, 'jul': 1, 'aug': 5, 'oct': 8, 'nov': 3, 'dec': 10, 'jan': 2, 'feb': 6, 'mar': 11,
               'apr': 7, 'sep': 9}
prev_outcome_mapping = {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3}

# Streamlit App
st.title('Customer Conversion Prediction App')

# Input Features
prev_outcome = st.selectbox('Previous Outcome', list(prev_outcome_mapping.keys()))
dur = st.slider('Duration (in seconds)', min_value=0, max_value=643, value=200)
call_type = st.selectbox('Call Type', list(call_type_mapping.keys()))
mon = st.selectbox('Month', list(mon_mapping.keys()))
day = st.slider('Day', min_value=1, max_value=31, value=1)
job = st.selectbox('Job', list(job_mapping.keys()))
age = st.slider('Age', min_value=0, max_value=70, value=35)
marital = st.selectbox('Marital', list(marital_mapping.keys()))
education_qual = st.selectbox('Education Qualification', list(education_qual_mapping.keys()))

# Additional checks for valid day selections
valid_days = {
    'jan': 31, 'feb': 29, 'mar': 31, 'apr': 30, 'may': 31, 'jun': 30,
    'jul': 31, 'aug': 31, 'sep': 30, 'oct': 31, 'nov': 30, 'dec': 31
}

if day > valid_days[mon]:
    st.error(f"Invalid day ({day}) selected for {mon} month.")
elif mon == 'feb' and day > 28:
    st.error("Invalid day selected for February (should be 28 in non-leap years).")
else:
    # Create a DataFrame from user inputs
    data = {
        'prev_outcome': [prev_outcome],
        'dur': [dur],
        'call_type': [call_type],
        'mon': [mon],
        'day': [day],
        'job': [job],
        'age': [age],
        'marital': [marital],
        'education_qual': [education_qual]
    }
    input_df = pd.DataFrame(data)

    # Map categorical features to their numerical values based on the mapping dictionaries
    input_df['job'] = input_df['job'].map(job_mapping)
    input_df['marital'] = input_df['marital'].map(marital_mapping)
    input_df['education_qual'] = input_df['education_qual'].map(education_qual_mapping)
    input_df['call_type'] = input_df['call_type'].map(call_type_mapping)
    input_df['mon'] = input_df['mon'].map(mon_mapping)
    input_df['prev_outcome'] = input_df['prev_outcome'].map(prev_outcome_mapping)

    # Make Predictions
    prediction = model.predict(input_df)

    # Display the Prediction
    if prediction == 1:
        st.success('Yes, The customer is highly likely to subscribe')
    else:
        st.write(":red:'No, The customer is not likely to subscribe'")
