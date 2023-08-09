import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image

scaler = StandardScaler()


# Load the pickled model
with open('xgboost_model.pkl', 'rb') as f:
    loaded = pickle.load(f)
model = loaded['model']
scaler = loaded['scaler']

# Mapping Dictionaries
job_mapping = {'management': 8, 'technician': 4, 'entrepreneur': 1, 'blue-collar': 0, 'unknown': 5, 'retired': 10,
               'admin.': 7, 'services': 3, 'self-employed': 6, 'unemployed': 9, 'housemaid': 2, 'student': 11}
marital_mapping = {'married': 0, 'single': 2, 'divorced': 1}
education_qual_mapping = {'tertiary': 3, 'secondary': 1, 'unknown': 2, 'primary': 0}
call_type_mapping = {'unknown': 0, 'cellular': 2, 'telephone': 1}
mon_mapping = {'may': 0, 'jun': 4, 'jul': 1, 'aug': 5, 'oct': 8, 'nov': 3, 'dec': 10, 'jan': 2, 'feb': 6, 'mar': 11,
               'apr': 7, 'sep': 9}
prev_outcome_mapping = {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3}

# ------------------------------ Streamlit App -------------------------------#

# Configure page layout
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("About the Developer")
st.sidebar.write("**Bhavana B**")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/bbhavana/)")
st.sidebar.markdown("[GitHub](https://github.com/Bhavana2896)")

# image = "insurance.jpeg"  # Replace with the correct filename

style = f"""
    <style>
    body {{
        background-color: #f0f0f0; /* Set background color */
    }}
    .overlay-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: red;
        font-size: 30px;  /* Increase font size */
        font-weight: bold; /* Make text bold */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);  /* Add shadow to text */
        z-index: 1; /* Ensure text is above the image */
    }}
    </style>
"""

# # Display the text overlay using HTML
st.markdown(style, unsafe_allow_html=True)
# Display the overlay text
st.markdown('<div class="overlay-text">Customer Conversion Prediction App</div>', unsafe_allow_html=True)
# # Display the image with text overlay
# st.markdown(f"""
#     <div class="image-container">
#         <img src="{image}" alt="Image" width="100%">
#         <div class="overlay-text">Customer Conversion Prediction</div>
#     </div>
# """, unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")
# Use columns for layout
col1, col2, col3 = st.columns(3)  # Divide into 3 columns

# First row of select boxes for Input Features
with col1:
    prev_outcome = st.selectbox('Previous Outcome', list(prev_outcome_mapping.keys()))
with col2:
    call_type = st.selectbox('Call Type', list(call_type_mapping.keys()))
with col3:
    mon = st.selectbox('Month', list(mon_mapping.keys()))

# Second row of select boxes for Input Features
with col1:
    job = st.selectbox('Job', list(job_mapping.keys()))
with col2:
    marital = st.selectbox('Marital', list(marital_mapping.keys()))
with col3:
    education_qual = st.selectbox('Education Qualification', list(education_qual_mapping.keys()))

# Sliders for Input Features
dur = st.slider('Duration (in seconds)', min_value=0, max_value=643, value=200)
day = st.slider('Day', min_value=1, max_value=31, value=10)
age = st.slider('Age', min_value=0, max_value=70, value=35)

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
    if st.button("Predict"):
        data = {
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education_qual': [education_qual],
            'call_type': [call_type],
            'day': [day],
            'mon': [mon],
            'dur': [dur],
            'prev_outcome': [prev_outcome]
        }
        input_df = pd.DataFrame(data)

        # Map categorical features to their numerical values based on the mapping dictionaries
        input_df['job'] = input_df['job'].map(job_mapping)
        input_df['marital'] = input_df['marital'].map(marital_mapping)
        input_df['education_qual'] = input_df['education_qual'].map(education_qual_mapping)
        input_df['call_type'] = input_df['call_type'].map(call_type_mapping)
        input_df['mon'] = input_df['mon'].map(mon_mapping)
        input_df['prev_outcome'] = input_df['prev_outcome'].map(prev_outcome_mapping)
        print(input_df)

        # Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)
        # Make Predictions
        prediction = model.predict(input_scaled)

        # Display the Prediction
        if prediction[0] == 1:
            st.success('Yes, The customer is highly likely to subscribe')
        else:
            st.error('No, The customer is not likely to subscribe')
