import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="RedRooster99/churn-model", filename="best_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Purchase Prediction
st.title("Wellness Tourism Package Purchase Prediction App")
st.write("The Wellness Tourism Package Purchase Prediction App is an internal tool for the tourism company staff that predicts whether customer are likely to purchase based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number of People Visiting (total number of people accompanying the customer on the trip)", min_value=1, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star (preferred hotel rating by the customer)", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips (average number of trips the customer takes annually)", min_value=0, value=1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (number of children below age 5 accompanying the customer)", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income (gross monthly income of the customer)", min_value=0.0, value=50000.0)
DurationOfPitch = st.number_input("Duration of Pitch (duration of the sales pitch delivered to the customer in minutes)", min_value=0.0, value=10.0)
NumberOfFollowups = st.number_input("Number of Follow-ups (total number of follow-ups by the salesperson after the sales pitch)", min_value=0, value=1)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score (score indicating the customer's satisfaction with the sales pitch)", min_value=1, max_value=5, value=3)
Passport = st.selectbox("Passport (whether the customer holds a valid passport)", ["Yes", "No"])
OwnCar = st.selectbox("Own Car (whether the customer owns a car)", ["Yes", "No"])
TypeofContact = st.selectbox("Type of Contact (method by which the customer was contacted)", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Freelancer", "Other"])
Gender = st.selectbox("Gender (gender of the customer)", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched (type of product pitched to the customer)", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status (marital status of the customer)", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation (customer's designation in their current organization)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
CityTier = st.selectbox("City Tier (city category based on development, population, and living standards)", ["Tier 1", "Tier 2", "Tier 3"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'CityTier': CityTier
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
