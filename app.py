import streamlit as st
import joblib
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the trained ML model
model = joblib.load('model_rf.pkl')  # Make sure this model is trained for text classification

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Function to make predictions
def predict_mental_health(features):
    # Convert input into NumPy array for prediction
    input_data = np.array([features])
    prediction = model.predict(input_data)[0]  # Get predictions
    
    # Map predictions to corresponding labels
    labels = ["Depression Diagnosis", "Anxiety Diagnosis", "Depression Treatment", 
              "Anxiety Treatment", "Suicidal", "Work Interference"]
    
    results = {labels[i]: "Yes" if prediction[i] == 1 else "No" for i in range(len(labels))}
    return results


# Function to get feedback from Gemini
def get_gemini_feedback(prediction_results):
    results_text = "\n".join([f"{condition}: {status}" for condition, status in prediction_results.items()])
    prompt = f"""
    A user completed a mental health assessment and received the following predictions:
    
    {results_text}
    
    Based on this assessment, provide mental health advice. Include:
    - General well-being tips
    - How to cope with potential mental health concerns
    - When professional help is necessary
    - Encouraging words for the user
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text

# Streamlit UI
st.title("Mental Health Assessment Chatbot")
st.write("Fill in the details below to assess your mental health.")

# Collect user input
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
phq_score = st.slider("PHQ Score (Depression Scale)", min_value=0, max_value=27, value=5)
gad_score = st.slider("GAD Score (Anxiety Scale)", min_value=0, max_value=21, value=5)
depression_severity = st.selectbox("Depression Severity", ["None-minimal","Mild","Moderate","Moderately severe", "Severe"])
depressiveness = st.selectbox("Depressiveness", ["False", "True"])
treatment=st.selectbox("Treatment",["No", "Yes"])
anxiousness = st.selectbox("Anxiousness", ["False", "True"])
anxiety_treatment=st.selectbox("Anxiety treatment", ["None-minimal","Mild","Moderate","Moderately severe", "Severe"])
family_history = st.selectbox("Family History of Mental Health Issues", ["No", "Yes"])

# Convert categorical inputs into numerical values
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
family_history_mapping = {"No": 0, "Yes": 1}
depression_severity_mapping={"None-minimal":1,"Mild":2 ,"Moderate":3, "Moderately severe":4, "Severe":5}
depressiveness_map={"False":0, "True": 1}
anxiousness_map={"False":0, "True": 1}
treatment_map={"No":0, "Yes":1}
anxiety_treatment_map={"None-minimal":1,"Mild":2 ,"Moderate":3, "Moderately severe":4, "Severe":5}

# Prepare input for model
input_features = [
    age, gender_mapping[gender], bmi, phq_score, depression_severity_mapping[depression_severity],
    depressiveness_map[depressiveness], gad_score, anxiousness_map[anxiousness],anxiety_treatment_map[anxiety_treatment] , family_history_mapping[family_history], treatment_map[treatment]
]

# Predict when user clicks the button
if st.button("Get Prediction"):
    prediction_results = predict_mental_health(input_features)
    st.write("**Predicted Mental Health Conditions:**")
    
    for condition, result in prediction_results.items():
        status = "Yes" if result == 1 else "No"
        st.write(f"- {condition}: **{status}**")

    # Generate AI feedback
    gemini_feedback = get_gemini_feedback(prediction_results)
    st.write("**AI Advice:**")
    st.write(gemini_feedback)