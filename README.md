# Self Analysis Mental Health
This project is a mental health assessment tool built with Streamlit, using Random Forest and Logistic Regression models to predict symptoms of depression, anxiety, and related conditions. Additionally, Gemini 1.5 Flash provides AI-driven insights and coping mechanisms based on the predictions.

## 🚀 Features
* ✅ Predicts multiple mental health conditions based on user inputs
* ✅ Uses Random Forest (RF) and Logistic Regression (LR) for classification
* ✅ Provides AI-generated mental health tips via Gemini 1.5 Flash
* ✅ Simple, interactive Streamlit UI for easy access

## 📌 Datasets Used
This model was trained using two datasets:
* Mental Health in Tech Survey
* Depression and Anxiety Symptoms Dataset
🔹 Preprocessing Steps
* ✔ Selected relevant features and target labels
* ✔ Filled missing values with "Unknown"
* ✔ Applied Ordinal Encoding and Label Encoding based on feature relationships

## Access the UI
After running the command, open the localhost URL or network URL provided in the terminal.

## Get Predictions
* Fill in the required fields.
* Click Get Prediction to receive mental health insights.
* Get AI-driven suggestions from Gemini 1.5 Flash.
📊 Model Performance
Both Random Forest (RF) and Logistic Regression (LR) were tested for multiclass classification. The models performed well, offering a balance of accuracy and interpretability.

## 🛠️ Tech Stack
* Python
* Scikit-learn (Machine Learning)
* Streamlit (UI)
* Google Gemini 1.5 Flash (AI insights)
