import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config and styling
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    h1 {color: #2c3e50;}
    .stButton>button {background-color: #2c3e50; color: white;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Diabetes Health Predictor")
st.subheader("Enter your health details below:")

# Load and clean data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)
df.fillna(df.mean(), inplace=True)

# Train model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Input form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
