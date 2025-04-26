import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import base64

# Streamlit page config
st.set_page_config(page_title="Smart Health Guardian", page_icon="🩺", layout="centered")

# Display logo and app title side by side
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("Smart Health Guardian")
    st.caption("Powered by AI & Streamlit")

# Load your trained model
model = pickle.load(open('svm_model.pkl', 'rb'))

# Load the dataset for SHAP explanations
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User selects input mode
input_mode = st.selectbox("Select Input Mode", ("Basic", "Detailed"))

# Input Form Based on Mode Selection
if input_mode == "Basic":
    # Basic inputs for regular users (only glucose, age, and optional BP/BMI)
    st.markdown("### 📝 Enter Patient Health Data (Basic Mode)")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    bp = st.number_input("Blood Pressure (mmHg) (Optional)", min_value=0, max_value=150, value=70)
    bmi = st.number_input("Body Mass Index (BMI) (Optional)", min_value=0.0, max_value=70.0, value=25.0)

    # Default assumptions for other features
    preg = 0
    skin = 20  # Average skin thickness
    insulin = 80  # Average insulin level
    dpf = 0.5  # Average Diabetes Pedigree Function

else:
    # Detailed inputs for doctors
    st.markdown("### 📝 Enter Patient Health Data (Detailed Mode)")
    preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

# Risk Level Function
def get_risk_level(glucose, bmi, age):
    if glucose >= 150 or bmi >= 35 or age >= 60:
        return "High Risk"
    elif glucose >= 120 or bmi >= 30 or age >= 45:
        return "Medium Risk"
    else:
        return "Low Risk"

# Health Advice Function
def get_health_advice(glucose, bmi, bp, age):
    advice = []
    if glucose >= 140:
        advice.append("Your glucose level is high. Reduce sugar and carb intake.")
    elif glucose < 70:
        advice.append("Glucose is low. You may need to eat more frequently.")
    else:
        advice.append("Glucose level is normal.")
    if bmi >= 30:
        advice.append("Your BMI suggests obesity. Try regular exercise and balanced diet.")
    elif bmi >= 25:
        advice.append("Your BMI is slightly high. Watch your weight and stay active.")
    elif bmi < 18.5:
        advice.append("BMI is low. You may need to gain some healthy weight.")
    else:
        advice.append("BMI is in a healthy range.")
    if bp >= 130:
        advice.append("Blood pressure is high. Reduce salt and manage stress.")
    elif bp < 80:
        advice.append("Blood pressure is low. Stay hydrated and consult a doctor.")
    else:
        advice.append("Blood pressure is in the normal range.")
    if age >= 60:
        advice.append("Age is a factor. Consider regular health checkups.")
    elif age >= 45:
        advice.append("Mid-age. Stay consistent with healthy habits.")
    return advice

# PDF Generator Function
def create_pdf(prediction, risk_level, health_advice, input_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.image("logo.png", x=10, y=8, w=25)
    pdf.set_xy(40, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Smart Health Guardian Report", ln=True)

    pdf.ln(20)  # Line break
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_level}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Health Advice:", ln=True)
    for tip in health_advice:
        pdf.multi_cell(0, 10, f"- {tip}")

    pdf.ln(5)
    pdf.cell(200, 10, txt="Input Data:", ln=True)
    features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
    for i, val in enumerate(input_data[0]):
        pdf.cell(200, 10, txt=f"{features[i]}: {val}", ln=True)

    pdf.output("report.pdf")

    # Encode PDF for download
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="SmartHealthReport.pdf">📄 Download Report as PDF</a>'
    return href

# Predict Button
if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    risk_level = get_risk_level(glucose, bmi, age)
    health_advice = get_health_advice(glucose, bmi, bp, age)

    if prediction[0] == 1:
        st.error("❗ The patient is likely diabetic.")
        st.warning(f"Risk Level: {risk_level}")
    else:
        st.success("✅ The patient is not diabetic.")
        st.info(f"Risk Level: {risk_level}")

    st.markdown("### 💡 Health Advice")
    for tip in health_advice:
        st.write(f"- {tip}")

    # SHAP Explainable AI
    st.markdown("### 🔍 Feature Importance (Model Explanation)")
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
    shap_values = explainer(input_data)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    
    # PDF Export
    st.markdown("# Export Report")
    download_link = create_pdf(prediction, risk_level, health_advice, input_data)
    st.markdown(download_link, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <small>🧑‍💻 Built by Anuj • Powered by SVM, SHAP & Streamlit</small>
    </div>
""", unsafe_allow_html=True)
