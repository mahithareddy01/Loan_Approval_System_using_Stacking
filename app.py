import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# -----------------------------------
# LOAD TRAINED MODEL (or train inside)
# -----------------------------------
# If you already saved model:
# stacking_model = joblib.load("stacking_model.pkl")

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict whether a loan will be approved by combining multiple ML models for better decision making."
)

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------
st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# -----------------------------------
# ENCODE INPUTS
# -----------------------------------
credit_history = 1 if credit_history == "Yes" else 0
self_employed = 1 if employment == "Self-Employed" else 0

urban = semiurban = rural = 0
if property_area == "Urban":
    urban = 1
elif property_area == "Semi-Urban":
    semiurban = 1
else:
    rural = 1

input_data = np.array([[
    app_income,
    co_income,
    loan_amt,
    loan_term,
    credit_history,
    self_employed,
    urban,
    semiurban,
    rural
]])

# -----------------------------------
# MODEL ARCHITECTURE DISPLAY
# -----------------------------------
st.markdown("## 🏗 Model Architecture (Stacking Ensemble)")

st.info("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression  

Predictions from base models are used as inputs to the meta-model.
""")

# -----------------------------------
# DEMO TRAINING (Sample Dummy Model)
# Replace this with your trained model
# -----------------------------------
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

stacking_model = StackingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('rf', rf)
    ],
    final_estimator=LogisticRegression()
)

# Dummy training (for demo only)
X_dummy = np.random.rand(100, 9)
y_dummy = np.random.randint(0, 2, 100)
stacking_model.fit(X_dummy, y_dummy)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    base_preds = stacking_model.named_estimators_

    lr_pred = base_preds['lr'].predict(input_data)[0]
    dt_pred = base_preds['dt'].predict(input_data)[0]
    rf_pred = base_preds['rf'].predict(input_data)[0]

    final_pred = stacking_model.predict(input_data)[0]
    confidence = max(stacking_model.predict_proba(input_data)[0]) * 100

    st.markdown("## 📊 Base Model Predictions")

    st.write(f"Logistic Regression → {'Approved' if lr_pred==1 else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred==1 else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred==1 else 'Rejected'}")

    st.markdown("## 🧠 Final Stacking Decision")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown(f"### 📈 Confidence Score: {confidence:.2f}%")

    # -----------------------------------
    # BUSINESS EXPLANATION
    # -----------------------------------
    st.markdown("## 💼 Business Explanation")

    if final_pred == 1:
        st.write("""
Based on the applicant’s income, credit history, and combined predictions from multiple models,  
the applicant is likely to repay the loan.

Therefore, the stacking model predicts **loan approval**.
""")
    else:
        st.write("""
Based on the applicant’s income, credit history, and combined predictions from multiple models,  
the applicant shows higher risk of default.

Therefore, the stacking model predicts **loan rejection**.
""")
