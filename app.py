import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("Customer Churn Prediction & Retention Analytics")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio(
    "Select Section",
    ["Dataset Overview", "EDA", "Model Training", "Feature Importance", "Retention Analytics"]
)

# -----------------------------
# Dataset Overview
# -----------------------------
if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

# -----------------------------
# EDA
# -----------------------------
elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
    st.pyplot(fig)

# -----------------------------
# Model Training
# -----------------------------
elif menu == "Model Training":
    st.subheader("Churn Prediction Model")

    df_model = df.copy()
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
    df_model.drop('customerID', axis=1, inplace=True)

    cat_cols = df_model.select_dtypes(include='object').columns
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {acc*100:.2f}%")

# -----------------------------
# Feature Importance
# -----------------------------
elif menu == "Feature Importance":
    st.subheader("Feature Importance")

    df_model = df.copy()
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
    df_model.drop('customerID', axis=1, inplace=True)

    cat_cols = df_model.select_dtypes(include='object').columns
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    st.dataframe(importance)

# -----------------------------
# Retention Analytics
# -----------------------------
elif menu == "Retention Analytics":
    st.subheader("Retention Analytics")

    df_model = df.copy()
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
    df_model.drop('customerID', axis=1, inplace=True)

    cat_cols = df_model.select_dtypes(include='object').columns
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    churn_prob = rf.predict_proba(X)[:, 1]

    df_results = df.copy()
    df_results['Churn_Probability'] = churn_prob

    high_risk = df_results[df_results['Churn_Probability'] > 0.6]

    st.write("High Risk Customers Count:", high_risk.shape[0])
    st.dataframe(high_risk[['tenure', 'MonthlyCharges', 'Churn_Probability']].head())
