import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC

warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# ========= Function Definitions ===========

@st.cache_data
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)

    # Step 1: Normalize column names (strip spaces and lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace('_', ' ')

    # Step 2: Create a mapping from expected columns to actual ones
    expected_cols = {
        'pH Min': ['ph min', 'phmin'],
        'pH Max': ['ph max', 'phmax'],
        'T Min': ['t min', 'temperature min'],
        'T Max': ['t max', 'temperature max'],
        'DO Min': ['do min'],
        'DO Max': ['do max'],
        'BOD Min': ['bod min'],
        'BOD Max': ['bod max'],
        'NO3 Min': ['no3 min', 'nitrate min'],
        'NO3 Max': ['no3 max', 'nitrate max'],
        'Cond Min': ['cond min', 'conductivity min'],
        'Cond Max': ['cond max', 'conductivity max'],
        'Tc Min': ['tc min', 'total coliform min'],
        'Tc Max': ['tc max', 'total coliform max'],
        'Fc Min': ['fc min', 'fecal coliform min'],
        'Fc Max': ['fc max', 'fecal coliform max'],
    }

    col_mapping = {}

    for key, aliases in expected_cols.items():
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break
        if found:
            col_mapping[key] = found
        else:
            df[key.lower()] = np.nan
            col_mapping[key] = key.lower()

    # Step 3: Compute mean columns (cleaned column names)
    df['pH'] = df[[col_mapping['pH Min'], col_mapping['pH Max']]].mean(axis=1)
    df['Temp'] = df[[col_mapping['T Min'], col_mapping['T Max']]].mean(axis=1)
    df['DO'] = df[[col_mapping['DO Min'], col_mapping['DO Max']]].mean(axis=1)
    df['BOD'] = df[[col_mapping['BOD Min'], col_mapping['BOD Max']]].mean(axis=1)
    df['NO3'] = df[[col_mapping['NO3 Min'], col_mapping['NO3 Max']]].mean(axis=1)
    df['Cond'] = df[[col_mapping['Cond Min'], col_mapping['Cond Max']]].mean(axis=1)
    df['Tc'] = df[[col_mapping['Tc Min'], col_mapping['Tc Max']]].mean(axis=1)
    df['Fc'] = df[[col_mapping['Fc Min'], col_mapping['Fc Max']]].mean(axis=1)

    df.fillna({
        'pH': 7.0, 'Temp': 20.0, 'DO': 7.5, 'BOD': 2.0,
        'NO3': 0.5, 'Cond': 500, 'Tc': 100, 'Fc': 50
    }, inplace=True)

    df['Tc'] = np.log1p(df['Tc'])
    df['Fc'] = np.log1p(df['Fc'])

    return df

def classify_water_quality(row):
    if (6.5 <= row['pH'] <= 8.5 and row['DO'] >= 6.0 and row['BOD'] <= 1.0 and
        row['NO3'] <= 10 and row['Cond'] <= 1000 and
        np.expm1(row['Tc']) <= 10 and np.expm1(row['Fc']) == 0):
        return 'Safe for Immediate Consumption'
    elif (6.0 <= row['pH'] <= 9.0 and row['DO'] >= 4.0 and row['BOD'] <= 3.0 and
          row['NO3'] <= 50 and row['Cond'] <= 2500 and
          np.expm1(row['Tc']) <= 1000 and np.expm1(row['Fc']) <= 100):
        return 'Good'
    else:
        return 'Bad'

def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    return model, scaler, le

def predict_quality(pH, Temp, DO, BOD, NO3, Cond, Tc, Fc, model, scaler, le):
    input_data = pd.DataFrame([[pH, Temp, DO, BOD, NO3, Cond, Tc, Fc]],
                              columns=['pH', 'Temp', 'DO', 'BOD', 'NO3', 'Cond', 'Tc', 'Fc'])
    input_data['Tc'] = np.log1p(input_data['Tc'])
    input_data['Fc'] = np.log1p(input_data['Fc'])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    return le.inverse_transform(pred)[0]

# ========= Streamlit UI Starts Here ===========

st.set_page_config(page_title="💧 Water Quality Classifier", layout="wide", page_icon="💧")
st.markdown("<h1 style='text-align: center; color: #0077b6;'>💧 Water Quality Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze and predict water safety from uploaded test results.</p>", unsafe_allow_html=True)

st.sidebar.header("🔎 Navigation")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

# Load data
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    df['Water_Quality'] = df.apply(classify_water_quality, axis=1)

    st.success("✅ Data Loaded & Processed Successfully!")

    with st.expander("📋 Preview & Summary"):
        st.write(df.head())
        st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("📊 Water Quality Distribution")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Water_Quality', palette='Set2', ax=ax)
        ax.set_title("Distribution of Water Quality", fontsize=14)
        ax.set_xlabel("")
        st.pyplot(fig)

    with col2:
        st.bar_chart(df['Water_Quality'].value_counts())

    st.markdown("---")
    features = ['pH', 'Temp', 'DO', 'BOD', 'NO3', 'Cond', 'Tc', 'Fc']
    X = df[features]
    y = df['Water_Quality']
    model, scaler, le = train_model(X, y)

    st.subheader("🔮 Manual Prediction")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            input_pH = st.number_input("pH", value=7.0, step=0.1)
            input_Temp = st.number_input("Temperature (°C)", value=20.0, step=0.1)
            input_DO = st.number_input("Dissolved Oxygen (mg/L)", value=6.5)
            input_BOD = st.number_input("Biological Oxygen Demand", value=1.5)

        with col2:
            input_NO3 = st.number_input("Nitrate (mg/L)", value=10.0)
            input_Cond = st.number_input("Conductivity (μmho/cm)", value=500.0)
            input_Tc = st.number_input("Total Coliform (MPN/100ml)", value=50.0)
            input_Fc = st.number_input("Fecal Coliform (MPN/100ml)", value=5.0)

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        result = predict_quality(input_pH, input_Temp, input_DO, input_BOD,
                                 input_NO3, input_Cond, input_Tc, input_Fc,
                                 model, scaler, le)
        st.success(f"💧 Predicted Water Quality: **{result}**")

        st.balloons()
        st.markdown("---")

        with st.expander("📁 Download Prediction Report"):
            pred_df = pd.DataFrame({
                "pH": [input_pH],
                "Temp": [input_Temp],
                "DO": [input_DO],
                "BOD": [input_BOD],
                "NO3": [input_NO3],
                "Conductivity": [input_Cond],
                "Total Coliform": [input_Tc],
                "Fecal Coliform": [input_Fc],
                "Predicted Quality": [result]
            })
            st.dataframe(pred_df)
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "water_quality_prediction.csv", "text/csv")
