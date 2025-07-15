import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib
base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, "models", "best_model.pkl")
bin_path = os.path.join(base_path, "models", "bin_edges.pkl")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    .stApp {
        background: linear-gradient(to bottom right, #0b181f, #00001f, #010a3c, #03154a, #08225d, #102f6a);
        background-attachment: fixed;
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1c1c1c !important;
        color: white !important;
    }

    .stSlider > div > div > div {
        color: white !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #145374;
        color: white;
        border-radius: 10px;
    }

    .stButton button:hover {
        background-color: #5588a3;
        color: black;
    }

    /* Blur Glow Glassmorphism for form */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Load model and bin edges
model = joblib.load(model_path)
bins = joblib.load(bin_path)


st.set_page_config(
    page_title="Titanic Survival Predictor 🚢",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>🛳️ Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Step Aboard — Predict Who Survives the Titanic’s Voyage!</p>", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 65px;'></div>", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.slider("Age", 0, 80, 25)
        sibsp = st.number_input("No. of Siblings/Spouses Aboard", 0, 10, 0)

    with col2:
        parch = st.number_input("No. of Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare Paid ($)", 0.0, 600.0, 32.20, step=10.0)
        embarked = st.selectbox("Port of Embarkation", ["Southampton - England", "Cherbourg - France", "Queenstown - Ireland"])
        title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

    submitted = st.form_submit_button("Predict 🚀")

if submitted:
    sex = 0 if sex == "Male" else 1
    embarked = {"Southampton - England": 0, "Cherbourg - France": 1, "Queenstown - Ireland": 2}[embarked]
    title = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}[title]
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    age_bin = pd.cut([age], bins=bins["age_bins"], labels=False)[0]
    fare_bin = pd.cut([fare], bins=bins["fare_bins"], labels=False, include_lowest=True)[0]

    # Final input
    final_input = pd.DataFrame([[
        pclass, sex, age, sibsp, parch, fare, embarked, title,
        family_size, is_alone, age_bin, fare_bin
    ]], columns=[
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
        "FamilySize", "IsAlone", "AgeBin", "FareBin"
    ])
    prediction = model.predict(final_input)[0]
    result = "⚰️ Lost to history — couldn't survive the voyage...." if prediction == 0 else "🛟 Defied the odds — survived the icy fate!"

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(to bottom right, #fefefe, #e0e0e0);
            border: 2px solid #8a8a8a;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            padding: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: {"green" if prediction == 1 else "red"};
        '>
            Prediction: {result}
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.markdown("<br><hr><center>Built by Akshat </center>", unsafe_allow_html=True)

st.sidebar.title("🧭 About this App")
st.sidebar.markdown("""
Step aboard a digital reconstruction of the Titanic’s fateful journey. This app uses historical data and machine learning to predict whether a passenger might have survived the 1912 disaster.

- 🔧 Crafted: Using Streamlit
- 🧠 Model: Random Forest Classifier
- 📊 Dataset: Kaggle Titanic
""")
