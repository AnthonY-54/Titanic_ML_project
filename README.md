# 🛳️ Titanic Survival Predictor

Step aboard and predict who would survive the Titanic's voyage. 
This is a prediction model made using machine learning and streamlit.

---

## 🚀 Live Demo

*Coming soon on Streamlit Cloud...*

---

## 💡 Features

- Streamlit-based web interface
- Clean, modern UI with gradient background and blur effect
- Handles inputs - age, fare, title, family size, port of embarkation, etc.
- Outputs prediction with custom emojis 🚀🛟⚰️

---

## 🧠 ML Model

- Model: Logistic Regression 
- Preprocessing:
  - Binning: `AgeBin`, `FareBin`
  - Feature Engineering: `FamilySize`, `IsAlone`, `Title`
- Encodings for categorical features

---

## ▶️ Run Locally

```bash
git clone https://github.com/AnthonyY-54/Titanic_ML_project.git
cd Titanic_ML_project
pip install -r requirements.txt
streamlit run scripts/app.py
