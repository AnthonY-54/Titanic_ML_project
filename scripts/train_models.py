# scripts/train_models.py

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data_preprocessing import load_and_preprocess

# 1. Load data
X, y, _ = load_and_preprocess()

# 2. Train-Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM (Linear)": SVC(kernel="linear"),
    "SVM (RBF)": SVC(kernel="rbf")
}

# 4. Train and evaluate
best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    print(f"{name}'s accuracy: {round(score * 100, 2)}%")

    if score > best_score:
        best_score = score
        best_model = model

import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)  

# 5. Save the best model
with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f, protocol=4)

print(f"\n✅ Best Model Saved: {best_model.__class__.__name__} with accuracy {round(best_score * 100, 2)}%")