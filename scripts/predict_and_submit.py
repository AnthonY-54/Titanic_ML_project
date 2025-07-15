import pandas as pd
import numpy as np
import pickle
import os

from data_preprocessing import preprocess_test_data

# Step 1: Define paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_path = os.path.join(project_root, "data", "test.csv")
model_path = os.path.join(project_root, "models", "best_model.pkl")
output_path = os.path.join(project_root, "submission.csv")

# Step 2: Load test data
test_df = pd.read_csv(test_path)

# Step 3: Preprocess test data (same transformations as training)
X_test = preprocess_test_data(test_df)

# Step 4: Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Create submission file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv(output_path, index=False)
print("✅ Submission file created at:", output_path)
