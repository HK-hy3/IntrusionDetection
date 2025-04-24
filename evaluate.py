import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Evaluation function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n📊 Evaluation Results for {name.upper()} Kernel:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print("\n🧾 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Define model names
kernels = ["linear", "poly", "rbf", "sigmoid"]

# Evaluate each saved model
for kernel in kernels:
    model_path = f"Models\svm_{kernel}_model.pkl"
    model = joblib.load(model_path)
    evaluate_model(kernel, model, X_test, y_test)
