# Advertising ML Pipeline

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -----------------------------
# Preprocess Data
# -----------------------------
def preprocess_data(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y


# -----------------------------
# Build Pipelines
# -----------------------------
def build_linear_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    return pipeline


def build_polynomial_pipeline(degree=2):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    return pipeline


# -----------------------------
# Train & Evaluate
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return mae, rmse, r2


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    
    # Load dataset
    df = load_data("dataset.csv")
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Models
    models = {
        "Linear Regression": build_linear_pipeline(),
        "Polynomial Regression": build_polynomial_pipeline(2)
    }
    
    # Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)
        
        print(f"\n{name}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")
