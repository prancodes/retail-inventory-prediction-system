import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import setup_logging

logger = setup_logging()

# Industry Standard Feature Selection for tabular model pipelines
REG_FEATURES = [
    'Holiday_Flag', 
    'Temperature', 
    'Fuel_Price', 
    'CPI', 
    'Unemployment', 
    'Weekly_Sales_Lag1',  # Past week's sales (Time Series Lag feature)
    'Month', 
    'Week'
]

def train_simple_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_col: str = 'Temperature') -> tuple:
    """
    Trains a Simple Linear Regression model using exactly one input feature.
    """
    logger.info(f"Training Simple Linear Regression with predictor feature: '{feature_col}'")
    
    X_train = train_df[[feature_col]]
    y_train = train_df['Weekly_Sales']
    X_test = test_df[[feature_col]]
    y_test = test_df['Weekly_Sales']
    
    # Pipeline: Scale data -> Fit model. Bundling prevents data leakage.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    logger.info(f"Simple Regression Metrics on Test: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    return pipeline, metrics, preds

def train_multiple_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Trains a Multiple Linear Regression model using environmental and lagged features.
    """
    logger.info("Training Multiple Linear Regression with features: " + str(REG_FEATURES))
    
    X_train = train_df[REG_FEATURES]
    y_train = train_df['Weekly_Sales']
    X_test = test_df[REG_FEATURES]
    y_test = test_df['Weekly_Sales']
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    logger.info(f"Multiple Regression Metrics on Test: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    return pipeline, metrics, preds

def train_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold_percentile: float = 0.8) -> tuple:
    """
    Classifies if weekly sales will spike (>80th percentile of historical sales).
    Trains a Logistic Regression classifier on features.
    """
    logger.info(f"Training Logistic Regression for high-demand spikes (> {threshold_percentile*100:.0f}th percentile).")
    
    # Calculate spike threshold ONLY from train data to prevent lookahead target leakage
    sales_threshold = train_df['Weekly_Sales'].quantile(threshold_percentile)
    logger.info(f"Spike sales threshold (based on training data): {sales_threshold:.2f}")
    
    # Create binary labels: 1 if sales exceeds threshold, 0 otherwise
    y_train = (train_df['Weekly_Sales'] > sales_threshold).astype(int)
    y_test = (test_df['Weekly_Sales'] > sales_threshold).astype(int)
    
    X_train = train_df[REG_FEATURES]
    X_test = test_df[REG_FEATURES]
    
    # Pipeline: Scale -> Logistic Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1] # Out of stock / spike probability
    
    # Calculate Classification Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    logger.info(f"Logistic Regression Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    return pipeline, metrics, preds, probs, y_test
