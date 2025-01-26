import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle as pp
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(df, target, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target (str): The target column name.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before the split.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    if target not in df.columns:
        logging.error(f"Target column {target} not found in dataset.")
        return None, None, None, None
    X = df.drop(columns=[target])
    y = df[target]
    logging.info(f"Features: {X.columns.tolist()}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=50, max_depth=5)  # Reduce trees and depth
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Trains an XGBoost Regressor."""
    model = XGBRegressor(n_estimators=50, max_depth=5)  # Reduce trees and depth
    model.fit(X_train, y_train)
    return model

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def train_linear_svr(X_train, y_train):
    """
    Trains a Linear Support Vector Regressor (LinearSVR) using Grid Search for hyperparameter tuning.
    """
    linear_svr = LinearSVR(max_iter=10000)
    param_grid = {
        'C': [0.1, 1, 10],       
        'epsilon': [0.1, 0.01]   
    }
    grid_search = GridSearchCV(
        estimator=linear_svr,
        param_grid=param_grid,
        cv=3,          
        n_jobs=2,       
        verbose=3       
    )
    
    # Set parallel backend
    logging.info("Starting Grid Search with LinearSVR...")
    with joblib.parallel_backend('loky'):  
        grid_search.fit(X_train, y_train)
    logging.info("Grid Search completed.")
    
    return grid_search.best_estimator_

# New function for Logistic Regression
def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.
    """
    y_pred = model.predict(X_test)

    logging.info(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    logging.info(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    logging.info(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

def train_model_with_timeout(model_func, X_train, y_train, timeout=300):
    """
    Trains a model with a specified timeout.
    """
    try:
        with ThreadPoolExecutor() as executor:
            start_time = time.time()
            future = executor.submit(model_func, X_train, y_train)
            model = future.result(timeout=timeout)
            logging.info(f"Model training completed in {time.time() - start_time:.2f} seconds.")
            return model
    except TimeoutError:
        logging.error("Model training timed out")
        return None

# Example workflow
if __name__ == "__main__":
    filepath = "/content/preprocessed_match.csv"
    target_column = "overall_rating"

    logging.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)

    logging.info(f"Dataset columns: {df.columns}")
    logging.info("Splitting data into train and test sets.")
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Train models with timeout handling
    logging.info("Training Linear Regression model.")
    Linear_model = train_model_with_timeout(train_linear_regression, X_train, y_train)
    if Linear_model:
        logging.info("Linear Regression model trained.")
        with open("Linear.pkl", "wb") as f:
          pp.dump(Linear_model, f)
        logging.info("Model trained and saved")

    logging.info("Training Random Forest Regressor.")
    rf_model = train_model_with_timeout(train_random_forest, X_train, y_train)
    if rf_model:
        logging.info("Random Forest model trained.")
        with open("rf.pkl", "wb") as f:
          pp.dump(rf_model, f)
        logging.info("Model trained and saved")

    logging.info("Training XGBoost Regressor.")
    xgb_model = train_model_with_timeout(train_xgboost, X_train, y_train)
    if xgb_model:
        logging.info("XGBoost model trained.")
        with open("xgb.pkl", "wb") as f:
          pp.dump(xgb_model, f)
        logging.info("Model trained and saved")

    logging.info("Training SVR model.")
    svr_model = train_model_with_timeout(train_svr, X_train, y_train)
    if svr_model:
        logging.info("SVR model trained.")
        with open("svr.pkl", "wb") as f:
          pp.dump(svr_model, f)
        logging.info("Model trained and saved")

    logging.info("Training Logistic Regression model.")
    logreg_model = train_model_with_timeout(train_logistic_regression, X_train, y_train)
    if logreg_model:
        logging.info("Logistic Regression model trained.")
        with open("logreg.pkl", "wb") as f:
          pp.dump(logreg_model, f)
        logging.info("Model trained and saved")

    # Evaluate models
    if Linear_model:
        logging.info("Evaluating Linear Regression model.")
        evaluate_model(Linear_model, X_test, y_test)

    if rf_model:
        logging.info("Evaluating Random Forest Regressor.")
        evaluate_model(rf_model, X_test, y_test)

    if xgb_model:
        logging.info("Evaluating XGBoost Regressor.")
        evaluate_model(xgb_model, X_test, y_test)

    if svr_model:
        logging.info("Evaluating SVR model.")
        evaluate_model(svr_model, X_test, y_test)

    if logreg_model:
        logging.info("Evaluating Logistic Regression model.")
        evaluate_model(logreg_model, X_test, y_test)
