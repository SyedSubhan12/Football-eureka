import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
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

def train_svr(X_train, y_train):
    """Trains a Support Vector Regressor (SVR) with a linear kernel using Grid Search for hyperparameter tuning."""
    # Define SVR model with a linear kernel
    svr = SVR(kernel='linear')
    
    # Define parameter grid for Grid Search
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'epsilon': [0.1, 0.01, 0.001]  # Epsilon-insensitive loss function
    }
    
    # Initialize Grid Search
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Fit Grid Search to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_model = grid_search.best_estimator_
    
    return best_model


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
    filepath = "D:/Football_Match_Outcome_Prediction/data/preprocessed_match.csv"
    target_column = "overall_rating"

    logging.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)

    logging.info(f"Dataset columns: {df.columns}")
    logging.info("Splitting data into train and test sets.")
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Train models with timeout handling
    logging.info("Training Linear Regression model.")
    linear_model = train_model_with_timeout(train_linear_regression, X_train, y_train)
    if linear_model:
        logging.info("Linear Regression model trained.")

    logging.info("Training Random Forest Regressor.")
    rf_model = train_model_with_timeout(train_random_forest, X_train, y_train)
    if rf_model:
        logging.info("Random Forest model trained.")

    logging.info("Training XGBoost Regressor.")
    xgb_model = train_model_with_timeout(train_xgboost, X_train, y_train)
    if xgb_model:
        logging.info("XGBoost model trained.")

    logging.info("Training SVR model.")
    svr_model = train_model_with_timeout(train_svr, X_train, y_train)
    if svr_model:
        logging.info("SVR model trained.")

    # Evaluate models
    if linear_model:
        logging.info("Evaluating Linear Regression model.")
        evaluate_model(linear_model, X_test, y_test)

    if rf_model:
        logging.info("Evaluating Random Forest Regressor.")
        evaluate_model(rf_model, X_test, y_test)

    if xgb_model:
        logging.info("Evaluating XGBoost Regressor.")
        evaluate_model(xgb_model, X_test, y_test)

    if svr_model:
        logging.info("Evaluating SVR model.")
        evaluate_model(svr_model, X_test, y_test)
