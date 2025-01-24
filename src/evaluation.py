import logging
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_logistic_regression, train_random_forest, train_xgboost, train_svm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

def evaluate_regression_model(model, X_test, y_test, model_name, report_file):
    """
    Evaluates a regression model and logs metrics.

    Parameters:
        model: Trained regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.
        model_name (str): Name of the model.
        report_file (str): Path to the report file to save evaluation metrics.

    Returns:
        None
    """
    try:
        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"{model_name} - MSE: {mse}, MAE: {mae}, R2: {r2}")

        # Save metrics to report
        with open(report_file, "a") as f:
            f.write(f"\n{model_name} Evaluation:\n")
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"R2 Score: {r2:.4f}\n")

        # Plot Predictions vs Actual
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name} Predictions vs Actual')
        plot_path = f"plots/{model_name}_predictions_vs_actual.png"
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"{model_name} evaluation plot saved to {plot_path}")

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}")

if __name__ == "__main__":
    # Paths
    data_path = "D:/Football_Match_Outcome_Prediction/data/preprocessed_match.csv"
    report_file = "evaluation_report.txt"
    os.makedirs("plots", exist_ok=True)

    # Clear report file
    open(report_file, "w").close()

    # Load dataset
    logging.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    target_column = "overall_rating"

    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        exit()

    # Split data
    logging.info("Splitting data into train and test sets.")
    from utils import split_data
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Train and evaluate models
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
        "SVM": train_svm(X_train, y_train)
    }

    for model_name, model in models.items():
        logging.info(f"Evaluating {model_name}.")
        evaluate_regression_model(model, X_test, y_test, model_name, report_file)

    logging.info("Evaluation completed. Report saved.")
