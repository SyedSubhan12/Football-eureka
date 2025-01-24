import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def load_data(filepath):
    """
    Loads data from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handles missing values in a DataFrame using the specified strategy.
    """
    try:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns  # Automatically detect numeric columns
        imputer = SimpleImputer(strategy=strategy)
        df[columns] = imputer.fit_transform(df[columns])
        print("Missing values handled successfully.")
        return df
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return df

def scale_features(df, columns):
    """
    Scales numerical features using StandardScaler.
    """
    try:
        columns = [col for col in columns if col in df.columns]  # Validate columns
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        print("Features scaled successfully.")
        return df
    except Exception as e:
        print(f"Error scaling features: {e}")
        return df

def encode_categorical(df, columns):
    """
    Encodes categorical features using OneHotEncoder.
    """
    try:
        if not columns:
            print("No categorical columns to encode.")
            return df
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
        df = df.drop(columns, axis=1).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        print("Categorical features encoded successfully.")
        return df
    except Exception as e:
        print(f"Error encoding categorical features: {e}")
        return df

def handle_outliers(df, columns):
    """
    Handles outliers using the IQR method.
    """
    try:
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print("Outliers handled successfully.")
        return df
    except Exception as e:
        print(f"Error handling outliers: {e}")
        return df

def save_data(df, filepath):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"Preprocessed data saved to {filepath}.")
    except Exception as e:
        print(f"Error saving data: {e}")

# Example usage
if __name__ == "__main__":
    filepath = "D:/Football_Match_Outcome_Prediction/data/match.csv"  # Adjust the path based on your folder structure
    output_filepath = "D:/Football_Match_Outcome_Prediction/data/preprocessed_match.csv"

    df = load_data(filepath)

    # Select relevant features and target
    relevant_features = ['potential', 'crossing', 'finishing', 'short_passing',
                         'dribbling', 'ball_control', 'acceleration', 'sprint_speed', 'stamina']
    target = 'overall_rating'

    # Ensure selected columns exist in the DataFrame
    relevant_features = [col for col in relevant_features if col in df.columns]

    df = df[relevant_features + [target]]  # Include only relevant features and target

    # Handle missing values
    df = handle_missing_values(df, strategy='mean')

    # Scale numerical features
    numerical_columns = relevant_features  # Target column is excluded from scaling
    df = scale_features(df, numerical_columns)

    # Encode categorical features (if applicable)
    categorical_columns = []  # Add any categorical columns here
    df = encode_categorical(df, categorical_columns)

    # Handle outliers
    df = handle_outliers(df, numerical_columns)

    # Save preprocessed data
    save_data(df, output_filepath)

    print("Preprocessing complete.")
