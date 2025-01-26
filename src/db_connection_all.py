import sqlite3
import pandas as pd

def connect_to_db(db_path):
    """
    Connect to the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Database connection object.
    """
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to SQLite database at {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to SQLite database: {e}")
        return None

def fetch_selected_data(query, conn):
    """
    Fetch specific columns from a table using a query.

    Args:
        query (str): SQL query to execute.
        conn (sqlite3.Connection): Database connection object.

    Returns:
        pd.DataFrame: DataFrame containing the queried data.
    """
    try:
        df = pd.read_sql_query(query, conn)
        print("Data fetched successfully")
        return df
    except sqlite3.Error as e:
        print(f"Error executing query: {e}")
        return None

def save_to_csv(data, output_path):
    """
    Save the DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the CSV file.
    """
    try:
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

if __name__ == "__main__":
    db_path = "D:/Football_Match_Outcome_Prediction/data/database.sqlite"
    conn = connect_to_db(db_path)

    if conn:
        # Define queries for each table with required columns
        queries = {
            "Country": "SELECT name FROM Country",
           
            "Match": "SELECT away_team_goal, home_team_goal, country_id FROM Match",
            "Player": "SELECT player_name, height, weight FROM Player",
            "Player_Attributes": "SELECT date FROM Player_Attributes",
            "Team": "SELECT team_long_name FROM Team"
        }

        # Loop through each query and save the extracted data
        for table_name, query in queries.items():
            print(f"Fetching data for table: {table_name}")
            data = fetch_selected_data(query, conn)
            if data is not None:
                print(f"Fetched {len(data)} rows from {table_name}")
                output_path = f"./data/{table_name}.csv"
                save_to_csv(data, output_path)
        
        conn.close()
        print("Database connection closed.")
