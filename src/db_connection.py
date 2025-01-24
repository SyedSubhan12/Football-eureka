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

def fetch_data(query, conn):
    """
    Fetch data from the SQLite database.

    Args:
        query (str): SQL query to execute.
        conn (sqlite3.Connection): Database connection object.

    Returns:
        pd.DataFrame: Query results as a DataFrame.
    """
    try:
        df = pd.read_sql_query(query, conn)
        print("Data fetched successfully")
        return df
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    db_path = "D:/Football_Match_Outcome_Prediction/data/database.sqlite"
    query = "SELECT * FROM Player_Attributes"

    conn = connect_to_db(db_path)

    if conn:
        data = fetch_data(query, conn)
        if data is not None:
            data.to_csv("./data/match.csv", index=False)
            print("Data exported to ./data/match.csv")
        conn.close()
        print("Database connection closed.")
