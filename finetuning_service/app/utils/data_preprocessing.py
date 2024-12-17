import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_data_from_db(db_path="/app/database/nvda_stock.db", table_name="nvidia_prices"):
    """
    Load data from the SQLite database.
    Returns a DataFrame with Date, Close, and potentially other columns.
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date ascending
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def compute_percentage_change(df, target_col="Close"):
    """
    Compute percentage change from the previous day if Change(%) doesn't exist.
    Drops rows with NaN created by pct_change.
    """
    if "Change(%)" not in df.columns:
        df["Change(%)"] = df[target_col].pct_change()
    # Drop initial NaN from pct_change
    df.dropna(subset=[target_col, "Change(%)"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def scale_features(df, feature_cols, scaler_path="scaler.pkl", 
                   scaled_db_path="/app/database/scaled_nvda_stock.db", 
                   scaled_table_name="scaled_nvidia_prices"):
    """
    Scale the selected features using MinMaxScaler and store in a SQLite database.
    Also save the scaler for future use.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_features = scaler.fit_transform(df[feature_cols].values)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_df['Date'] = df['Date'].values

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    # Save scaled data to SQLite database
    conn = sqlite3.connect(scaled_db_path)
    # If table exists, replace it
    scaled_df.to_sql(scaled_table_name, conn, if_exists='replace', index=False)

    # Optional: Ensure no duplicates if necessary
    conn.execute(f"""
        DELETE FROM {scaled_table_name}
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM {scaled_table_name}
            GROUP BY Date
        );
    """)
    conn.commit()
    conn.close()

    print(f"Scaled data stored in {scaled_db_path}, table: {scaled_table_name}")
    print("Scaler saved to:", scaler_path)
    return scaled_df

if __name__ == "__main__":
    # Settings
    raw_db_path = "/app/database/nvda_stock.db"
    raw_table_name = "nvidia_prices"
    feature_cols = ["Close", "Change(%)"]
    target_col = "Close"
    scaler_path = "/app/scaler.pkl"
    scaled_db_path = "database/scaled_nvda_stock.db"
    scaled_table_name = "/app/database/scaled_nvidia_prices"

    # Load raw data from DB
    df = load_data_from_db(db_path=raw_db_path, table_name=raw_table_name)

    # Compute percentage change if not present
    df = compute_percentage_change(df, target_col=target_col)

    # Scale features and store in SQLite DB
    scaled_df = scale_features(df, feature_cols, 
                               scaler_path=scaler_path, 
                               scaled_db_path=scaled_db_path, 
                               scaled_table_name=scaled_table_name)

    print("Data preprocessing complete. Scaled data stored in database and scaler saved.")
