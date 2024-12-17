import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import sqlite3


def fetch_last_6_month_data(ticker="NVDA"):
    """"
    Fetch the last 6 month data for a stock from Yahoo Finance"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def save_data_to_db(df, db_path="app/database/nvda_stock.db", table_name="nvidia_prices"):

    """"
    Store the fetched data in a SQLite database
    If the table does not exist, it will be created"""

    conn = sqlite3.connect(db_path)

    # Flatten MultiIndex columns

    df = df.copy()

    df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]

    df.reset_index(inplace=True)
    
    df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')



    #create table if not exists
    df.to_sql(table_name, conn, if_exists="replace", index=False)



    # remove any existing data
    conn.execute(f"""
        DELETE FROM {table_name}
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM {table_name}
            GROUP BY Date
        );
    """)
    conn.commit()
    conn.close()

    print(f"Data saved to {table_name} table")

if __name__ == "__main__":
    df = fetch_last_6_month_data()
    save_data_to_db(df)
    print("Data saved to database")

