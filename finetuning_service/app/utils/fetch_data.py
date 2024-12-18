import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import sqlite3

class DataFetcher:
    def __init__(self, ticker="NVDA", db_path="/shared_model/database/nvda_stock.db", table_name="nvidia_prices", months=6):
        self.ticker = ticker
        self.db_path = db_path
        self.table_name = table_name
        self.months = months

    def fetch_data(self):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.months*30) # approx 6 months
        df = yf.download(self.ticker, start=start_date, end=end_date)
        return df

    def flatten_columns(self, df):
        # colums are multidimensional, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            # Example approach: take just the first level of the column name if not 'Date'
            # If 'Date' is in the index as a column, we will reassign after reset_index
            df.columns = [col[0] if col[0] != 'Date' else 'Date' for col in df.columns]
        return df

    def save_to_db(self, df):
        # Flatten columns before saving if needed
        df = self.flatten_columns(df)

        # Reset index so that Date becomes a column
        df.reset_index(inplace=True)
        # Ensure Date is string to easily store in SQL
        df['Date'] = df['Date'].astype(str)  

        conn = sqlite3.connect(self.db_path)
        df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        print(f"Data saved to {self.table_name} in {self.db_path}")

    def run(self):
        df = self.fetch_data()
        self.save_to_db(df)
        print("Data fetching complete.")
