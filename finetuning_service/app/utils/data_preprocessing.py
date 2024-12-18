import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self, 
                 raw_db_path="/shared_model/database/nvda_stock.db", 
                 raw_table_name="nvidia_prices",
                 scaled_db_path="/shared_model/database/scaled_nvda_stock.db",
                 scaled_table_name="scaled_nvidia_prices",
                 scaler_path="/shared_model/scaler.pkl",
                 feature_cols=["Close", "Change(%)"],
                 target_col="Close"):
        self.raw_db_path = raw_db_path
        self.raw_table_name = raw_table_name
        self.scaled_db_path = scaled_db_path
        self.scaled_table_name = scaled_table_name
        self.scaler_path = scaler_path
        self.feature_cols = feature_cols
        self.target_col = target_col

    def load_data_from_db(self):
        conn = sqlite3.connect(self.raw_db_path)
        query = f"SELECT * FROM {self.raw_table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def compute_percentage_change(self, df):
        if "Change(%)" not in df.columns:
            df["Change(%)"] = df[self.target_col].pct_change()
        df.dropna(subset=[self.target_col, "Change(%)"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def scale_features(self, df):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_features = scaler.fit_transform(df[self.feature_cols].values)
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_cols)
        scaled_df['Date'] = df['Date'].values

        # Save scaler
        joblib.dump(scaler, self.scaler_path)

        # Save scaled data to DB
        conn = sqlite3.connect(self.scaled_db_path)
        scaled_df.to_sql(self.scaled_table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()

        print(f"Scaled data stored in {self.scaled_db_path}, table: {self.scaled_table_name}")
        print("Scaler saved to:", self.scaler_path)
        return scaled_df

    def run(self):
        df = self.load_data_from_db()
        df = self.compute_percentage_change(df)
        scaled_df = self.scale_features(df)
        print("Data preprocessing complete.")
