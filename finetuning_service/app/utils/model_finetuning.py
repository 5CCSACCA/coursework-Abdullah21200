import os
import shutil
import numpy as np
import pandas as pd
import sqlite3
import mlflow
import mlflow.keras
from datetime import datetime, timedelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class ModelFinetuner:
    def __init__(self,
                 scaled_db_path="/shared_model/database/scaled_nvda_stock.db",
                 scaled_table_name="scaled_nvidia_prices",
                 model_path="/shared_model/nvda_stock_model", # local directory where model is stored
                 fine_tuned_model_path="/shared_model/fine_tuned_model", # local directory to save fine-tuned model
                 timestep=30,
                 forecast_days=10,
                 months=6,
                 batch_size=32,
                 fine_tuning_epochs=10,
                 learning_rate=0.0005,
                 target_col="Close"):
        self.scaled_db_path = scaled_db_path
        self.scaled_table_name = scaled_table_name
        self.model_path = model_path  # path to directory containing the model
        self.fine_tuned_model_path = fine_tuned_model_path
        self.timestep = timestep
        self.forecast_days = forecast_days
        self.months = months
        self.batch_size = batch_size
        self.fine_tuning_epochs = fine_tuning_epochs
        self.learning_rate = learning_rate
        self.target_col = target_col

    def load_recent_data(self):
        conn = sqlite3.connect(self.scaled_db_path)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.months*30)
        query = f"""
        SELECT * FROM {self.scaled_table_name}
        WHERE Date >= '{start_date}'
        ORDER BY Date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def create_sequences(self, df):
        feature_cols = [c for c in df.columns if c not in ["Date"]]
        arr = df[feature_cols].values
        target_idx = feature_cols.index(self.target_col)
        X, y = [], []
        for i in range(len(arr)-self.timestep-self.forecast_days+1):
            X.append(arr[i:i+self.timestep])
            y.append(arr[i+self.timestep:i+self.timestep+self.forecast_days, target_idx])
        X, y = np.array(X), np.array(y)
        if self.forecast_days == 1:
            y = y.reshape(-1)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def run(self):
        with mlflow.start_run():
            # Log fine-tuning parameters
            mlflow.log_param("fine_tuning_epochs", self.fine_tuning_epochs)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("timestep", self.timestep)
            mlflow.log_param("forecast_days", self.forecast_days)
            mlflow.log_param("months_of_data", self.months)
            mlflow.log_param("batch_size", self.batch_size)

            df = self.load_recent_data()
            X, y = self.create_sequences(df)

            # Simple train/validation split
            val_split = int(len(X)*0.9)
            X_train, X_val = X[:val_split], X[val_split:]
            y_train, y_val = y[:val_split], y[val_split:]
            print(f"X_train data shape: {X_train.shape}, y_train data shape: {y_train.shape}")

            # Load model from local path
            # The model_path should contain an MLflow model
            model = mlflow.keras.load_model(self.model_path)
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=self.fine_tuning_epochs,
                                batch_size=self.batch_size,
                                callbacks=[early_stopping],
                                verbose=1)

            # Log metrics per epoch
            for epoch_i, (tr_l, val_l) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                mlflow.log_metric("train_loss", tr_l, step=epoch_i)
                mlflow.log_metric("val_loss", val_l, step=epoch_i)
            if 'mae' in history.history and 'val_mae' in history.history:
                for epoch_i, (tr_mae, val_mae) in enumerate(zip(history.history['mae'], history.history['val_mae'])):
                    mlflow.log_metric("train_mae", tr_mae, step=epoch_i)
                    mlflow.log_metric("val_mae", val_mae, step=epoch_i)
            
            if os.path.exists(self.fine_tuned_model_path): # remove the existing model and save the new
                shutil.rmtree(self.fine_tuned_model_path)

            # Save the updated model back to the local path
            mlflow.keras.save_model(model, self.fine_tuned_model_path)

        print(f"Model fine-tuning completed and updated model saved at {self.fine_tuned_model_path}.")
