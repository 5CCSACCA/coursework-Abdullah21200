import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def load_scaled_data(db_path="app/database/scaled_nvda_stock.db", table_name="scaled_nvidia_prices", months=6):
    """
    Load the last 'months' months of scaled data from the SQLite database.
    Assume data is sorted by date ascending.
    """
    conn = sqlite3.connect(db_path)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=months*30)  # approx months*30 days
    query = f"""
    SELECT * FROM {table_name}
    WHERE Date >= '{start_date}'
    ORDER BY Date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_sequences(data, target_col="Close", timestep=30, forecast_days=1):
    """
    Create sequences for LSTM. 
    data: DataFrame with at least target_col and other features
    timestep: how many past steps to use
    forecast_days: how far ahead to predict (1 = next day)
    """
    # Assuming you want X features: [Close, Change(%)]
    feature_cols = [c for c in data.columns if c not in ["Date"]]
    X, y = [], []
    arr = data[feature_cols].values
    target_idx = feature_cols.index(target_col)
    for i in range(len(arr) - timestep - forecast_days + 1):
        X.append(arr[i:i+timestep])
        y.append(arr[i+timestep:i+timestep+forecast_days, target_idx])
    X, y = np.array(X), np.array(y)
    # If forecast_days=1, y shape might be (samples, 1), reshape if needed
    if forecast_days == 1:
        y = y.reshape(-1)
    return X, y

if __name__ == "__main__":
    # Configuration
    db_path = "database/scaled_nvda_stock.db"
    table_name = "scaled_nvidia_prices"
    timestep = 30
    forecast_days = 1
    months = 6
    batch_size = 16
    fine_tuning_epochs = 10
    learning_rate = 0.00005  # Maybe smaller for fine-tuning
    hidden_units = 50
    dropout = 0.2
    weight_decay = 1e-4

    # MLflow model URI (Adjust this to your registered model URI or run ID)
    model_uri = "runs:/<your_run_id>/model"  # or "models:/YourModelName/Production"

    # Load recent data
    df = load_scaled_data(db_path=db_path, table_name=table_name, months=months)
    print("Loaded recent scaled data:", df.shape)

    # Create sequences
    X, y = create_sequences(df, target_col="Close", timestep=timestep, forecast_days=forecast_days)
    print("X shape:", X.shape, "y shape:", y.shape)

    # Split into training and validation 
    # 10% for validation
    val_split = int(len(X)*0.9)
    X_train, X_val = X[:val_split], X[val_split:]
    y_train, y_val = y[:val_split], y[val_split:]

    # Load previously trained model from MLflow
    model = mlflow.keras.load_model(model_uri)
    # Optionally recompile with a potentially lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    with mlflow.start_run():
        # Log some fine-tuning parameters
        mlflow.log_param("fine_tuning_epochs", fine_tuning_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("timestep", timestep)
        mlflow.log_param("forecast_days", forecast_days)
        mlflow.log_param("months_of_data", months)
        mlflow.log_param("batch_size", batch_size)

        # Fine-tune model
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=fine_tuning_epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping],
                            verbose=1)

        # Log metrics per epoch
        for epoch_i, (tr_l, val_l) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            mlflow.log_metric("train_loss", tr_l, step=epoch_i)
            mlflow.log_metric("val_loss", val_l, step=epoch_i)

            # If you have mae:
            if 'mae' in history.history and 'val_mae' in history.history:
                tr_mae = history.history['mae'][epoch_i]
                val_mae = history.history['val_mae'][epoch_i]
                mlflow.log_metric("train_mae", tr_mae, step=epoch_i)
                mlflow.log_metric("val_mae", val_mae, step=epoch_i)

        # Log final model
        mlflow.keras.log_model(model, "/app/shared_model/nvda_stock_finetuned")

    print("Fine-tuning completed and logged to MLflow.")
