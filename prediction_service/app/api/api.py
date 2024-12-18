from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated
from uuid import UUID, uuid4
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from pydantic import BaseModel
import mlflow.keras
import numpy as np
import os
import sqlite3
import joblib

router = APIRouter()

# Simple user database (just for auth example)
users = {
    "user1": {"username": "user1", "password": "xxxxxx", "user_id": 1},
    "user2": {"username": "user2", "password": "yyyyyy", "user_id": 2},
}

# Session Data Model
class SessionData(BaseModel):
    username: str

# In-memory session backend
backend = InMemoryBackend[UUID, SessionData]()

cookie_params = CookieParameters()
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",  # Replace with a secure key in production
    cookie_params=cookie_params,
)
cookie.cookie_params.max_age = None

class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(self, *, identifier: str, auto_error: bool, backend: InMemoryBackend[UUID, SessionData], auth_http_exception: HTTPException):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        # If session exists, considered valid
        return True

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

security = HTTPBasic()

@router.post("/login")
async def login(response: Response, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    user = users.get(credentials.username)
    if user is None or user["password"] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    session_id = uuid4()
    data = SessionData(username=user["username"])
    await backend.create(session_id, data)
    cookie.attach_to_response(response, session_id)
    return f"created session for {user['username']}"

@router.get("/whoami", dependencies=[Depends(cookie)])
async def whoami(session_data: SessionData = Depends(verifier)):
    return {"username": session_data.username}

@router.post("/delete_session")
async def del_session(response: Response, session_id: UUID = Depends(cookie)):
    await backend.delete(session_id)
    cookie.delete_from_response(response)
    return "deleted session"

@router.get("/predict")
async def predict_stock(days: int = 1, session_data: SessionData = Depends(verifier)):
    # Validate days
    if days < 1 or days > 10:
        raise HTTPException(status_code=400, detail="days must be between 1 and 10")

    # Paths (assuming everything is mounted)
    scaled_db_path = "/shared_model/database/scaled_nvda_stock.db"
    scaler_path = "/shared_model/scaler.pkl"
    model_path = "/shared_model/fine_tuned_model"  # or nvda_stock_model if not fine-tuned

    feature_cols = ["Close", "Change(%)"]  # Adjust as needed
    target_col = "Close"
    timestep = 30  # model input length
    forecast_days = 10  # model always predicts 10 steps

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Load the model
    model = mlflow.keras.load_model(model_path)

    # Connect to DB and get the last 30 rows for input
    conn = sqlite3.connect(scaled_db_path)
    query = f"SELECT * FROM scaled_nvidia_prices ORDER BY Date DESC LIMIT {timestep}"
    df = ( 
        # Reverse after fetching because LIMIT + ORDER BY DESC returns recent first
        # We want oldest to newest in correct order
        pd.read_sql_query(query, conn).iloc[::-1].reset_index(drop=True)
    )
    conn.close()

    if len(df) < timestep:
        raise HTTPException(status_code=500, detail="Not enough data to form input sequence.")

    # Extract the input sequence
    X_input = df[feature_cols].values.reshape(1, timestep, len(feature_cols))

    # Predict with the model (outputs shape: (1,10))
    predictions = model.predict(X_input)
    requested_days = min(days, predictions.shape[1])
    sliced_predictions = predictions[0][:requested_days]

    # Denormalize predictions if needed:
    # Create dummy array to inverse_transform. We must create a full row of features,
    # placing predictions in the target column and leaving others as last known values.
    last_row = df[feature_cols].iloc[-1].values.reshape(1, len(feature_cols))
    future_pred_array = np.zeros((requested_days, len(feature_cols)))
    # Insert predicted scaled values into target column
    target_idx = feature_cols.index(target_col)
    future_pred_array[:, target_idx] = sliced_predictions

    # Inverse transform each predicted day separately (assuming stationary features or replicate last known?)
    # If your scaling was done across entire dataset, you can attempt to inverse all at once.
    # For simplicity: Just inverse the predictions by replacing last known row:
    # This simple approach might not reflect real conditions if other features must be updated too.
    # If not sure, skip inverse transform or handle more sophisticated logic.
    # We'll just inverse transform by placing predictions into a dummy row identical to last_row for each predicted day.
    denormalized_predictions = []
    for pred in sliced_predictions:
        dummy_row = last_row.copy()
        dummy_row[0, target_idx] = pred  # Insert predicted scaled value
        denorm = scaler.inverse_transform(dummy_row)[0, target_idx]
        denormalized_predictions.append(float(denorm))

    return {
        "requested_days": requested_days,
        "predictions_scaled": sliced_predictions.tolist(),
        "predictions_denormalized": denormalized_predictions
    }
