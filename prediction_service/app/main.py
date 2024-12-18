from fastapi import FastAPI
from api.api import router
from utils.db import init_db

app = FastAPI()

init_db()  # Initialize DB and ensure user table, test user

app.include_router(router)

