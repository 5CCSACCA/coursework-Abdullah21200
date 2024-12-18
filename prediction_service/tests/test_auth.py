import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_login_success():
    response = client.post("/login", auth=("user1", "xxxxxx"))
    assert response.status_code == 200
    assert "created session for user1" in response.text
    # Check if cookie was set
    assert "Set-Cookie" in response.headers

def test_login_failure():
    response = client.post("/login", auth=("wronguser", "wrongpass"))
    assert response.status_code == 401
