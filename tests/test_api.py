import os
import pytest
from fastapi.testclient import TestClient

# We must ensure the DB dir exists for the test to avoid failures during API startup
os.environ["AEGISAUTH_DB_DIR"] = "./tests/test_db"
os.environ["TESTING"] = "True"
os.makedirs("./tests/test_db", exist_ok=True)

# Import the FastAPI app
from api.main import app

client = TestClient(app)

def test_docs_endpoint():
    """Ensure the Swagger UI is accessible, indicating the app starts correctly."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_verify_no_file():
    """Verify endpoint should fail gracefully if no file is provided."""
    response = client.post("/verify")
    # Validation error from FastAPI for missing required form-data field
    assert response.status_code == 422

def test_enroll_no_file():
    """Enroll endpoint should fail gracefully if no file is provided."""
    response = client.post("/enroll")
    assert response.status_code == 422
