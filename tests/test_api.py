from fastapi.testclient import TestClient
from src.api.server import app

# Create a test client
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "active"
    assert "System" in data["message"]

def test_upload_invalid_file_type():
    # Attempting to upload a .txt file should fail
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]