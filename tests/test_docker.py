import io
import httpx
import pytest
import requests
from fastapi.testclient import TestClient



@pytest.fixture(scope="session")
def client(http_service):
    client = httpx.Client(base_url=http_service)
    yield client
    client.close()


def test_warm_up(client):
    response = client.get("/warmup")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_check_health(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_inference(negative_img, client):
    # Create a sample image for testing
    img_byte_arr = io.BytesIO()
    negative_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Create an UploadFile instance for testing

    response = client.post("/inference", files={"file": ("test_image.png", img_byte_arr, "image/png")})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()

def test_inference_invalid_file(client):
    invalid_file = io.BytesIO(b"Invalid content")
    response = client.post("/inference", files={"file": ("invalid_file.txt", invalid_file, "text/plain")})
    assert response.status_code == 200

    res_dict = response.json()
    assert "message" in res_dict
    assert "exception" in res_dict
    assert res_dict["message"] == "There was an error in inferencing"