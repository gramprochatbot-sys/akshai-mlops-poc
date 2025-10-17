"""
API TESTING
"""
from fastapi.testclient import TestClient
import app as app_module
import pytest

def test_train_flow():
    client = TestClient(app_module.app)
    sample = [5.1, 3.5, 1.4, 0.2]
    resp = client.post("/predict", json={"data": sample})

    assert resp.status_code == 200
    print("Status code:", resp.status_code)
    print("Response JSON:", resp.json())

@pytest.mark.parametrize(
    "sample,species_name",
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),       # Iris-setosa
        ([6.0, 2.9, 4.5, 1.5], "versicolor"),   # Iris-versicolor
        ([6.9, 3.1, 5.4, 2.1], "virginica"),    # Iris-virginica
    ],
)
def test_train_flow_species(sample, species_name):
    """
    Test /predict endpoint for different Iris species samples
    """
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"data": sample})

    assert resp.status_code == 200, f"Failed for {species_name}"
    result = resp.json()

    print(f"✅ Species: {species_name}")
    print("Status code:", resp.status_code)
    print("Response JSON:", result)

    # optional: sanity check on output
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 1