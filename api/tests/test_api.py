import pytest
from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify health endpoint returns status ok and stub mode when models are missing."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] in ["stub", "classical", "transformer"]

def test_analyze_valid_input():
    """Verify /analyze returns correct structure conforming to API contract."""
    sample_text = (
        "We spent three nights at this hotel. The room was spacious and clean, "
        "although the air conditioner was a bit noisy at night. The staff was friendly."
    )
    response = client.post("/analyze", json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()

    # Check authorship
    assert "authorship" in data
    assert 0.0 <= data["authorship"]["ai_probability"] <= 1.0
    assert data["authorship"]["label"] in ["human", "ai"]

    # Check deception
    assert "deception" in data
    assert 0.0 <= data["deception"]["deceptive_probability"] <= 1.0
    assert data["deception"]["label"] in ["genuine", "deceptive"]

    # Check verdict
    assert data["verdict"] in ["genuine_human", "deceptive_human", "ai_genuine", "ai_deceptive"]

    # Check explanations
    assert "explanations" in data
    assert "authorship" in data["explanations"]
    assert "deception" in data["explanations"]
    assert isinstance(data["explanations"]["authorship"], list)
    assert isinstance(data["explanations"]["deception"], list)
    if data["explanations"]["authorship"]:
        first = data["explanations"]["authorship"][0]
        assert "token" in first
        assert "weight" in first

    # Check model info & warnings
    assert data["model_info"]["mode"] in ["stub", "classical", "transformer"]
    assert "Authorship model trained on Amazon-style and movie reviews" in data["warnings"]
    assert "Deception model trained on Chicago hotel reviews only; treat as indicative" in data["warnings"]

def test_analyze_determinism():
    """Verify that same input text yields identical deterministic probabilities and tokens."""
    sample = "A fabulous experience at the Grand Palace hotel. Everything was flawless and perfection."
    res1 = client.post("/analyze", json={"text": sample}).json()
    res2 = client.post("/analyze", json={"text": sample}).json()

    assert res1["authorship"] == res2["authorship"]
    assert res1["deception"] == res2["deception"]
    assert res1["verdict"] == res2["verdict"]
    assert res1["explanations"] == res2["explanations"]

def test_analyze_text_too_short():
    """Verify input under 20 chars fails validation."""
    response = client.post("/analyze", json={"text": "Too short"})
    assert response.status_code == 422

def test_analyze_text_too_long():
    """Verify input over 3000 chars fails validation."""
    long_text = "word " * 700
    response = client.post("/analyze", json={"text": long_text})
    assert response.status_code == 422

def test_analyze_cleaning():
    """Verify that leading chat preambles and HTML tags are cleaned transparently."""
    dirty_text = "Sure! Here is my review of the hotel: We stayed for two nights.<br /><br />Everything was great!"
    response = client.post("/analyze", json={"text": dirty_text})
    assert response.status_code == 200
    data = response.json()
    assert data["authorship"]["label"] in ["human", "ai"]
    assert "Authorship model trained on Amazon-style and movie reviews" in data["warnings"]
    assert "Deception model trained on Chicago hotel reviews only; treat as indicative" in data["warnings"]


