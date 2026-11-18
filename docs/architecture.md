# System Architecture

## Project Overview
**Title:** Detection of Fake and AI-Generated Reviews: A Combined Approach for E-Commerce Trust and Safety  
**Degree Project:** M.Sc. Data Science Minor Project

The goal of this system is to evaluate online reviews (hotel, e-commerce, product) across two orthogonal dimensions:
1. **Authorship:** Human-written vs. AI-generated.
2. **Deception:** Genuine vs. Deceptive / Fake.

These two dimensions are never conflated into a single label. A combiner maps the 2D prediction space into one of four distinct actionable verdicts:
- `genuine_human`: Likely genuine, human-written
- `deceptive_human`: Likely deceptive, human-written
- `ai_genuine`: Likely AI-written, no sign of deception
- `ai_deceptive`: Likely AI-generated and deceptive

---

## High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                 Next.js Frontend                                  |
|  - Modern responsive UI (Tailwind CSS, TypeScript, Next.js App Router)            |
|  - Review input textarea (20 - 3000 chars) & 3 quick-test review buttons          |
|  - Authorship and Deception probability gauges                                    |
|  - Combiner Verdict Card (with 4 distinct states & descriptive copy)              |
|  - Word-level explanation highlight board (bipolar green/red weight encoding)     |
|  - Demo Mode banner (activated whenever models are running in stub mode)          |
+---------------------------------------------------------+-------------------------+
                                                          | HTTP POST /analyze
                                                          | HTTP GET /health
                                                          v
+-----------------------------------------------------------------------------------+
|                                FastAPI Backend Service                            |
|  - CORS-enabled REST API with Pydantic validation (length 20-3000 chars)          |
|  - Dynamic Model Manager (auto-detects stub, classical, or transformer mode)     |
|                                                                                   |
|  +------------------------+  +------------------------+  +---------------------+  |
|  | Authorship Predictor   |  | Deception Predictor    |  | Explanation Service |  |
|  | (Stub / Classical /    |  | (Stub / Classical /    |  | (Heuristic / SHAP)  |  |
|  |  DistilBERT)           |  |  DistilBERT)           |  |                     |  |
|  +-----------+------------+  +-----------+------------+  +----------+----------+  |
|              \                           /                          |             |
|               \                         /                           |             |
|                v                       v                            |             |
|              +---------------------------+                          |             |
|              |     Combiner Service      |                          |             |
|              | (Four Orthogonal Verdicts)|                          |             |
|              +-------------+-------------+                          |             |
|                            |                                        |             |
|                            +-------------------+--------------------+             |
|                                                v                                  |
|                                     JSON API Response Payload                     |
+------------------------------------------------+----------------------------------+
                                                 |
                               +-----------------+-----------------+
                               | Stored Models (api/models/)       |
                               | - DistilBERT checkpoints          |
                               | - Classical TF-IDF + Calibrated   |
                               |   SVM / Logistic Regression       |
                               +-----------------+-----------------+
                                                 ^
                                                 | Trained weights exported from
                               +-----------------+-----------------+
                               | Offline Training Pipeline (ml/)   |
                               | - Google Colab (GPU accelerated)  |
                               | - DistilBERT fine-tuning          |
                               | - Multi-dataset unified schema    |
                               +-----------------------------------+
```

---

## API Contract Summary

- **GET `/health`**
  - Response: `{"status": "ok", "mode": "stub" | "classical" | "transformer"}`
- **POST `/analyze`**
  - Request: `{"text": "string (20-3000 chars)"}`
  - Response:
    ```json
    {
      "authorship": {"ai_probability": 0.12, "label": "human"},
      "deception": {"deceptive_probability": 0.88, "label": "deceptive"},
      "verdict": "deceptive_human",
      "explanations": {
        "authorship": [{"token": "word", "weight": -0.42}],
        "deception": [{"token": "amazing", "weight": 0.65}]
      },
      "model_info": {"mode": "stub"},
      "warnings": ["Demo mode: models not loaded"]
    }
    ```

