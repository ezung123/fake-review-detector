# Detection of Fake and AI-Generated Reviews: A Combined Approach for E-Commerce Trust and Safety

**Degree:** M.Sc. Data Science Minor Project  
**Operating System:** Windows  
**Architecture:** Next.js (App Router, TypeScript, Tailwind CSS) + FastAPI (Python 3.10+)

---

## 1. Project Overview

Online reviews suffer from two distinct integrity threats that must not be conflated:
1. **Authorship Risk:** Synthetic vs. Human origin (e.g., LLM-generated hallucinated text vs. genuine human writing).
2. **Deception Risk:** Fraudulent vs. Truthful intent (e.g., paid promotional spam, competitor slander vs. authentic customer experience).

This system evaluates both dimensions independently, feeds them into a 2D combiner matrix, and delivers one of four actionable verdicts:
- `genuine_human`: Likely genuine, human-written
- `deceptive_human`: Likely deceptive, human-written
- `ai_genuine`: Likely AI-written, no sign of deception
- `ai_deceptive`: Likely AI-generated and deceptive

Along with the verdict, an explanation service highlights the specific tokens pushing each judgment.

---

## 2. Directory Structure

```text
fake-review-detector/
  README.md
  .gitignore
  docs/
    architecture.md          # Full layered architecture and flow diagram
    decisions.md             # Architectural decision records (ADRs)
  web/                       # Next.js 15 App Router Frontend
    package.json
    tsconfig.json
    tailwind.config.ts
    src/
      app/                   # Layout, page, styling
      components/
      types/                 # TypeScript API contract definitions
  api/                       # FastAPI Backend Service
    app/
      main.py                # FastAPI routes (/health, /analyze, CORS)
      config.py              # Decision thresholds & configuration
      schemas.py             # Pydantic v2 schemas and validation
      services/
        analyzer.py          # Dual predictor + explanation pipeline + stub mode
        combiner.py          # Orthogonal combiner mapping
    models/                  # Trained checkpoints (stub fallback if missing)
    tests/
      test_api.py            # Pytest test suite
    requirements.txt         # Pinned backend dependencies
  ml/                        # Offline Training Pipeline (Google Colab GPU)
    notebooks/               # Colab-ready Jupyter notebooks
    src/
      data/adapters/         # Dataset adapters (Ott, Salminen, RAID)
      features/              # Stylometric & linguistic feature extraction
    reports/                 # Metrics JSON & evaluation plots
  data/
    raw/                     # Raw datasets placed manually (git-ignored)
    processed/               # Unified processed train/val/test splits
```

---

## 3. Quickstart: Running Both Servers on Windows

### Prerequisites
- Python 3.10+ (Installed at `C:\Program Files\Python313` or in PATH)
- Node.js 20+ & npm (Installed at `C:\Program Files\nodejs` or in PATH)

---

### Step A: Start the FastAPI Backend

Open a PowerShell terminal:

```powershell
# Navigate to project root
cd C:\fake-review-detector

# Set PYTHONPATH to project root
$env:PYTHONPATH = "C:\fake-review-detector"

# Start FastAPI server on port 8000
python -m uvicorn api.app.main:app --reload --port 8000 --host 127.0.0.1
```

Verify backend health in browser or PowerShell:
```powershell
curl http://127.0.0.1:8000/health
# Response: {"status":"ok","mode":"stub"}
```

---

### Step B: Start the Next.js Web Frontend

Open a second PowerShell terminal:

```powershell
# Navigate to web directory
cd C:\fake-review-detector\web

# Start Next.js development server on port 3000
npm run dev
```

Open your browser and navigate to:
**[http://localhost:3000](http://localhost:3000)**

---

### Step C: Run Automated API Tests

```powershell
cd C:\fake-review-detector
$env:PYTHONPATH = "C:\fake-review-detector"
python -m pytest api/tests -v
```

---

## 4. API Contract Reference

### `GET /health`
Returns system status and active inference mode (`"stub"`, `"classical"`, or `"transformer"`).

### `POST /analyze`
**Request Body:**
```json
{
  "text": "Review text between 20 and 3000 characters."
}
```

**Response Payload:**
```json
{
  "authorship": {
    "ai_probability": 0.68,
    "label": "ai"
  },
  "deception": {
    "deceptive_probability": 0.33,
    "label": "genuine"
  },
  "verdict": "ai_genuine",
  "explanations": {
    "authorship": [
      {"token": "quintessential", "weight": 0.78},
      {"token": "smelled", "weight": -0.65}
    ],
    "deception": [
      {"token": "flawless", "weight": 0.82},
      {"token": "pricey", "weight": -0.44}
    ]
  },
  "model_info": {
    "mode": "stub"
  },
  "warnings": [
    "Demo mode: models not loaded. Showing deterministic heuristic estimates."
  ]
}
```

---

## 5. Phase Status

- [x] **Phase 1: Scaffold + Stub App** (Completed)
  - Full project structure scaffolded
  - FastAPI `/health` and `/analyze` with deterministic hash stub mode
  - Comprehensive Pytest test suite (100% passing)
  - Next.js Web App with dual probability bars, 2x2 verdict card, bipolar word attribution highlights, sample reviews, demo mode banner
- [ ] **Phase 2: Data Adapters & Unified Dataset Pipeline** (Awaiting Phase 2 trigger)
- [ ] **Phase 3: Features & Classical Baselines**
- [ ] **Phase 4: DistilBERT Colab Training Notebooks**
- [ ] **Phase 5: Real Model Integration & SHAP Explanations**
- [ ] **Phase 6: Polish & Architecture Documentation**

