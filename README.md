# 🛡️ AI-Powered Explainable Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A network intrusion detection system that combines a 3-model stacking ensemble (Random Forest + XGBoost + Neural Net), a from-scratch explainability layer, and a live-simulation dashboard — trained and evaluated on NSL-KDD, the standard academic benchmark for intrusion detection research.

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard Overview](https://github.com/KP1801/AI-Powered-IDS/blob/main/Screenshots/Dashboard.png)

### Live Alert Feed + Category Breakdown
> Running stats, color-coded attack feed, and a live category chart

![Alert Feed](screenshots/alert_feed.png)

### Feature-Contribution Explanation
> Every alert shows which features pushed the model toward "attack" (red) or "normal" (blue)

![Explanation Chart](screenshots/explanation_chart.png)

*(Run `streamlit run dashboard/app.py`, click ▶ Play, and save your own screenshots into `screenshots/` to replace these placeholders.)*

---

## ⚙️ How It Works

```
Raw NSL-KDD Traffic (41 fields per connection)
             │
             ▼
┌───────────────────────────────┐
│        Preprocessing          │
│  Encode categoricals + scale  │
└───────────────┬────────────────┘
                ▼
┌────────────────────────────────────────┐
│           Stacking Ensemble             │
│  Random Forest · XGBoost · Neural Net   │
│         └── Logistic Regression ──┘     │
│               (meta-learner)            │
└─────────┬───────────────────┬───────────┘
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│  Verdict + Class  │  │  Feature Explanation │
│ attack/normal +   │  │  pure NumPy decision- │
│ DoS/Probe/R2L/U2R │  │  path attribution     │
└─────────┬──────────┘  └──────────┬───────────┘
          │                        │
          └───────────┬────────────┘
                       ▼
        ┌───────────────────────────────┐
        │     Streamlit Live Dashboard   │
        │      http://localhost:8501     │
        └───────────────────────────────┘
```

---

## 🚀 Features

- **3-Model Stacking Ensemble** — Random Forest + XGBoost + Neural Net (MLP), combined by a Logistic Regression meta-learner instead of relying on a single algorithm
- **Two-Stage Classification** — a fast binary filter (attack vs. normal) followed by a multi-class model that identifies the attack category
- **4 Attack Categories** — DoS, Probe, R2L (Remote-to-Local), and U2R (User-to-Root), grouped from 22+ raw NSL-KDD attack labels
- **From-Scratch Explainability** — every alert reports which features drove the decision and by how much, implemented in pure Python/NumPy with no `shap`/`numba` dependency, so it runs on locked-down machines where native compiled extensions get blocked by Application Control policies
- **Live-Simulation Dashboard** — Streamlit app that replays real held-out NSL-KDD test traffic with running stats, a color-coded alert feed, a category breakdown chart, and a live explanation panel
- **Adversarial Robustness Testing** — a black-box perturbation search that measures how easily an attacker could evade detection by tweaking attacker-controllable features
- **Offline-Friendly** — falls back to a synthetic, schema-matched dataset if the real NSL-KDD mirror is unreachable

---

## 📊 Live Results

From an actual run of this project, evaluated on the official NSL-KDD test set (`KDDTest+`), which deliberately includes attack variants not seen during training:

| Metric | Binary (attack vs. normal) | Multi-class (attack category) |
|---|---|---|
| Accuracy | 80.6% | 74.4% |
| Precision (weighted) | 85.2% | 86.4% |
| Recall (weighted) | 80.6% | 74.4% |
| F1-score (weighted) | 80.5% | 75.1% |

**Per-category breakdown:**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.79 | 0.90 | 0.84 | 9,711 |
| DoS | 0.96 | 0.82 | 0.89 | 7,458 |
| Probe | 0.81 | 0.68 | 0.74 | 2,421 |
| R2L | 0.92 | 0.06 | 0.12 | 2,887 |
| U2R | 0.02 | 0.67 | 0.03 | 67 |

**Adversarial robustness check** (300 known attacks, bounded feature perturbation):

| Metric | Value |
|---|---|
| Caught by model (baseline) | 222 / 300 (74.0% recall) |
| Evaded via bounded perturbation | 117 / 222 (52.7%) |
| Recall after adversarial attempt | 35.0% |

*R2L/U2R recall is low because these are the rarest, stealthiest attack types (a few dozen to a few thousand samples out of 125,973) and the official test set includes variants never seen in training — see **Future Work** for how this project would address it.*

---

## 📂 Project Structure

```
ai-powered-ids/
├── data/
│   └── download_data.py        # downloads NSL-KDD (offline synthetic fallback)
├── src/
│   ├── utils.py                 # dataset schema, attack-category mapping
│   ├── preprocess.py            # encoding + scaling pipeline
│   ├── ensemble_model.py        # stacking classifier definition
│   ├── train.py                 # trains + evaluates both models
│   ├── predict.py                # clean inference interface
│   ├── explain.py                # pure Python/NumPy explanation layer
│   └── adversarial_test.py       # robustness evaluation
├── dashboard/
│   └── app.py                    # Streamlit live-simulation dashboard
├── tests/
│   └── test_pipeline.py          # pytest suite
├── screenshots/                  # dashboard screenshots (add your own)
├── models/                       # trained models + metrics (gitignored, generated)
├── .github/workflows/ci.yml      # GitHub Actions test workflow
├── requirements.txt
├── TROUBLESHOOTING.md
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-powered-ids.git
cd ai-powered-ids
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset
```bash
python data/download_data.py
```

### 4. Train the Models
```bash
python src/train.py
# ~6-8 minutes on a typical laptop CPU
```

### 5. Launch the Dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## 🧪 Testing It Out

```bash
# Run the automated test suite
pip install pytest
pytest tests/ -v

# Check adversarial robustness
python src/adversarial_test.py

# Quick CLI prediction test (no dashboard needed)
python -c "
from src.predict import IDSPredictor
from src.explain import IDSExplainer
import numpy as np

X = np.load('models/demo_traffic_sample_scaled.npy')
predictor, explainer = IDSPredictor(), IDSExplainer()
print(predictor.predict(X[:5]))
print(explainer.explain_instance(X[0]))
"
```

In the dashboard, use **▶ Play** for continuous playback or **⏭ Step one row** to walk through the simulation and inspect each alert's explanation one at a time.

---

## 🤖 ML Models Used

| Model | Purpose |
|---|---|
| Random Forest | Base learner — strong on non-linear tabular splits; also powers the explanation layer |
| XGBoost | Base learner — strong on structured, imbalanced data |
| MLP (Neural Net) | Base learner — catches patterns the tree models miss |
| Logistic Regression | Meta-learner — combines the three base learners' out-of-fold predictions |

**Attack categories classified:**
- `DoS` — Denial of Service (neptune, smurf, back, teardrop, ...)
- `Probe` — surveillance / port scanning (satan, nmap, portsweep, ...)
- `R2L` — Remote-to-Local unauthorized access (guess_passwd, ftp_write, ...)
- `U2R` — User-to-Root privilege escalation (buffer_overflow, rootkit, ...)

---

## 🎛️ Configuration

The dashboard's sidebar exposes runtime controls — no environment variables or config files needed:

| Control | What it does |
|---|---|
| Rows per second | Playback speed during ▶ Play |
| Total rows to simulate | How many held-out test connections to replay |
| ▶ Play / ⏸ Pause | Continuous autoplay |
| ⏭ Step one row | Advance exactly one connection at a time |
| ⟲ Reset | Clear the alert log and stats |

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Machine Learning | scikit-learn, XGBoost, NumPy, pandas |
| Explainability | Pure Python/NumPy (custom decision-path attribution — no shap/numba) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, GitHub Actions CI |
| Dataset | NSL-KDD (KDD Cup 99 successor) |

---

## ⚠️ Notes & Limitations

- **This is a simulation, not a live sensor.** The dashboard replays real, historical NSL-KDD test traffic from a saved CSV/array — it is not watching live packets on your network. See the README's "Future Work" for what a live `scapy`/`pyshark` version would need.
- **`models/` starts empty on purpose.** Always train locally with `python src/train.py` rather than copying `.pkl` files between machines — scikit-learn does not guarantee pickled models load correctly across different library versions. See `TROUBLESHOOTING.md`.
- This project is for **educational and portfolio purposes**. It is not a production security product.

---

## 🔭 Future Work

- **Class imbalance for R2L/U2R** — SMOTE, class-weighted focal loss, or a dedicated anomaly-detection second stage (e.g. Isolation Forest)
- **Real packet capture input** — swap the CSV replay for a live `scapy`/`pyshark` feature extractor with the same 41-feature schema
- **Model drift monitoring** — track live-accuracy-vs-ground-truth over time and trigger retraining alerts

---

## 📄 License

MIT License — free to use, modify, and distribute. The NSL-KDD dataset is a public research benchmark, freely available for academic and research use.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

---

⭐ If this project helped you, consider giving it a star on GitHub!
