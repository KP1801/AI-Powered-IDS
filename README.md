# 🛡️ Explainable Stacked-Ensemble Intrusion Detection System

An AI-powered Network Intrusion Detection System (NIDS) that goes beyond the typical
"train one classifier, report accuracy" project. It combines three things that are
rarely put together in one repo:

1. **A stacking ensemble** (Random Forest + XGBoost + Neural Net → Logistic Regression meta-learner)
   instead of a single model, to reduce blind spots any one algorithm has.
2. **Explainable AI** — every alert tells you *which features* drove the
   decision and by how much, not just "attack detected." Implemented from
   scratch in pure Python/NumPy (no `shap`/`numba`), so it runs on
   locked-down machines where native compiled extensions get blocked by
   Application Control policies.
3. **A live-simulation dashboard** (Streamlit) that replays real held-out network
   traffic as if it were arriving in real time, with running stats, per-category
   breakdowns, and a live explanation panel for the most recent alert.

Trained and evaluated on **NSL-KDD**, the standard benchmark dataset for
intrusion-detection research (an improved, de-duplicated version of the classic
KDD Cup 1999 dataset).

---

## Why this is different from most IDS repos on GitHub

| Typical IDS project | This project |
|---|---|
| One model (RF or a single NN) | 3-model stacking ensemble with a learned meta-combiner |
| "Attack / Normal" label only | Attack **category** (DoS / Probe / R2L / U2R) + confidence |
| A confusion matrix in a notebook | A live dashboard simulating real traffic |
| Black-box predictions | Per-alert feature-contribution explanation (which features, which direction) |
| Notebook-only, hard to reuse | Clean `src/` package: preprocess → train → explain → predict, reusable end to end |

---

## Architecture

```
Raw NSL-KDD traffic (41 features per connection)
            │
            ▼
   ┌─────────────────┐
   │  Preprocessing   │  categorical encoding + standard scaling
   └────────┬─────────┘
            ▼
   ┌────────────────────────────────────────────┐
   │            Stacking Ensemble                │
   │  ┌───────────┐ ┌──────────┐ ┌────────────┐  │
   │  │  Random   │ │ XGBoost  │ │  Neural    │  │
   │  │  Forest   │ │          │ │  Net (MLP) │  │
   │  └─────┬─────┘ └────┬─────┘ └─────┬──────┘  │
   │        └───────────┬─────────────┘          │
   │              Logistic Regression             │
   │              (meta-learner)                  │
   └────────────────────┬─────────────────────────┘
                         ▼
        Binary verdict (normal / attack)
        + Attack category (DoS / Probe / R2L / U2R)
                         │
                         ▼
              Feature-contribution explanation
              (pure Python/NumPy decision-path
              walk over the Random Forest
              base-learner — no compiled/native
              dependencies required)
                         │
                         ▼
            Streamlit live dashboard
```

Two models are trained: a **binary** model (fast first-pass filter: is this
malicious at all?) and a **multi-class** model (what kind of attack is it?).
This mirrors how layered detection is often designed in practice.

---

## Results (NSL-KDD official test set — `KDDTest+`, contains attack types not
seen during training, which is the standard, harder evaluation protocol)

**Binary classification (normal vs. attack)**

| Metric | Score |
|---|---|
| Accuracy | 80.6% |
| Precision (weighted) | 85.2% |
| Recall (weighted) | 80.6% |
| F1-score (weighted) | 80.5% |

**Multi-class classification (attack category)**

| Metric | Score |
|---|---|
| Accuracy | 74.4% |
| Precision (weighted) | 86.4% |
| Recall (weighted) | 74.4% |
| F1-score (weighted) | 75.1% |

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.79 | 0.90 | 0.84 | 9,711 |
| DoS | 0.96 | 0.82 | 0.89 | 7,458 |
| Probe | 0.81 | 0.68 | 0.74 | 2,421 |
| R2L | 0.92 | 0.06 | 0.12 | 2,887 |
| U2R | 0.02 | 0.67 | 0.03 | 67 |

**Why R2L/U2R recall is low, and why that's expected and worth discussing:**
NSL-KDD's official test set deliberately includes attack *variants* the model
never saw in training, specifically to test generalization rather than
memorization — this is well documented in IDS literature, and R2L/U2R are the
rarest, stealthiest classes (a few dozen to a few thousand examples out of
125k). Reporting this honestly, rather than hiding it, is intentional: a real
strength of the explainability layer is that it lets an analyst inspect
*why* a borderline U2R/R2L connection was or wasn't flagged, which is exactly
where a black-box model would otherwise fail silently. See **Future Work**
for concrete ways to improve this (oversampling, anomaly-based second stage).

---

## Project structure

```
ai-powered-ids/
├── data/
│   └── download_data.py      # downloads NSL-KDD (with offline synthetic fallback)
├── src/
│   ├── utils.py               # dataset schema, attack-category mapping
│   ├── preprocess.py          # encoding + scaling pipeline
│   ├── ensemble_model.py      # stacking classifier definition
│   ├── train.py                # trains + evaluates both models, saves artifacts
│   ├── predict.py              # clean inference interface
│   └── explain.py              # pure Python/NumPy explanation layer (no shap/numba)
├── dashboard/
│   └── app.py                  # Streamlit live-simulation dashboard
├── models/                     # trained models + metrics (generated)
├── requirements.txt
└── README.md
```

---

## Setup

> **Note:** `models/` starts empty on purpose. You train the models
> locally in step 2 below using whatever library versions `pip` resolves
> for your machine — this avoids scikit-learn's well-known cross-version
> pickle incompatibility (loading a `.pkl` model trained with a different
> scikit-learn version than the one installed can throw an `AttributeError`
> on an internal sklearn attribute). See `TROUBLESHOOTING.md` if you ever
> hit that.

```bash
git clone <your-repo-url>
cd ai-powered-ids
pip install -r requirements.txt

# 1. Download the dataset (falls back to a synthetic dataset if offline)
python data/download_data.py

# 2. Train both models (~6-8 minutes on a typical laptop CPU)
python src/train.py

# 3. Launch the live dashboard
streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), set a
playback speed in the sidebar, and click **▶ Play** (or **⏭ Step one row**
to advance manually).

> **Getting an error right after launch, especially anything mentioning
> `AttributeError` on a scikit-learn class?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
> — it's almost always a library-version mismatch with the pre-shipped
> `.pkl` models, fixed in under a minute by deleting `models/*.pkl` and
> running `python src/train.py` to regenerate them with your own installed
> versions.

### Quick CLI test (no dashboard)

```python
from src.predict import IDSPredictor
from src.explain import IDSExplainer
import numpy as np

X = np.load("models/demo_traffic_sample_scaled.npy")
predictor, explainer = IDSPredictor(), IDSExplainer()

print(predictor.predict(X[:5]))
print(explainer.explain_instance(X[0]))
```

---

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — a refined version of
KDD Cup 1999 that removes redundant records and rebalances difficulty, making
it a fairer benchmark than the original. Each of the 125,973 training and
22,544 test connections is described by 41 features (traffic volume,
connection flags, login behavior, host-based statistics, etc.) and is
labeled either `normal` or with one of 22+ specific attack types, which this
project groups into 4 top-level categories: **DoS, Probe, R2L, U2R**.

---

## Future work

- **Class imbalance for R2L/U2R**: SMOTE / class-weighted focal loss, or a
  dedicated anomaly-detection second stage (e.g. Isolation Forest) for rare
  classes instead of relying on the multi-class softmax alone.
- **Real packet capture input**: swap the CSV replay in the dashboard for a
  live `scapy`/`pyshark` feature extractor with the same 41-feature schema.
- **Adversarial robustness testing**: evaluate how easily an attacker could
  perturb traffic features to evade detection (feature-space adversarial
  examples), and harden the ensemble against it.
- **Model drift monitoring**: track live-accuracy-vs-ground-truth over time
  in production and trigger retraining alerts.

---

## License

This project is released under the MIT License. The NSL-KDD dataset is a
public research benchmark, freely available for academic and research use.
