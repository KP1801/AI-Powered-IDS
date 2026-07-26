"""
dashboard/app.py
Streamlit dashboard that SIMULATES a live network traffic feed and runs it
through the trained stacking-ensemble IDS in real time, showing:
  - a running alert log (with attack category + confidence)
  - live stats (traffic processed, attacks detected, breakdown by category)
  - a SHAP explanation chart for the most recent alert, so you can see
    exactly which features drove the model's decision

Data source: a held-out sample of REAL NSL-KDD test traffic (saved by
src/train.py), replayed row-by-row to simulate a live feed.

DESIGN NOTE (why this file looks the way it does):
Streamlit has two common patterns for "live" apps:
  1. A python for-loop with time.sleep() inside a single script run, writing
     into st.empty() placeholders repeatedly. This is fragile: every
     repeated widget call within one run needs a unique `key`, and the
     exact key-collision rules have changed across Streamlit versions.
  2. "One step per rerun": each script execution advances the simulation by
     exactly one row, then calls st.rerun() to trigger the next frame as a
     brand new, independent script run. Every widget is created exactly
     ONCE per run, so there is no key-collision class of bug at all.
This file uses pattern (2) — it's slightly less smooth visually (a full
rerun per row) but is far more robust across Streamlit versions/environments.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import IDSPredictor
from src.explain import IDSExplainer
from src.utils import MODELS_DIR

st.set_page_config(page_title="AI-Powered IDS", layout="wide", page_icon="🛡️")


@st.cache_resource
def load_models():
    return IDSPredictor(), IDSExplainer()


@st.cache_data
def load_demo_data():
    raw = pd.read_csv(os.path.join(MODELS_DIR, "demo_traffic_sample.csv"))
    scaled = np.load(os.path.join(MODELS_DIR, "demo_traffic_sample_scaled.npy"))
    y_bin = np.load(os.path.join(MODELS_DIR, "demo_traffic_sample_ybin.npy"))
    return raw, scaled, y_bin


def missing_models_notice():
    st.error(
        "No trained models found in `models/`. Run the training pipeline first:\n\n"
        "```\npython data/download_data.py\npython src/train.py\n```"
    )


st.title("🛡️ AI-Powered Intrusion Detection System")
st.caption(
    "Stacking ensemble (Random Forest + XGBoost + Neural Net) with SHAP-based "
    "explainability — simulated live traffic from the NSL-KDD benchmark dataset."
)

if not os.path.exists(os.path.join(MODELS_DIR, "binary_model.pkl")):
    missing_models_notice()
    st.stop()

predictor, explainer = load_models()
raw_df, scaled_arr, y_true = load_demo_data()

# --------------------------- Session state ---------------------------------
if "log" not in st.session_state:
    st.session_state.log = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "attacks": 0, "by_category": {}, "correct": 0}
if "cursor" not in st.session_state:
    st.session_state.cursor = 0          # index of the next row to process
if "playing" not in st.session_state:
    st.session_state.playing = False

# ------------------------- Sidebar controls -------------------------------
st.sidebar.header("Simulation Controls")
speed = st.sidebar.slider("Rows per second", min_value=1, max_value=20, value=5)
n_rows = st.sidebar.slider("Total rows to simulate", min_value=20,
                            max_value=len(raw_df), value=min(150, len(raw_df)))

col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶ Play", use_container_width=True):
    st.session_state.playing = True
if col_b.button("⏸ Pause", use_container_width=True):
    st.session_state.playing = False

step = st.sidebar.button("⏭ Step one row", use_container_width=True)

if st.sidebar.button("⟲ Reset", use_container_width=True):
    st.session_state.log = []
    st.session_state.stats = {"total": 0, "attacks": 0, "by_category": {}, "correct": 0}
    st.session_state.cursor = 0
    st.session_state.playing = False

st.sidebar.divider()
st.sidebar.markdown(
    "**About this demo**\n\n"
    "Traffic is replayed from a held-out sample of real NSL-KDD test "
    "connections (not used in training) to simulate a live feed. "
    "Use ▶ Play for continuous playback or ⏭ Step for manual, one-row-at-a-time control."
)

progress = min(st.session_state.cursor, n_rows) / n_rows if n_rows else 0
st.sidebar.progress(progress, text=f"Row {st.session_state.cursor}/{n_rows}")


# --------------------------- Core: process one row --------------------------
def process_next_row():
    """Advance the simulation by exactly one row. Returns False if there is
    nothing left to process."""
    i = st.session_state.cursor
    if i >= min(n_rows, len(raw_df)):
        st.session_state.playing = False
        return False

    x_scaled = scaled_arr[i:i + 1]
    result = predictor.predict(x_scaled)[0]

    verdict = "🚨 ATTACK" if result["is_attack"] else "✅ normal"
    ground_truth = "attack" if y_true[i] == 1 else "normal"
    correct = (result["is_attack"] == bool(y_true[i]))

    entry = {
        "Row": i, "Verdict": verdict,
        "Category": result["category"],
        "Confidence": f"{result['category_confidence']*100:.1f}%",
        "Protocol": raw_df.iloc[i]["protocol_type"],
        "Service": raw_df.iloc[i]["service"],
        "Src Bytes": int(raw_df.iloc[i]["src_bytes"]),
        "Ground Truth": ground_truth,
        "_sample_idx": i,
    }
    st.session_state.log.append(entry)

    s = st.session_state.stats
    s["total"] += 1
    if result["is_attack"]:
        s["attacks"] += 1
        s["by_category"][result["category"]] = s["by_category"].get(result["category"], 0) + 1
    if correct:
        s["correct"] += 1

    st.session_state.cursor += 1
    return True


# Advance the simulation by one row BEFORE drawing anything, if requested.
if step:
    process_next_row()
elif st.session_state.playing:
    process_next_row()

# ------------------------------ Rendering -----------------------------------
# Everything below runs exactly once per script execution -- no placeholders,
# no repeated widget calls, so no key-collision issue is possible.

s = st.session_state.stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Connections analyzed", s["total"])
col2.metric("Attacks detected", s["attacks"])
rate = (s["attacks"] / s["total"] * 100) if s["total"] else 0
col3.metric("Attack rate", f"{rate:.1f}%")
acc = (s["correct"] / s["total"] * 100) if s["total"] else 0
col4.metric("Live accuracy vs ground truth", f"{acc:.1f}%")

col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.subheader("Live Alert Feed")
    if not st.session_state.log:
        st.info("Press ▶ Play or ⏭ Step in the sidebar to begin.")
    else:
        df_log = pd.DataFrame(st.session_state.log[::-1][:50]).drop(columns=["_sample_idx"])

        def highlight(row):
            color = "background-color: #ffe0e0" if row["Verdict"] == "🚨 ATTACK" else ""
            return [color] * len(row)

        st.dataframe(df_log.style.apply(highlight, axis=1), use_container_width=True, height=520)

with col_right:
    st.subheader("Category Breakdown")
    if not s["by_category"]:
        st.info("No attacks detected yet.")
    else:
        fig = go.Figure(data=[go.Bar(
            x=list(s["by_category"].keys()), y=list(s["by_category"].values()),
            marker_color="#e74c3c",
        )])
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why was the last alert flagged?")
    if st.session_state.log:
        last_idx = st.session_state.log[-1]["_sample_idx"]
        x_row = scaled_arr[last_idx]
        exp = explainer.explain_instance(x_row)
        feats = [f["feature"] for f in exp["top_features"]]
        impacts = [f["impact"] for f in exp["top_features"]]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in impacts]

        fig2 = go.Figure(go.Bar(x=impacts, y=feats, orientation="h", marker_color=colors))
        fig2.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            title=f"SHAP impact (red = towards ATTACK, blue = towards NORMAL) "
                  f"— confidence {exp['confidence']*100:.1f}%",
            xaxis_title="SHAP value",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No alert selected yet.")

# --------------------------- Autoplay driver --------------------------------
# If playing and there's more data, wait briefly then trigger the next
# script run. This is the ENTIRE autoplay mechanism -- no in-run loop.
if st.session_state.playing and st.session_state.cursor < min(n_rows, len(raw_df)):
    time.sleep(1.0 / speed)
    st.rerun()
