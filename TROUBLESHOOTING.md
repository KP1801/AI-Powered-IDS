# Troubleshooting

## `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`
(or similar `AttributeError`/`TypeError` immediately after loading a `.pkl` model)

**Cause:** you have `.pkl` model files in `models/` that were trained with a
different version of scikit-learn (or xgboost) than the one currently
installed on your machine — for example if you copied `models/*.pkl` over
from another computer. Pickled scikit-learn models are tied to the exact
library version that created them — this is a known scikit-learn
limitation, not a bug in this project's code.

**Fix — always train models locally, never copy `.pkl` files between machines:**

```bash
# From the project root, with your virtual environment activated:
del models\*.pkl        # Windows
rm models/*.pkl          # macOS/Linux

python data/download_data.py
python src/train.py      # regenerates models with YOUR installed versions
streamlit run dashboard/app.py
```

This is why the project ships with an empty `models/` folder by default —
`pip install -r requirements.txt` followed by `python src/train.py` always
produces models that match whatever library versions `pip` resolved for
your specific machine, so this error can't happen on a fresh setup.

---

## `ImportError: DLL load failed ... Application Control policy has blocked this file`
(usually mentions `numba`, `llvmlite`, or a file like `_box.pyd`)

**Cause:** this is **not** a Python or project bug — it's Windows itself
(via an Application Control / WDAC policy, common on corporate or managed
laptops) refusing to load a specific compiled native binary. It typically
shows up via the `numba` package, which `shap` used to depend on internally
for some utility functions.

**Fix:** this project's explainability layer (`src/explain.py`) does **not**
use `shap` or `numba` — it's implemented from scratch in pure Python/NumPy
specifically to avoid this class of problem. If you're seeing this error,
make sure you have the current version of `src/explain.py` (it should have
no `import shap` line at all) and that `shap` isn't listed in
`requirements.txt`. If you installed an older copy of this project, pull/
copy the latest `src/explain.py` and `requirements.txt` and reinstall:
```bash
pip install -r requirements.txt
```

If you hit a *similar* DLL-blocked error on a **different** package (not
numba/shap), that means your organization's Application Control policy is
blocking some other native extension used by pandas/numpy/scikit-learn/
xgboost. Those are far more load-bearing and harder to route around — in
that case, talk to your IT/security team about an exception, or try running
the project in a different environment (e.g. a personal machine, WSL, or a
cloud dev environment) where you have full control over installed software.

---

## `streamlit: command not found` / `ModuleNotFoundError: No module named 'streamlit'`
Your virtual environment isn't activated, or dependencies aren't installed:
```bash
pip install -r requirements.txt
```

## Dashboard shows "No trained models found in `models/`"
You haven't trained yet:
```bash
python data/download_data.py
python src/train.py
```

## Training seems to hang
The stacking ensemble trains 3 base models × multiple cross-validation folds
for both the binary and multi-class classifier — this legitimately takes
several minutes (roughly 6-8 minutes total on a typical laptop CPU). If it's
been more than ~20 minutes, something is likely wrong; check you're not
accidentally training on a much larger dataset than default.

## Windows: `pip install` fails on a specific package
Some packages (e.g. `xgboost`, `numpy`) compile C extensions and
occasionally lack prebuilt wheels for very new Python versions on Windows.
If a single package fails, try:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
If it still fails, note the specific package/error and either downgrade
Python to a more widely-supported version (3.10-3.12 tend to have the best
wheel availability) or install that one package with a slightly older
version pin.
