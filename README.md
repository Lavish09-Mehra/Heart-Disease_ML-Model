# Heart Disease Risk Predictor

A small Streamlit app that predicts the likelihood of heart disease from a
patient's clinical values, using a Logistic Regression model trained on the
UCI heart-disease dataset (918 records, `heart.csv`).

> ⚠️ Educational project only. This is **not** medical advice — always consult
> a qualified doctor for a real diagnosis.

---

## Quick start

```bash
cd /home/lavish09/Desktop/try
pip install -r requirements.txt   # or use the existing ~/jupyter-env
./run.sh
```

Then open **http://127.0.0.1:8501**.

If `./run.sh` is not executable:

```bash
chmod +x run.sh
./run.sh
```

Manual start:

```bash
python -m streamlit run server.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

---

## Inputs

| Field | Type | Notes |
|---|---|---|
| Age | years | |
| Sex | M / F | |
| ChestPainType | ASY / ATA / NAP / TA | |
| RestingBP | mm Hg | |
| Cholesterol | mg/dL | collected, but not used by the final model (see caveat) |
| FastingBS | 0 / 1 | 1 = fasting blood sugar > 120 mg/dL |
| RestingECG | Normal / ST / LVH | |
| MaxHR | bpm | |
| ExerciseAngina | Y / N | |
| Oldpeak | float | ST depression |
| ST_Slope | Up / Flat / Down | |

The app returns an estimated risk percentage and a HIGH / LOW label
(threshold = 50%).

---

## Project structure

```
try/
├── server.py           # Streamlit app (the active frontend + prediction logic)
├── LR_model.pkl        # trained LogisticRegression (15 features)
├── scaler.pkl          # StandardScaler fitted on 5 numeric columns
├── columns.pkl         # full 18-column schema used at training time
├── requirements.txt    # pinned dependencies
├── run.sh              # one-command launcher
├── README.md
└── index.html          # legacy static front-end (not used by Streamlit)
```

### Artifact shapes

| File | Content | Size |
|---|---|---|
| `LR_model.pkl` | `LogisticRegression(max_iter=2000, random_state=42)` | 15 input features |
| `scaler.pkl` | `StandardScaler` fitted on `Age, RestingBP, Cholesterol, MaxHR, Oldpeak` | 5 features |
| `columns.pkl` | list of the 18 raw/one-hot columns | 18 columns |

The model uses these 15 features, in this order:

```
Age, is_male, RestingBP, FastingBS, MaxHR, is_ExerciseAngina, Oldpeak,
ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
RestingECG_ST, ST_Slope_Down, ST_Slope_Flat, ST_Slope_Up
```

---

## Preprocessing (must match training)

The artifacts were produced in stages, so inference has to reproduce the
same order:

1. Build all **18** columns from the raw record (`columns.pkl`).
2. Standard-scale the **5** numeric columns (`Age, RestingBP, Cholesterol,
   MaxHR, Oldpeak`) using `scaler.pkl`.
3. Drop `Cholesterol`, `RestingECG_LVH`, `RestingECG_Normal`.
4. Select the **15** features the model expects and predict.

This is implemented in `build_features()` in `server.py`.

> Getting this wrong is silent: feeding raw (unscaled) values still returns a
> prediction, but the model's accuracy collapses (~41% instead of ~86%).

---

## Model performance

Evaluated on an 80/20 hold-out split (`random_state=42`, 184 test rows) with
the preprocessing above:

| Metric | Value |
|---|---|
| Accuracy | 0.8641 |
| Precision | 0.9100 |
| Recall | 0.8505 |
| F1-score | 0.8792 |

Confusion matrix (`0 = no disease`, `1 = disease`):

```
[[68  9]
 [16 91]]
```

---

## Notes & caveats

- **Cholesterol does not affect the output.** It is scaled and then dropped
  before prediction, so changing it never changes the risk. It appears in the
  UI only because it is part of the original 18-column schema. Retrain with
  Cholesterol kept if you want it to matter.
- The saved model is **Logistic Regression**, even though earlier drafts of
  `server.py` referenced a KNN artifact (`knn_heart_model.pkl`). That file does
  not exist.
- The scaler is stored **separately** from the model; both must be applied in
  the correct order.
- Single-split metrics are noisy — a cross-validated score would be a more
  reliable estimate.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt` |
| `FileNotFoundError: LR_model.pkl` | Run the app from this folder; keep the `.pkl` files next to `server.py`. |
| `X has N features, but StandardScaler is expecting 5` | The scaler must only receive the 5 numeric columns. |
| Port 8501 already in use | `python -m streamlit run server.py --server.port 8502` |
| Predictions look wrong / random | Check preprocessing order in `build_features()`. |
