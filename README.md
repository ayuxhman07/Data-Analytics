# Food Ordering Decisions — Project Plan

## Top-Level Overview

**Goal:** Build a full Python project that uses `online_food_delivery_dataset.csv` (388 rows, 13 columns) to:
1. Explore and visualise the data (EDA)
2. Train a **Random Forest Classifier** that predicts whether a customer will order food (`Output`: Yes / No)
3. Expose all of this through a **Streamlit** multi-page web dashboard

**Stack:**
- Frontend / UI: Streamlit (multi-page app)
- Backend / ML: scikit-learn, pandas, numpy
- Visualisation: matplotlib, seaborn, plotly
- Serialisation: joblib (save/load trained model)

**Dataset columns:**
`Age`, `Gender`, `Marital Status`, `Occupation`, `Monthly Income`, `Educational Qualifications`, `Family size`, `Customer Type`, `latitude`, `longitude`, `Pin code`, `Output` (target), `Feedback`(https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset)

**Project folder layout:**
```
food_ordering_project/
├── data/
│   └── online_food_delivery_dataset.csv   (copy of source file)
├── models/
│   └── rf_model.pkl                        (trained model, auto-generated)
│   └── preprocessor.pkl                    (column transformer, auto-generated)
├── backend/
│   └── preprocess.py                       (data cleaning + feature engineering)
│   └── train.py                            (model training + serialisation)
│   └── predict.py                          (single-record prediction helper)
├── pages/
│   └── 1_EDA.py                            (Streamlit EDA page)
│   └── 2_Prediction.py                     (Streamlit Prediction page)
│   └── 3_Model_Performance.py              (Streamlit Model Performance page)
├── app.py                                  (Streamlit entry point / Home page)
└── requirements.txt
```

---

## Sub-Task 1 — Project Scaffold & Data Setup

**Intent:** Create the folder structure, copy the dataset into `data/`, and write `requirements.txt`.

**Expected Outcomes:**
- All project directories exist
- `data/online_food_delivery_dataset.csv` is present
- `requirements.txt` lists all dependencies with pinned-compatible versions

**Todo List:**
1. Create directories: `food_ordering_project/data`, `food_ordering_project/models`, `food_ordering_project/backend`, `food_ordering_project/pages`
2. Copy `online_food_delivery_dataset.csv` into `food_ordering_project/data/`
3. Write `requirements.txt`

**Relevant Context:**
- Source file: `online_food_delivery_dataset.csv` (workspace root)
- Dependencies needed: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `plotly`

**Status:** [x] done

---

## Sub-Task 2 — Backend: Data Preprocessing (`preprocess.py`)

**Intent:** Write a reusable preprocessing module that cleans the raw CSV, encodes categoricals, and returns train/test splits ready for the model.

**Expected Outcomes:**
- `backend/preprocess.py` exports a `load_and_preprocess()` function
- The function drops `Unnamed: 13`, encodes all categorical features with a `ColumnTransformer` (OrdinalEncoder / OneHotEncoder)
- Returns `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`, `raw_df`
- The fitted `ColumnTransformer` is serialised to `models/preprocessor.pkl`

**Todo List:**
1. Load CSV, drop `Unnamed: 13` column
2. Define categorical columns: `Gender`, `Marital Status`, `Occupation`, `Monthly Income`, `Educational Qualifications`, `Customer Type`, `Feedback`
3. Define numeric columns: `Age`, `Family size`, `latitude`, `longitude`, `Pin code`
4. Build a `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` for categoricals, `passthrough` for numerics
5. Encode target `Output` (Yes=1, No=0)
6. Split 80/20 stratified on target
7. Fit transformer on train set, transform both sets
8. Save fitted transformer to `models/preprocessor.pkl` with joblib
9. Return all required objects

**Relevant Context:**
- `Monthly Income` has string categories: `No Income`, `Below Rs.10000`, `10001 to 25000`, `25001 to 50000`, `More than 50000`
- `Feedback` has values `Positive`, `Negative ` (trailing space — strip before encoding)
- `Output` column is the binary target (Yes/No)

**Status:** [x] done

---

## Sub-Task 3 — Backend: Model Training (`train.py`)

**Intent:** Train a Random Forest Classifier, evaluate it, and serialise the trained model to disk.

**Expected Outcomes:**
- `backend/train.py` can be run as a standalone script (`python backend/train.py`)
- Prints train accuracy, test accuracy, classification report
- Saves `models/rf_model.pkl`

**Todo List:**
1. Import `load_and_preprocess` from `preprocess.py`
2. Instantiate `RandomForestClassifier(n_estimators=100, random_state=42)`
3. Fit on `X_train`, `y_train`
4. Evaluate on both splits; print metrics
5. Save model to `models/rf_model.pkl` with joblib

**Relevant Context:**
- Model and preprocessor must both be saved to `models/` so Streamlit pages can load them at runtime
- `train.py` must be idempotent (re-running overwrites the pkl files)

**Status:** [ ] pending

---

## Sub-Task 4 — Backend: Prediction Helper (`predict.py`)

**Intent:** Provide a clean `predict_single(input_dict)` function that the Streamlit Prediction page calls to get a Yes/No verdict plus class probabilities.

**Expected Outcomes:**
- `backend/predict.py` exports `predict_single(input_dict) -> dict`
- Returns `{"decision": "Yes"/"No", "probability_yes": float, "probability_no": float}`
- Loads model and preprocessor from `models/` on first call (cached)

**Todo List:**
1. Load `rf_model.pkl` and `preprocessor.pkl` at module level
2. Accept a dict of raw feature values matching the original column names
3. Convert to a single-row DataFrame
4. Apply the fitted preprocessor transform
5. Run model `predict` and `predict_proba`
6. Return result dict

**Relevant Context:**
- Input dict keys must match training column names exactly (minus `Output`)
- Preprocessor handles unknown categories gracefully (`handle_unknown='ignore'`)

**Status:** [ ] pending

---

## Sub-Task 5 — Streamlit Home Page (`app.py`)

**Intent:** Create the Streamlit entry-point that shows a project overview and navigation guidance.

**Expected Outcomes:**
- Running `streamlit run app.py` launches the app
- Home page displays project title, dataset summary stats, and links to the three pages

**Todo List:**
1. Write `app.py` with `st.title`, brief description
2. Load raw CSV and show `st.dataframe` preview (first 10 rows)
3. Show key KPIs: total records, % ordered (Yes), % positive feedback
4. Add navigation hint (Streamlit sidebar is auto-generated from `pages/`)

**Relevant Context:**
- Streamlit multi-page apps: any `.py` file placed in `pages/` is auto-registered as a sidebar page

**Status:** [ ] pending

---

## Sub-Task 6 — Streamlit EDA Page (`pages/1_EDA.py`)

**Intent:** Build the exploratory data analysis page with interactive charts covering all major columns.

**Expected Outcomes:**
- Page renders without errors
- Contains at least 6 visualisations covering demographics, income, customer type, feedback, and geographic spread

**Charts to include:**
1. Ordering decision distribution (pie chart — Yes vs No)
2. Age distribution by ordering decision (histogram, Plotly)
3. Gender × Ordering decision (grouped bar, Plotly)
4. Monthly Income × Ordering decision (stacked bar, Plotly)
5. Customer Type distribution (bar, Plotly)
6. Feedback distribution (bar, Plotly)
7. Correlation heatmap of numeric columns (seaborn)
8. Location scatter map (latitude/longitude, Plotly scatter_mapbox)

**Todo List:**
1. Load raw CSV
2. Add sidebar filters: Gender, Occupation, Marital Status (multiselect)
3. Apply filters to dataframe reactively
4. Render each chart using `st.plotly_chart` / `st.pyplot`
5. Add `st.metric` cards for filtered summary stats

**Relevant Context:**
- Use `plotly.express` for interactive charts
- Strip whitespace from `Feedback` column before plotting

**Status:** [ ] pending

---

## Sub-Task 7 — Streamlit Prediction Page (`pages/2_Prediction.py`)

**Intent:** Build a form where a user fills in all feature values and gets a real-time Yes/No ordering decision prediction.

**Expected Outcomes:**
- All input widgets render with sensible defaults
- Clicking "Predict" calls `predict_single()` and displays decision + probability gauge
- Displays a confidence bar (green for Yes, red for No)

**Form fields (widgets):**
| Column | Widget |
|---|---|
| Age | `st.slider(18, 60)` |
| Gender | `st.selectbox` |
| Marital Status | `st.selectbox` |
| Occupation | `st.selectbox` |
| Monthly Income | `st.selectbox` |
| Educational Qualifications | `st.selectbox` |
| Family size | `st.slider(1, 10)` |
| Customer Type | `st.selectbox` |
| latitude | `st.number_input` |
| longitude | `st.number_input` |
| Pin code | `st.number_input` |
| Feedback | `st.selectbox` |

**Todo List:**
1. Import `predict_single` from `backend/predict.py`
2. Render two-column layout for form fields
3. On button press, build input dict and call `predict_single`
4. Display result using `st.success` / `st.error` + `st.metric` for probabilities
5. Render a Plotly gauge chart showing confidence

**Status:** [ ] pending

---

## Sub-Task 8 — Streamlit Model Performance Page (`pages/3_Model_Performance.py`)

**Intent:** Show the model's evaluation metrics, confusion matrix, and feature importance so users can trust the predictions.

**Expected Outcomes:**
- Displays accuracy, precision, recall, F1 (Yes/No classes)
- Renders a colour-coded confusion matrix heatmap
- Renders a horizontal bar chart of top-15 feature importances

**Todo List:**
1. Load model and preprocessor from `models/`
2. Call `load_and_preprocess()` to get test set
3. Compute `accuracy_score`, `classification_report`, `confusion_matrix`
4. Render metrics as `st.metric` cards
5. Plot confusion matrix with `seaborn.heatmap` inside `st.pyplot`
6. Extract `feature_importances_` from Random Forest, map back to feature names, plot top 15 with Plotly horizontal bar

**Status:** [ ] pending

---

## Dependencies Summary

```
streamlit>=1.33
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
joblib>=1.3
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20
```

---

## Execution Order

1. Sub-Task 1 (Scaffold)
2. Sub-Task 2 (Preprocess)
3. Sub-Task 3 (Train)
4. Sub-Task 4 (Predict helper)
5. Sub-Task 5 (Home page)
6. Sub-Task 6 (EDA page)
7. Sub-Task 7 (Prediction page)
8. Sub-Task 8 (Model Performance page)

After all sub-tasks are done, run `python backend/train.py` once to generate the `.pkl` files, then `streamlit run food_ordering_project/app.py` to launch the app.
