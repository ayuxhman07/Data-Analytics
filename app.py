"""
app.py  —  Food Ordering Decisions Dashboard
Home page: dataset preview + KPI cards
Run with:  streamlit run app.py
"""

import pathlib
import sys
import pandas as pd
import streamlit as st

# Make backend importable from any working directory
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

DATA_PATH = ROOT / "data" / "online_food_delivery_dataset.csv"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Ordering Decisions",
    page_icon="🍔",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    str_cols = df.select_dtypes(include="str").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df

df = load_raw()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🍔 Food Ordering Decisions")
st.markdown(
    """
    A machine-learning dashboard built on the **Online Food Delivery Dataset** (Bengaluru, India).
    Use the **sidebar** to navigate between pages.
    """
)
st.divider()

# ── KPI cards ──────────────────────────────────────────────────────────────────
total      = len(df)
pct_yes    = (df["Output"] == "Yes").mean() * 100
pct_pos    = (df["Feedback"] == "Positive").mean() * 100
avg_age    = df["Age"].mean()
avg_family = df["Family size"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records",        f"{total:,}")
c2.metric("Ordered Food (Yes)",   f"{pct_yes:.1f}%")
c3.metric("Positive Feedback",    f"{pct_pos:.1f}%")
c4.metric("Average Age",          f"{avg_age:.1f} yrs")
c5.metric("Avg Family Size",      f"{avg_family:.1f}")

st.divider()

# ── Dataset preview ────────────────────────────────────────────────────────────
st.subheader("Dataset Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

# ── Column summary ─────────────────────────────────────────────────────────────
st.subheader("Column Summary")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Numeric columns**")
    st.dataframe(df[["Age", "Family size", "latitude", "longitude", "Pin code"]].describe().T,
                 use_container_width=True)

with col_b:
    st.markdown("**Categorical value counts**")
    for col in ["Gender", "Marital Status", "Occupation", "Monthly Income",
                "Customer Type", "Output", "Feedback"]:
        st.markdown(f"*{col}*")
        st.dataframe(
            df[col].value_counts().rename_axis(col).reset_index(name="Count"),
            use_container_width=True,
            hide_index=True,
        )

# ── Navigation hint ────────────────────────────────────────────────────────────
st.divider()
st.info("Use the **sidebar** to navigate to: **EDA**, **Prediction**, or **Model Performance**.")
