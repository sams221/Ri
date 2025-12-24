import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
QUERIES_DIR = "Queries"
RANKINGS_DIR = "rankings"
PLOTS_DIR = "plots"
METRICS_DIR = "metrics"

st.set_page_config(page_title="IR Evaluation UI", layout="wide")
st.title("IR Evaluation – MEDLINE")
st.caption("USTHB – M2 SII | LAB 5")

# ===================== MODELS =====================
MODEL_CONFIG = {
    "MLE": {
        "ranking": "ranking_mle.csv",
        "metrics_per_query": "metrics_per_query_MLE.csv",
        "metrics_overall": "metrics_overall_summary_MLE.csv"
    },
    "Dirichlet": {
        "ranking": "ranking_dirichlet.csv",
        "metrics_per_query": "metrics_per_query_Dirichlet Smoothing.csv",
        "metrics_overall": "metrics_overall_summary_Dirichlet Smoothing.csv"
    },
    "Jelinek-Mercer": {
        "ranking": "ranking_jelinek-mercer.csv",
        "metrics_per_query": "metrics_per_query_Jelinek-Mercer Smoothing.csv",
        "metrics_overall": "metrics_overall_summary_Jelinek-Mercer Smoothing.csv"
    },
    "Laplace": {
        "ranking": "ranking_laplace.csv",
        "metrics_per_query": "metrics_per_query_Laplace Smoothing.csv",
        "metrics_overall": "metrics_overall_summary_Laplace Smoothing.csv"
    }
}

# ===================== SIDEBAR =====================
section = st.sidebar.radio("Navigation", ["Ranking", "Plots", "Metrics", "Comparison"])
model = st.sidebar.selectbox("Model", list(MODEL_CONFIG.keys()))

query_files = sorted(f for f in os.listdir(QUERIES_DIR) if f.startswith("query_"))
query_ids = [f"Q{int(f.split('_')[1].split('.')[0])}" for f in query_files]
query = st.sidebar.selectbox("Query", query_ids)
query_number = int(query.replace("Q", ""))

# ===================== RANKING =====================
if section == "Ranking":
    st.subheader("Ranked Results")
    df = pd.read_csv(os.path.join(RANKINGS_DIR, MODEL_CONFIG[model]["ranking"]))
    df = df[df["Query"] == f"Query {query_number}"]
    st.dataframe(df, use_container_width=True)

# ===================== PLOTS =====================
elif section == "Plots":
    st.subheader("Precision–Recall Plots")
    for img in sorted(os.listdir(PLOTS_DIR)):
        if model.lower() in img.lower():
            st.image(os.path.join(PLOTS_DIR, img), caption=img)

# ===================== METRICS =====================
elif section == "Metrics":
    mode = st.radio("Metrics Mode", ["Overall", "Per Query"])

    if mode == "Overall":
        df = pd.read_csv(os.path.join(METRICS_DIR, MODEL_CONFIG[model]["metrics_overall"]))
        st.dataframe(df, use_container_width=True)

    else:
        df = pd.read_csv(os.path.join(METRICS_DIR, MODEL_CONFIG[model]["metrics_per_query"]))
        row = df[df["Query_ID"] == f"Query {query_number}"]
        st.dataframe(row, use_container_width=True)

# ===================== COMPARISON =====================
elif section == "Comparison":
    comp_mode = st.radio("Comparison Mode", ["Per Query", "Average"])

    data = []

    for model_name, files in MODEL_CONFIG.items():
        df = pd.read_csv(os.path.join(METRICS_DIR, files["metrics_per_query"]))

        if comp_mode == "Per Query":
            row = df[df["Query_ID"] == f"Query {query_number}"]
        else:
            row = df.mean(numeric_only=True)

        if not row.empty:
            record = row.iloc[0].to_dict() if comp_mode == "Per Query" else row.to_dict()
            record["Model"] = model_name
            data.append(record)

    comp_df = pd.DataFrame(data)
    comp_df = comp_df.sort_values("nDCG@20", ascending=False)

    st.subheader("Model Comparison (Ranked by nDCG@20)")
    st.dataframe(comp_df, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(comp_df["Model"], comp_df["nDCG@20"])
    ax.set_ylabel("nDCG@20")
    ax.set_title("Model Ranking by nDCG@20")
    st.pyplot(fig)
