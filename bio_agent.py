import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from lifelines import KaplanMeierFitter

# Optional AI
USE_AI = True
try:
    import ollama
except:
    USE_AI = False

st.set_page_config(page_title="Bioinformatics Agent", layout="wide")
st.title("🧬 Bioinformatics Agent")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_files = st.file_uploader(
    "Upload dataset(s)",
    type=["csv", "tsv", "xlsx"],
    accept_multiple_files=True
)

def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)

# -------------------------
# MAIN DATA
# -------------------------
if uploaded_files:

    dfs = []
    for file in uploaded_files:
        df_temp = load_file(file)
        df_temp.columns = df_temp.columns.str.strip()
        df_temp["source"] = file.name
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    st.success("✅ Data Loaded")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # -------------------------
    # MSI SCORING
    # -------------------------
    st.subheader("🧬 MSI Scoring")

    if st.button("Calculate MSI Score"):
        cols = [c for c in df.columns if "repeat_change_status" in c]

        if cols:
            df["MSI_score"] = df[cols].apply(
                lambda x: sum(x.astype(str).str.contains("Expanded|Contracted")),
                axis=1
            )

            df["MSI_status"] = pd.cut(
                df["MSI_score"],
                bins=[-1, 1, 3, 100],
                labels=["MSS", "MSI-Low", "MSI-High"]
            )

            st.dataframe(df[["MSI_score", "MSI_status"]].head())

            st.download_button("Download MSI Results",
                               df.to_csv(index=False),
                               "msi_results.csv")
        else:
            st.warning("No repeat_change_status columns found")

    # -------------------------
    # STATISTICS
    # -------------------------
    st.subheader("📊 Statistical Analysis")

    test_type = st.selectbox("Test", [
        "t-test", "Wilcoxon", "ANOVA",
        "Chi-square", "Pearson", "Spearman"
    ])

    x_col = st.selectbox("X", df.columns)
    y_col = st.selectbox("Y", df.columns)

    if st.button("Run Test"):

        try:
            if test_type in ["t-test", "Wilcoxon", "ANOVA", "Pearson", "Spearman"]:
                if x_col not in numeric_cols or y_col not in numeric_cols:
                    st.error("Numeric columns required")
                    st.stop()

            if test_type == "Chi-square":
                if x_col not in categorical_cols or y_col not in categorical_cols:
                    st.error("Categorical columns required")
                    st.stop()

            if test_type == "t-test":
                stat, p = stats.ttest_ind(df[x_col].dropna(), df[y_col].dropna())

            elif test_type == "Wilcoxon":
                stat, p = stats.ranksums(df[x_col], df[y_col])

            elif test_type == "ANOVA":
                stat, p = stats.f_oneway(df[x_col], df[y_col])

            elif test_type == "Chi-square":
                table = pd.crosstab(df[x_col], df[y_col])
                stat, p, _, _ = stats.chi2_contingency(table)

            elif test_type == "Pearson":
                stat, p = stats.pearsonr(df[x_col], df[y_col])

            elif test_type == "Spearman":
                stat, p = stats.spearmanr(df[x_col], df[y_col])

            result = pd.DataFrame({
                "Test": [test_type],
                "Statistic": [stat],
                "p-value": [p]
            })

            st.dataframe(result)
            st.download_button("Download Stats", result.to_csv(index=False), "stats.csv")

        except Exception as e:
            st.error(e)

    # -------------------------
    # DGE
    # -------------------------
    st.subheader("🧬 Differential Gene Expression")

    dge_file = st.file_uploader("Upload Expression Matrix", type=["csv", "tsv"], key="dge")

    if dge_file:

        if dge_file.name.endswith(".csv"):
            expr = pd.read_csv(dge_file, index_col=0)
        else:
            expr = pd.read_csv(dge_file, sep="\t", index_col=0)

        st.dataframe(expr.head())

        group1 = st.multiselect("Group 1", expr.columns)
        group2 = st.multiselect("Group 2", expr.columns)

        method = st.selectbox("Test", ["t-test", "Wilcoxon"])

        if st.button("Run DGE"):

            results = []

            for gene in expr.index:
                g1 = expr.loc[gene, group1].dropna()
                g2 = expr.loc[gene, group2].dropna()

                if len(g1) < 2 or len(g2) < 2:
                    continue

                fc = np.log2(g1.mean() + 1) - np.log2(g2.mean() + 1)

                if method == "t-test":
                    stat, p = stats.ttest_ind(g1, g2)
                else:
                    stat, p = stats.ranksums(g1, g2)

                results.append([gene, fc, p])

            dge_df = pd.DataFrame(results, columns=["Gene", "log2FC", "p-value"])

            from statsmodels.stats.multitest import multipletests
            dge_df["adj_p"] = multipletests(dge_df["p-value"], method="fdr_bh")[1]

            dge_df = dge_df.sort_values("adj_p")

            st.dataframe(dge_df.head(50))

            st.download_button("Download DGE", dge_df.to_csv(index=False), "dge.csv")

            # Volcano plot
            dge_df["-log10(p)"] = -np.log10(dge_df["p-value"])

            fig = px.scatter(dge_df, x="log2FC", y="-log10(p)", hover_data=["Gene"])
            st.plotly_chart(fig)

            # -------------------------
            # ENRICHMENT
            # -------------------------
            st.subheader("🧬 Functional Enrichment")

            p_cut = st.slider("Adj p cutoff", 0.001, 0.1, 0.05)
            fc_cut = st.slider("log2FC cutoff", 0.5, 3.0, 1.0)

            filtered = dge_df[
                (dge_df["adj_p"] < p_cut) &
                (abs(dge_df["log2FC"]) > fc_cut)
            ]

            genes = filtered["Gene"].tolist()

            if st.button("Run Enrichment"):

                import gseapy as gp

                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=["GO_Biological_Process_2021", "KEGG_2021_Human"],
                    organism="Human",
                    outdir=None
                )

                res = enr.results.sort_values("Adjusted P-value")

                st.dataframe(res.head(20))

                fig = px.bar(res.head(10), x="Combined Score", y="Term", orientation="h")
                st.plotly_chart(fig)

    # -------------------------
    # SURVIVAL
    # -------------------------
    st.subheader("⏳ Survival Analysis")

    time_col = st.selectbox("Time", ["None"] + list(df.columns))
    event_col = st.selectbox("Event", ["None"] + list(df.columns))
    group_col = st.selectbox("Group", ["None"] + list(df.columns))

    if st.button("Run Survival"):

        if time_col != "None" and event_col != "None" and group_col != "None":

            kmf = KaplanMeierFitter()

            for group in df[group_col].dropna().unique():
                subset = df[df[group_col] == group]
                kmf.fit(subset[time_col], subset[event_col], label=str(group))
                kmf.plot()

            st.pyplot()

    # -------------------------
    # VISUALIZATION
    # -------------------------
    st.subheader("📈 Visualization")

    plot_type = st.selectbox("Plot", ["Scatter", "Box", "Histogram", "Bar"])
    x_axis = st.selectbox("X-axis", df.columns)
    y_axis = st.selectbox("Y-axis", ["None"] + list(df.columns))
    title = st.text_input("Title", "Plot")

    if st.button("Generate Plot"):

        if plot_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=None if y_axis == "None" else y_axis)

        elif plot_type == "Box":
            fig = px.box(df, x=x_axis, y=None if y_axis == "None" else y_axis)

        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_axis)

        elif plot_type == "Bar":
            fig = px.bar(df, x=x_axis, y=None if y_axis == "None" else y_axis)

        fig.update_layout(title=title)
        st.plotly_chart(fig)

    # -------------------------
    # AI
    # -------------------------
    st.subheader("💬 AI Assistant")

    query = st.text_input("Ask about your data")

    if query and USE_AI:
        sample = df.head(20).to_string()

        response = ollama.chat(
            model="phi",
            messages=[{
                "role": "user",
                "content": f"Dataset:\n{sample}\n\nQuestion:\n{query}"
            }]
        )

        st.write(response["message"]["content"])

else:
    st.info("Upload dataset(s) to begin")