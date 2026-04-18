import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Performance Analysis", layout="wide")
st.title("Student Performance Analysis using Unsupervised Learning")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    data = load_data("StudentsPerformance.csv")
except FileNotFoundError:
    st.error("Could not find StudentsPerformance.csv in the app directory.")
    st.stop()

numeric_columns = ["math score", "reading score", "writing score"]
category_columns = ["gender", "parental level of education", "lunch", "test preparation course"]

if "k" not in st.session_state:
    st.session_state.k = 3

@st.cache_data
def compute_clustering(df, k):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_columns])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    result = df.copy()
    result["cluster"] = labels
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(scaled)
    result["pca_1"] = pca_features[:, 0]
    result["pca_2"] = pca_features[:, 1]
    return result, scaler, kmeans, pca

page = st.sidebar.radio("Navigation", ["Overview", "EDA", "Clustering", "Insights"])

missing_counts = data.isnull().sum()
clean_data = data.dropna(subset=numeric_columns).copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(clean_data[numeric_columns])
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(scaled_features)

if page == "Overview":
    st.header("Overview Dashboard")

    total_students = len(data)
    avg_math = data["math score"].mean()
    avg_reading = data["reading score"].mean()
    avg_writing = data["writing score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total students", total_students)
    col2.metric("Average math score", f"{avg_math:.2f}")
    col3.metric("Average reading score", f"{avg_reading:.2f}")
    col4.metric("Average writing score", f"{avg_writing:.2f}")

    st.subheader("Score distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(data["math score"], kde=True, color="#4C72B0", ax=axes[0])
    axes[0].set_title("Math Score Distribution")
    sns.histplot(data["reading score"], kde=True, color="#55A868", ax=axes[1])
    axes[1].set_title("Reading Score Distribution")
    sns.histplot(data["writing score"], kde=True, color="#C44E52", ax=axes[2])
    axes[2].set_title("Writing Score Distribution")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_counts = data["gender"].value_counts()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="pastel", ax=ax)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Parental education levels")
        parent_counts = data["parental level of education"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=parent_counts.values, y=parent_counts.index, palette="muted", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Parental education")
        st.pyplot(fig)

elif page == "EDA":
    st.header("Data Exploration")
    st.subheader("Dataset preview")
    st.dataframe(data.head(15))

    st.subheader("Missing values")
    st.write(missing_counts.to_frame(name="Missing values"))

    st.subheader("Basic statistics")
    st.write(data[numeric_columns].describe().round(2))

    st.subheader("Scatter plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=data, x="math score", y="reading score", hue="gender", palette="Set2", ax=axes[0])
    axes[0].set_title("Math vs Reading")
    sns.scatterplot(data=data, x="writing score", y="reading score", hue="gender", palette="Set2", ax=axes[1])
    axes[1].set_title("Writing vs Reading")
    st.pyplot(fig)

    st.subheader("Boxplots for score comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data[numeric_columns], palette="Set3", ax=ax)
    ax.set_title("Score Boxplots")
    ax.set_ylabel("Score")
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    corr = data[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation between numeric scores")
    st.pyplot(fig)

elif page == "Clustering":
    st.header("Clustering (K-Means)")
    st.write("Apply K-Means clustering to math, reading, and writing scores.")

    k = st.slider("Number of clusters", 2, 5, st.session_state.k)
    st.session_state.k = k
    clean_data, scaler, kmeans, pca = compute_clustering(clean_data, k)

    st.subheader("PCA scatter plot of clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=k)
    for cluster_id in range(k):
        cluster_points = clean_data[clean_data["cluster"] == cluster_id]
        ax.scatter(
            cluster_points["pca_1"],
            cluster_points["pca_2"],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            s=40,
            color=palette[cluster_id],
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means clusters in PCA space")
    ax.legend(title="Cluster")
    st.pyplot(fig)

    st.subheader("Cluster centers and sizes")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=numeric_columns)
    center_df.index.name = "cluster"
    st.write(center_df.round(2))

    cluster_counts = clean_data["cluster"].value_counts().sort_index()
    cluster_count_df = pd.DataFrame({"Student count": cluster_counts})
    st.write(cluster_count_df)

    st.subheader("Processed dataset")
    st.dataframe(clean_data[["gender", "math score", "reading score", "writing score", "cluster"]].head(20))

    csv = clean_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed dataset",
        data=csv,
        file_name="student_performance_clustered.csv",
        mime="text/csv",
    )

elif page == "Insights":
    st.header("Cluster Insights")

    if "cluster" not in clean_data.columns:
        clean_data, scaler, kmeans, pca = compute_clustering(clean_data, st.session_state.k)

    avg_scores = (
        clean_data.groupby("cluster")[numeric_columns]
        .mean()
        .round(2)
    )
    avg_scores["student_count"] = clean_data["cluster"].value_counts().sort_index().values
    avg_scores["average_total_score"] = avg_scores[numeric_columns].mean(axis=1).round(2)

    n_clusters = clean_data["cluster"].nunique()
    score_ranks = avg_scores["average_total_score"].rank(method="first", ascending=False)
    avg_scores["performance_segment"] = score_ranks.map(
        lambda x: "High-performing" if x <= n_clusters / 3 else ("Low-performing" if x > 2 * n_clusters / 3 else "Average")
    )

    st.subheader("Average scores per cluster")
    st.write(avg_scores.drop(columns=["average_total_score"]))

    st.subheader("Cluster performance summary")
    for cluster_id, row in avg_scores.sort_values("average_total_score", ascending=False).iterrows():
        st.write(
            f"Cluster {cluster_id}: {row['performance_segment']} students with an average math score of {row['math score']:.1f}, "
            f"reading score of {row['reading score']:.1f}, and writing score of {row['writing score']:.1f}."
        )

    st.subheader("Feature contribution from PCA")
    pca_loadings = pd.Series(np.abs(pca.components_[0]), index=numeric_columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=pca_loadings.values, y=pca_loadings.index, palette="viridis", ax=ax)
    ax.set_xlabel("Absolute loading")
    ax.set_ylabel("Feature")
    ax.set_title("Feature contribution to first PCA component")
    st.pyplot(fig)

    st.subheader("Behavior differences by cluster")
    merged = data.loc[clean_data.index].copy()
    merged["cluster"] = clean_data["cluster"].values
    behavior_summary = (
        merged.groupby("cluster")[category_columns]
        .agg(lambda x: x.value_counts(normalize=True).iloc[0])
        .round(2)
    )
    st.write(behavior_summary)
