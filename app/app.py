import streamlit as st
import pandas as pd
from collections import Counter
import re
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
    
st.title("📄 Flexible Electronics Patent Explorer")

    #uploaded_file = st.file_uploader("Upload Patent Excel File", type=["xlsx"])
    #if uploaded_file:
    #  df = pd.read_excel(uploaded_file) 
    #st.success("✅ Data Loaded!")
    #st.dataframe(df)  # Tam tabloyu göster
df = pd.read_excel("data/Flexible Electronics 2020-2025 Data.xlsx")
st.success("✅ Data Loaded from local file!")
agree = st.checkbox("✅ I agree to the Terms of Service")

if not agree:
    st.warning("Please accept the terms to proceed.")
    st.stop()

st.dataframe(df)



# ====== ÖZET ======
st.write("## 📋 Column Summary")
st.write(df.columns)

# ====== TOP APPLICANTS ======
if "Applicants" in df.columns:
    st.subheader("🏢 Top Applicants")
    top_applicants = df["Applicants"].value_counts().head(10)
    st.bar_chart(top_applicants)

# ====== KEYWORD ANALYSIS ======
if "Title" in df.columns:
    st.subheader("🧠 Frequent Terms in Patent Titles")
    all_titles = " ".join(df["Title"].dropna().astype(str)).lower()
    words = re.findall(r'\b[a-z]{2,}\b', all_titles)
    word_counts = Counter(words).most_common(20)
    word_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"]).set_index("Word")
    st.bar_chart(word_df)

# ====== YEAR ANALYSIS ======
if "Publication number" in df.columns:
    st.subheader("📅 Patent Publications by Year")
    if "Publication date" in df.columns:
        df["PubYear"] = pd.to_datetime(df["Publication date"], errors='coerce').dt.year
    else:
        df["PubYear"] = df["Publication number"].astype(str).str.extract(r'(\d{4})')
    year_counts = df["PubYear"].value_counts().sort_index()
    st.bar_chart(year_counts)

# ====== CLUSTERING + INTERACTIVE TOOLTIP ======
if "Title" in df.columns:
    st.subheader("📍 Patent Clustering by Title (Interactive)")

    titles = df["Title"].dropna().astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(titles)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X.toarray())

    cluster_df = pd.DataFrame({
        "x": components[:, 0],
        "y": components[:, 1],
        "Title": titles,
        "Cluster": labels
    })

    # Her cluster için en sık geçen kelimeyi bul
    cluster_keywords = {}

    for cluster_id in sorted(cluster_df["Cluster"].unique()):
        cluster_titles = cluster_df[cluster_df["Cluster"] == cluster_id]["Title"]
        words = re.findall(r'\b[a-z]{3,}\b', " ".join(cluster_titles).lower())
        common = Counter(words).most_common(1)
        if common:
            keyword = common[0][0]
            cluster_keywords[cluster_id] = f"{cluster_id}: {keyword}"
        else:
            cluster_keywords[cluster_id] = str(cluster_id)

    # Etiketli sütun ekle
    cluster_df["Cluster_Label"] = cluster_df["Cluster"].map(cluster_keywords)

    fig = px.scatter(
        cluster_df, x="x", y="y",
        color=cluster_df["Cluster_Label"],
        hover_data=["Title"],
        title="Patent Clusters (2D PCA View)",
        labels={"Cluster": "Cluster"}
    )
    st.plotly_chart(fig)

# ====== FILTER PANEL ======
st.sidebar.header("📌 Filter Options")

year_options = df["PubYear"].dropna().unique()
selected_year = st.sidebar.selectbox("Filter by Publication Year", sorted(year_options))

if "Applicants" in df.columns:
    applicants_list = df["Applicants"].dropna().unique()
    selected_applicants = st.sidebar.multiselect("Select Applicants", applicants_list)

if "Inventors" in df.columns:
    inventors_list = df["Inventors"].dropna().unique()
    selected_inventors = st.sidebar.multiselect("Select Inventors", inventors_list)

title_keyword = st.sidebar.text_input("Search Keyword in Title")
inventor_keyword = st.sidebar.text_input("Search Keyword in Inventors")

# ====== FILTER LOGIC ======
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["PubYear"] == selected_year]

if selected_applicants:
    filtered_df = filtered_df[filtered_df["Applicants"].isin(selected_applicants)]

if selected_inventors:
    filtered_df = filtered_df[filtered_df["Inventors"].isin(selected_inventors)]

if title_keyword:
    filtered_df = filtered_df[filtered_df["Title"].str.contains(title_keyword, case=False, na=False)]

if inventor_keyword:
    filtered_df = filtered_df[filtered_df["Inventors"].str.contains(inventor_keyword, case=False, na=False)]

# ====== FILTERED RESULTS & EXPORT ======
st.subheader("🔍 Filtered Results")
st.dataframe(filtered_df)

csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_patents.csv',
    mime='text/csv',
)
