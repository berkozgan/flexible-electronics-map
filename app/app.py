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

# ====== INFRINGEMENT RISK CHECKER ======
st.header("🚨 Infringement Risk Checker")

if "Title" in df.columns and "PubYear" in df.columns and "Inventors" in df.columns:
    titles = df["Title"].dropna().astype(str).tolist()
    selected_title = st.selectbox("Select a Patent to Check for Infringement Risk", titles)

    # Seçilen patentin detayları
    selected_row = df[df["Title"] == selected_title].iloc[0]
    selected_year = selected_row["PubYear"]
    selected_inventors = str(selected_row["Inventors"]).lower()
    selected_words = set(re.findall(r'\b[a-z]{3,}\b', selected_title.lower()))

    # Aynı yıldaki diğer patentleri filtrele
    same_year_df = df[(df["Title"] != selected_title) & (df["PubYear"] == selected_year)]

    potential_conflicts = []

    for idx, row in same_year_df.iterrows():
        comp_title = str(row["Title"])
        comp_inventors = str(row["Inventors"]).lower()

        # Ortak kelimeler
        comp_words = set(re.findall(r'\b[a-z]{3,}\b', comp_title.lower()))
        shared_keywords = selected_words.intersection(comp_words)

        # Ortak mucit kontrolü
        shared_inventors = any(inv in comp_inventors for inv in selected_inventors.split(", "))

        # Basit risk puanı: 0–1 arası
        risk_score = 0
        if shared_keywords:
            risk_score += len(shared_keywords) * 0.1
        if shared_inventors:
            risk_score += 0.4
        if comp_title.lower() in selected_title.lower() or selected_title.lower() in comp_title.lower():
            risk_score += 0.3

        if risk_score > 0.3:
            potential_conflicts.append((comp_title, risk_score, shared_keywords, shared_inventors))

    if potential_conflicts:
        st.markdown("### ⚠️ Possible Infringement Alerts")
        for title, score, keywords, inventors_match in sorted(potential_conflicts, key=lambda x: -x[1])[:5]:
            matched_row = df[df["Title"] == title].iloc[0]

            pub_number = matched_row.get("Publication number", "N/A")
            pub_date = matched_row.get("Publication date", "N/A")
            applicant = matched_row.get("Applicants", "N/A")
            inventors = matched_row.get("Inventors", "N/A")
            country_code = pub_number[:2] if isinstance(pub_number, str) else "N/A"

            st.markdown(f"""
            - **{title}**
             📊 Risk Score: `{score:.2f}` {'🔴 High' if score > 0.7 else '🟠 Medium' if score > 0.4 else '🟡 Low'}  
             🔑 Shared keywords: `{', '.join(keywords) if keywords else 'None'}`
             👥 Inventor overlap: `{"Yes" if inventors_match else "No"}`
             🌍 Country: `{country_code}`  
             📅 Publication date: `{pub_date}`  
             🏢 Applicant: `{applicant}`  
             🧠 Inventors: `{inventors}`
            """)

    else:
        st.success("✅ No significant infringement risks found for this patent.")

