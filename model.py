import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.markdown("## 🛍️ **Customer Segmentation Dashboard**")

# LOAD YOUR CSV [file:12]
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# SIDEBAR
st.sidebar.title("⚙️ Controls")
k = st.sidebar.slider("Clusters", 2, 8, 5)

# CLUSTERING
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters
sil_score = silhouette_score(X, clusters)

# METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Customers", len(df))
col2.metric("Clusters", k)
col3.metric("Silhouette", f"{sil_score:.3f}")

# DATA
st.subheader("📊 Data")
st.dataframe(df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(10))

# INCOME vs SPENDING CHART (Native Streamlit)
st.subheader("📈 Income vs Spending")
chart_data = df.groupby(['Annual Income (k$)', 'Cluster']).size().unstack(fill_value=0)
st.bar_chart(chart_data)

# CLUSTER SUMMARY
st.subheader("💎 Segments")
cluster_stats = df.groupby('Cluster')[features].mean().round(1)
st.dataframe(cluster_stats.T)

# INSIGHTS
st.subheader("🎯 Insights")
for i in range(k):
    cluster_df = df[df['Cluster'] == i]
    income = cluster_df['Annual Income (k$)'].mean()
    spending = cluster_df['Spending Score (1-100)'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric(f"Size", len(cluster_df))
    col2.metric("Income", f"${income:.0f}k")
    
    st.write(f"**Age**: {cluster_df['Age'].mean():.0f} | **Spending**: {spending:.0f}")
    
    if income > 60 and spending > 60:
        st.success("💎 **VIP Customers**")
    elif income > 60:
        st.info("🎯 **High Income**")
    elif spending > 60:
        st.warning("🚀 **High Spenders**")
    else:
        st.markdown("💰 **Budget Buyers**")

st.success("✅ **Dashboard Complete!**")
