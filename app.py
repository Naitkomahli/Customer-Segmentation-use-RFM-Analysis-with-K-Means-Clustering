# app.py — Customer Segmentation RFM + K-Means (Streamlit)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")


# Cache: pipeline jalan sekali, hasilnya dipakai ulang tiap interaksi
@st.cache_data
def load_and_segment():
    # 1. Load
    df = pd.read_csv('dataset/online_retail_II.csv')

    # 2. Cleaning (sama persis dengan notebook)
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # 3. RFM
    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    r = (snapshot - df.groupby('Customer ID')['InvoiceDate'].max()).dt.days
    f = df.groupby('Customer ID')['Invoice'].nunique()
    m = df.groupby('Customer ID')['TotalPrice'].sum()
    rfm = pd.concat([r, f, m], axis=1).reset_index()
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

    # 4. Preprocessing (log + scale)
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log)
    rfm_scaled = StandardScaler().fit_transform(rfm_log)

    # 5. K-Means k=4 (random_state sama -> hasil identik notebook)
    km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)
    seg_map = {0: 'At Risk', 1: 'Loyal Customers',
               2: 'Lost / Hibernating', 3: 'Champions'}
    rfm['Segment'] = rfm['Cluster'].map(seg_map)
    return rfm


rfm = load_and_segment()

st.title("Customer Segmentation — RFM + K-Means")
st.write(f"Total customer tersegmentasi: **{len(rfm):,}**")

# Metric cards per segmen
cols = st.columns(4)
for col, seg in zip(cols, ['Champions', 'Loyal Customers', 'At Risk', 'Lost / Hibernating']):
    col.metric(seg, f"{(rfm['Segment'] == seg).sum():,}")

# Tabel profil
st.subheader("Profil Rata-rata per Segmen")
prof = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
prof['Jumlah'] = rfm['Segment'].value_counts()
st.dataframe(prof)

# Dua chart berdampingan: kiri = bar jumlah, kanan = scatter segmentasi
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Jumlah Customer per Segmen")
    fig, ax = plt.subplots(figsize=(6, 4))
    rfm['Segment'].value_counts().plot(kind='bar', ax=ax)
    plt.xticks(rotation=20)
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.subheader("Peta Segmentasi (Recency vs Monetary)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for seg in rfm['Segment'].unique():
        sub = rfm[rfm['Segment'] == seg]
        ax2.scatter(sub['Recency'], sub['Monetary'], label=seg, s=12, alpha=0.6)
    ax2.set_xlabel('Recency')
    ax2.set_ylabel('Monetary')
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

# Fitur interaktif: cek 1 customer
st.subheader("Cek Segmen Customer")
cid = st.number_input("Masukkan Customer ID", min_value=0.0, step=1.0)
row = rfm[rfm['Customer ID'] == cid]
if len(row):
    row = row.iloc[0]
    st.success(f"Segmen: **{row['Segment']}**  |  R={row['Recency']:.0f}, "
               f"F={row['Frequency']:.0f}, M={row['Monetary']:.0f}")
else:
    st.warning("Customer ID tidak ditemukan.")
