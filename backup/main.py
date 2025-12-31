# streamlit_dashboard.py
# Streamlit dashboard reproducing layout from provided template + user's processing
# Run with: streamlit run streamlit_dashboard.py

import streamlit as st
import pandas as pd
from dateutil import parser
import plotly.express as px
from io import BytesIO
from datetime import datetime
import streamlit.components.v1 as components
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import bcrypt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm



# =============================
# USER DATABASE (ROLE-BASED)
# =============================
USERS = {
    "admin": {
        "password": b"$2b$12$/FeSxkEROMGfYzyqrhQ0wOufha25HsKZI29AJjI.i.KkklrSEX9IO",  # hashed 'adminpass'
        "role": "admin"
    }
}

def check_password(username, password):
    if username not in USERS:
        return False
    return bcrypt.checkpw(
        password.encode(),
        USERS[username]["password"]
    )


def login():
    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_password(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = USERS[username]["role"]
            st.success("Login berhasil")
            st.rerun()
        else:
            st.error("Username atau password salah")


st.set_page_config(page_title="Dashboard Penjualan", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# Role-based access
if st.session_state.get("role") != "admin":
    st.error("❌ Anda tidak memiliki akses ke dashboard ini.")
    st.stop()



# ----------------- Google Sheets Config -----------------
spreadsheet_id = "1Cuyksq2t1xiYeokaQ5J_WGTezqMa-1TR"

KELUHAN_SHEET_GID = "1709081933"  # GID sheet keluhan


SHEETS = {
    # "Semua Produk": "900187485",
    "Course": "709240030",
    "Webinar": "1395015593"
}

st.sidebar.markdown("---")
st.sidebar.write(f"👤 Login sebagai: **{st.session_state['username']}**")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.subheader("Sumber Data")

selected_sheet = st.sidebar.selectbox(
    "Pilih Sheet",
    options=list(SHEETS.keys()),
    index=0
)

# ----------------- Load data (Google Sheets CSV export) -----------------
@st.cache_data(ttl=300)
# https://docs.google.com/spreadsheets/d/1Cuyksq2t1xiYeokaQ5J_WGTezqMa-1TR/edit?gid=1395015593#gid=1395015593
def load_sheet(spreadsheet_id: str, gid: str) -> pd.DataFrame:
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{spreadsheet_id}/export?format=csv&gid={gid}"
    )
    print("Loading data from URL:", url, "Nama Sheet:", selected_sheet)
    return pd.read_csv(url)


@st.cache_data(ttl=300)
def load_keluhan_sheet(gid: str) -> pd.DataFrame:
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"16ZB5dIUuserc3XG0lS3ldAfZAp7prUY5/export?format=csv&gid={gid}"
    )
    return pd.read_csv(url)

# Load data
try:
    gid = SHEETS[selected_sheet]
    df = load_sheet(spreadsheet_id, gid)
except Exception as e:
    st.error("Gagal memuat data dari Google Sheets.")
    st.exception(e)
    st.stop()

# Load keluhan data
try:
    keluhan_df = load_keluhan_sheet(KELUHAN_SHEET_GID)
except Exception as e:
    st.error("Gagal memuat data keluhan dari Google Sheets")
    st.exception(e)
    keluhan_df = pd.DataFrame(columns=[
        "No", "Nama", "Courses", "Tanggal Keluhan", "Keluhan"
    ])


if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()



# Make a copy to work with
df = df.copy()

keluhan_df["Tanggal Keluhan"] = pd.to_datetime(
    keluhan_df["Tanggal Keluhan"], errors="coerce"
)

keluhan_df['Tanggal Keluhan'] = keluhan_df['Tanggal Keluhan'].apply(
    lambda x: parser.parse(x, dayfirst=True) if pd.notna(x) else pd.NaT
)

df['Tanggal_dt'] = pd.to_datetime(df['Tanggal'], errors='coerce')

mask = df['Tanggal_dt'].isna()

df.loc[mask, 'Tanggal_dt'] = df.loc[mask, 'Tanggal'].apply(
    lambda x: parser.parse(x, dayfirst=True) if pd.notna(x) else pd.NaT
)

parsed = df['Tanggal_dt']

print("Gagal parse:", parsed.isna().sum())
print("Min:", df['Tanggal_dt'].min())
print("Max:", df['Tanggal_dt'].max())


assert df['Tanggal'].notna().all()
assert df['Tanggal_dt'].notna().all()
assert df['Tanggal_dt'].max()


@st.cache_resource
def load_sbert():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

sbert_model = load_sbert()

# =============================
# HYBRID SBERT KELUHAN CLASSIFIER
# =============================

def smart_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


RULES = [
    {
        "label": "Trust / Kredibilitas",
        "keywords": ["penipuan", "rekening pribadi", "ragu"]
    },
    {
        "label": "Payment",
        "keywords": ["qris", "ewallet", "e-wallet", "transfer", "shopeepay", "gopay", "bank", "biaya admin"]
    },
    {
        "label": "Admin Response",
        "keywords": ["admin", "slow", "respon lama", "di-read", "dibaca", "balas", "jutek"]
    },
    {
        "label": "Onboarding / Akses",
        "keywords": ["link", "zoom", "wa", "whatsapp", "grup", "invite", "undangan"]
    },
    {
        "label": "Content / Teknis",
        "keywords": ["rekaman", "materi", "silabus", "tidak bisa dibuka", "error"]
    },
    {
        "label": "Process / SOP",
        "keywords": ["manual", "ribet", "isi ulang", "form"]
    }
]

SBERT_CATEGORIES = {
    "Payment": "masalah pembayaran qris transfer ewallet biaya admin gagal bayar",
    "Admin Response": "admin lambat respon tidak membalas chat",
    "Onboarding / Akses": "link zoom whatsapp grup undangan belum dikirim",
    "Trust / Kredibilitas": "penipuan rekening pribadi kepercayaan ragu",
    "Content / Teknis": "materi rekaman error tidak bisa dibuka",
    "Process / SOP": "proses manual ribet isi form"
}

CATEGORY_EMB = {
    k: sbert_model.encode(v, convert_to_tensor=True)
    for k, v in SBERT_CATEGORIES.items()
}

CATEGORY_THRESHOLDS = {
    "Trust / Kredibilitas": 0.60,
    "Payment": 0.45,
    "Admin Response": 0.40,
    "Onboarding / Akses": 0.45,
    "Content / Teknis": 0.45,
    "Process / SOP": 0.45
}


def classify_keluhan(text):
    # DEFAULT SAFE RETURN
    default = ("Other", 0.0, "system")

    if not isinstance(text, str) or not text.strip():
        return default

    t = smart_normalize(text)

    # 1️⃣ RULE-BASED
    for rule in RULES:
        if any(kw in t for kw in rule["keywords"]):
            return (rule["label"], 1.0, "rule")

    # 2️⃣ SBERT SEMANTIC
    try:
        text_emb = sbert_model.encode(t, convert_to_tensor=True)

        scores = {
            label: util.cos_sim(text_emb, emb).item()
            for label, emb in CATEGORY_EMB.items()
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_label, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1]
        margin = best_score - second_score
        confidence = round((best_score + margin) / 2, 3)

        threshold = CATEGORY_THRESHOLDS.get(best_label, 0.45)

        if best_score >= threshold:
            return (best_label, confidence, "sbert")

        return ("Other", confidence, "sbert")

    except Exception as e:
        # FAIL-SAFE (never crash Streamlit)
        return default


keluhan_df[["Kategori Keluhan", "Confidence", "Method"]] = keluhan_df["Keluhan"].apply(
    lambda x: pd.Series(classify_keluhan(x))
)


keluhan_df["Tanggal Keluhan"] = pd.to_datetime(keluhan_df["Tanggal Keluhan"])


# Keep only desired columns if present
cols_to_keep = ["Judul Barang", "Harga", "qty", "sub total", "Total", "Tanggal", "Status", "Buyer Email"]
existing_keep = [c for c in cols_to_keep if c in df.columns]
if not existing_keep:
    st.error("Kolom yang diperlukan tidak ditemukan pada dataset. Periksa nama kolom di Google Sheets.")
    st.write("Kolom tersedia:", df.columns.tolist())
    st.stop()

# select available columns (this preserves others if some optional missing)

# Tanggal -> datetime
# try dayfirst True then fallback

df = df[existing_keep + ['Tanggal_dt']]

# Convert types safely
# Harga, sub total, Total -> float
for col in ["Harga", "sub total", "Total"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast='float', errors="coerce")

# qty -> int (coerce then fill 0)
if "qty" in df.columns:
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)

# Status -> category
if "Status" in df.columns:
    df["Status"] = df["Status"].astype("category")

print(pd.__version__)


print(df['Tanggal'].sort_values(ascending=False))


# Create month period column
if df["Tanggal_dt"].notna().any():
    df["Bulan"] = df["Tanggal_dt"].dt.to_period("M")
else:
    df["Bulan"] = pd.NaT

# =============================
# EXECUTIVE FEATURE ENGINEERING
# =============================

# Final status (CEO definition)
df["final_status"] = df["Status"].apply(
    lambda x: "SUCCESS" if str(x).upper() == "SUCCESS" else "CANCELLED"
)

# Revenue = Harga x qty (SOURCE OF TRUTH)
df["revenue"] = (
    df.get("Harga", 0).fillna(0) *
    df.get("qty", 0).fillna(0)
)

# Flags
df["is_success"] = df["final_status"] == "SUCCESS"
df["is_cancelled"] = df["final_status"] == "CANCELLED"

# Time features
df["year"] = df["Tanggal_dt"].dt.year
df["month"] = df["Tanggal_dt"].dt.month
df["month_name"] = df["Tanggal_dt"].dt.strftime("%b")

# Price bucket (for root-cause inference)
df["price_bucket"] = pd.cut(
    df["Harga"],
    bins=[-1, 100_000, 300_000, 10_000_000],
    labels=["Low", "Medium", "High"]
)

# ----------------- CSS + header rendering using components.html -----------------
CSS_HTML = """
<style>
.reportview-container, .main { background: linear-gradient(180deg,#f6f9fb,#ffffff); }
.kpi-card { background: #ffffff; border-radius: 12px; box-shadow: 0 6px 18px rgba(17,24,39,0.06); padding: 18px; margin-bottom: 12px; }
.kpi-value {font-size:28px; font-weight:700; color:#0f172a}
.kpi-label {font-size:12px; color:#6b7280}
.header-title {font-size:20px; font-weight:700; color:#0f172a; padding:6px 0}
.chart-card { background: #ffffff; border-radius: 12px; padding: 14px; box-shadow: 0 6px 18px rgba(17,24,39,0.04); }
</style>
<div class="header-title">Overview Operational &amp; Bisnis</div>
"""
components.html(CSS_HTML, height=90)

# ----------------- Sidebar filters -----------------
st.sidebar.image("https://cdn.lynkid.my.id/profile/11-02-2025/1739270933854_7487135", width=120)
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih halaman", [
    "Dashboard Utama",
    "Data Transaksi",
    "Analisis Strategis",
    "Layanan"])


st.sidebar.markdown("---")
st.sidebar.subheader("Filter Waktu")

# Ambil daftar bulan unik (format YYYY-MM)
available_months = (
    df["Bulan"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

# === PILIH RENTANG BULAN ===
use_range = st.sidebar.checkbox("Gunakan Rentang Bulan", value=False)

range_start, range_end = None, None
if use_range and available_months:
    range_start, range_end = st.sidebar.select_slider(
        "Rentang Bulan",
        options=available_months,
        value=(available_months[0], available_months[-1])
    )

# === PILIH BULAN TERTENTU ===
selected_months = st.sidebar.multiselect(
    "Pilih Bulan (Opsional)",
    options=available_months,
    default=[]
)


# Apply filters
mask = pd.Series(True, index=df.index)

# =============================
# FILTER BULAN (SMART LOGIC)
# =============================
if use_range and range_start and range_end:
    # PRIORITAS 1: Rentang bulan
    start_p = pd.Period(range_start)
    end_p = pd.Period(range_end)
    mask &= df["Bulan"].between(start_p, end_p)

elif selected_months:
    # PRIORITAS 2: Bulan terpilih
    selected_periods = [pd.Period(m) for m in selected_months]
    mask &= df["Bulan"].isin(selected_periods)

# else: TIDAK ADA FILTER → SEMUA DATA

filtered = df[mask].copy()

# ----------------- KPI cards -----------------
# =============================
# EXECUTIVE KPI CARDS
# =============================
kpi_df = filtered.copy()

total_revenue = kpi_df[kpi_df.is_success]["revenue"].sum()
total_transactions = len(kpi_df)

success_rate = (kpi_df["is_success"].mean() * 100) if total_transactions > 0 else 0

cancelled_value = kpi_df[kpi_df.is_cancelled]["revenue"].sum()

top_product = (
    kpi_df[kpi_df.is_success]
    .groupby("Judul Barang")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_product_name = top_product.index[0] if not top_product.empty else "-"

col1,col2,col3,col4,col5 = st.columns(5)

col1.markdown(
    f'<div class="kpi-card"><div class="kpi-value">Rp {total_revenue:,.0f}</div>'
    '<div class="kpi-label">~~Total Pemasukan (SUCCESS)</div></div>',
    unsafe_allow_html=True
)

col2.markdown(
    f'<div class="kpi-card"><div class="kpi-value">{total_transactions}</div>'
    '<div class="kpi-label">~~Total Transaksi</div></div>',
    unsafe_allow_html=True
)

col3.markdown(
    f'<div class="kpi-card"><div class="kpi-value">{success_rate:.1f}%</div>'
    '<div class="kpi-label">~~Success Ratio</div></div>',
    unsafe_allow_html=True
)

col4.markdown(
    f'<div class="kpi-card"><div class="kpi-value">Rp {cancelled_value:,.0f}</div>'
    '<div class="kpi-label">~~Cancelled Value</div></div>',
    unsafe_allow_html=True
)

col5.markdown(
    f'<div class="kpi-card"><div class="kpi-value">{top_product_name}</div>'
    '<div class="kpi-label">~~Top Product</div></div>',
    unsafe_allow_html=True
)


# ----------------- Pages -----------------
if menu == "Dashboard Utama":
    left, right = st.columns([2,1])
    with left:
        st.markdown('<div class="chart-card"><h4>Alur Pemasukan (Revenue Trend)</h4>', unsafe_allow_html=True)
        rev = filtered.copy()
        rev.info()

        print(rev.describe())
        if not rev.empty and rev['Tanggal_dt'].notna().any():
            
            rev['period'] = rev['Tanggal_dt'].dt.to_period('M').dt.to_timestamp()
            # start = pd.to_datetime(start_date).to_period('M').to_timestamp()
            # end = pd.to_datetime(end_date).to_period('M').to_timestamp()
            # full_idx = pd.date_range(start=start, end=end, freq='MS')
            rev_group = (
                rev[rev.is_success]
                .groupby('period', as_index=False)['revenue']
                .sum()
                .sort_values('period')
            )
        if rev_group.empty:
            st.info("Tidak ada data untuk rentang tanggal ini.")
        else:
            fig = px.area(
                rev_group,
                x='period',
                y='revenue',
                markers=True,
                title='Pemasukan (Revenue - SUCCESS)',
                labels={'period':'Bulan', 'revenue':'Revenue (IDR)'}
            )
            fig.update_xaxes(tickmode='linear',
                dtick='M1',
                tickformat='%b %Y',
                tickangle=-45
            )
            st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card"><h4>🎯 Product Impact Analysis</h4>', unsafe_allow_html=True)

    impact_df = (
        filtered
        .groupby("Judul Barang")
        .agg(
            revenue=("revenue","sum"),
            demand=("Judul Barang","count"),
            success_rate=("is_success","mean")
        )
    )

    impact_df["impact_score"] = (
        impact_df["revenue"].rank(pct=True) *
        impact_df["demand"].rank(pct=True) *
        impact_df["success_rate"]
    )

    impact_df = impact_df.sort_values("impact_score", ascending=False)

    st.dataframe(
        impact_df.style.format({
            "revenue": "Rp {:,.0f}",
            "success_rate": "{:.0%}",
            "impact_score": "{:.2f}"
        }),
        use_container_width=True,
        height=300
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🔍 Analisis Transaksi CANCELLED (Inferensi)")

    c1, c2, c3 = st.columns(3)

    # Cancelled by Product
    cancel_prod = (
        filtered[filtered.is_cancelled]
        .groupby("Judul Barang")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_cp = px.bar(
        cancel_prod,
        x=cancel_prod.values,
        y=cancel_prod.index,
        orientation="h",
        title="Cancelled by Product"
    )
    c1.plotly_chart(fig_cp, use_container_width=True)

    # Cancelled by Price Bucket
    cancel_price = (
        filtered[filtered.is_cancelled]
        .groupby("price_bucket")
        .size()
        .reset_index(name="count")
    )

    fig_cb = px.bar(
        cancel_price,
        x="price_bucket",
        y="count",
        title="Cancelled by Price Category"
    )
    c2.plotly_chart(fig_cb, use_container_width=True)

    # Cancelled by Month (ASCENDING by calendar month)
    cancel_month = (
        filtered[filtered.is_cancelled]
        .assign(month_num=filtered["Tanggal_dt"].dt.month)
        .groupby(["month_num", "month_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("month_num", ascending=True)
    )

    fig_cm = px.line(
        cancel_month,
        x="month_name",
        y="count",
        markers=True,
        title="Cancelled by Month"
    )
    c3.plotly_chart(fig_cm, use_container_width=True)

    # ----------------- Pie chart of participation per product (small card) -----------------
    if 'Judul Barang' in filtered.columns and 'qty' in filtered.columns:
        st.markdown('---')
        st.markdown('<div class="chart-card"><h4>Bagikan Partisipasi per Produk</h4>', unsafe_allow_html=True)
        participation_by_judul = filtered.groupby('Judul Barang')['qty'].sum().reset_index()
        participation_by_judul = participation_by_judul.sort_values('qty', ascending=False)
        if participation_by_judul.empty:
            st.info("Tidak ada data partisipasi.")
        else:
            fig_pie = px.pie(participation_by_judul, values='qty', names='Judul Barang', title='Partisipasi per Judul Barang')
            fig_pie.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig_pie, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)



    with right:
        
        st.markdown("### 🧠 Executive Insight")

        st.markdown(f"""
        - **Produk paling berdampak terhadap revenue:** **{top_product_name}**
        - **Success rate transaksi:** **{success_rate:.1f}%**
        - **Potensi revenue yang hilang:** **Rp {cancelled_value:,.0f}**
        - **Rekomendasi:** fokus scale produk dengan impact score tertinggi dan evaluasi produk dengan rasio CANCELLED tinggi.
        """)    


elif menu == "Data Transaksi":
    st.markdown('<div class="chart-card"><h4>Pendaftar per Produk</h4>', unsafe_allow_html=True)
    if 'Judul Barang' in filtered.columns:
        cnt = filtered['Judul Barang'].value_counts().reset_index()
        cnt.columns = ['Judul Barang','Jumlah']
    else:
        cnt = pd.DataFrame(columns=['Judul Barang','Jumlah'])
    if cnt.empty:
        st.info("Tidak ada data untuk ditampilkan.")
    else:
        fig2 = px.bar(cnt, x='Judul Barang', y='Jumlah', title='Jumlah Peserta per Produk')
        fig2.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=350)
        st.plotly_chart(fig2, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Tabel Transaksi Lengkap")
    st.dataframe(filtered.sort_values('Tanggal_dt', ascending=False).reset_index(drop=True))

elif menu == "Analisis Strategis":
    st.subheader("📊 Analisis Strategis (CEO View)")
    st.caption("Fokus pada kebocoran revenue dan keputusan produk")

    # =========================
    # A. Revenue SUCCESS vs CANCELLED (Trend)
    # =========================
    st.markdown("### 💰 Revenue: SUCCESS vs CANCELLED")

    filtered = filtered.copy()
    filtered["period"] = (
        filtered["Tanggal_dt"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    trend_df = (
        filtered
        .groupby(["period", "final_status"], as_index=False)
        ["revenue"]
        .sum()
    )

    fig_trend = px.line(
        trend_df,
        x="period",
        y="revenue",
        color="final_status",
        markers=True,
        title="Revenue Trend: SUCCESS vs CANCELLED",
        labels={"period": "Bulan", "revenue": "Revenue (IDR)", "final_status": "Status"}
    )
    fig_trend.update_xaxes(tickformat="%b %Y", dtick="M1", tickangle=-45)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # =========================
    # B. Product Quadrant (Decision Map)
    # =========================
    st.markdown("### 🎯 Product Decision Quadrant")

    quad_df = (
        filtered
        .groupby("Judul Barang")
        .agg(
            revenue=("revenue", "sum"),
            demand=("Judul Barang", "count"),
            success_rate=("is_success", "mean")
        )
        .reset_index()
    )

    fig_quad = px.scatter(
        quad_df,
        x="demand",
        y="revenue",
        size="revenue",
        color="success_rate",
        hover_name="Judul Barang",
        color_continuous_scale="Blues",
        title="Product Quadrant: Demand vs Revenue",
        labels={
            "demand": "Jumlah Transaksi (Demand)",
            "revenue": "Revenue (IDR)",
            "success_rate": "Success Rate"
        }
    )

    st.plotly_chart(fig_quad, use_container_width=True)

    st.markdown("""
    **Interpretasi CEO:**
    - 🟢 **Kanan Atas** → SCALE
    - 🟡 **Kanan Bawah** → OPTIMASI HARGA / UPSELL
    - 🔵 **Kiri Atas** → NICHE
    - 🔴 **Kiri Bawah** → EVALUASI / SUNSET
    """)

    st.markdown("---")

    # =========================
    # C. Cancelled Concentration
    # =========================
    st.markdown("### ❌ Konsentrasi CANCELLED")

    c1, c2 = st.columns(2)

    # Cancelled by Product
    cancel_prod = (
        filtered[filtered.is_cancelled]
        .groupby("Judul Barang")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_cp = px.bar(
        cancel_prod,
        x=cancel_prod.values,
        y=cancel_prod.index,
        orientation="h",
        title="Cancelled by Product"
    )
    c1.plotly_chart(fig_cp, use_container_width=True)

    # Cancelled by Month
    cancel_month = (
        filtered[filtered.is_cancelled]
        .groupby(filtered["Tanggal_dt"].dt.strftime("%b"))
        .size()
        .reset_index(name="count")
    )
    cancel_month.columns = ["Bulan", "Jumlah"]

    fig_cm = px.bar(
        cancel_month,
        x="Bulan",
        y="Jumlah",
        title="Cancelled by Month"
    )
    c2.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**Insight:** Fokuskan perbaikan pada produk dan bulan dengan CANCELLED tertinggi.")



elif menu == "Layanan":
    st.markdown("### 🚨 Ringkasan Risiko Layanan")

    avg_conf = keluhan_df["Confidence"].mean()


    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Total Keluhan", len(keluhan_df))
    c2.metric(
        "Kategori Dominan",
        keluhan_df["Kategori Keluhan"].value_counts().idxmax()
    )
    c3.metric(
        "Produk Paling Bermasalah",
        keluhan_df["Courses"].value_counts().idxmax()
    )
    c4.metric("Rata-rata Confidence", f"{avg_conf:.2f}")

    st.markdown("### 📊 Distribusi Jenis Keluhan")

    keluhan_cat = (
        keluhan_df["Kategori Keluhan"]
        .value_counts()
        .reset_index()
    )
    keluhan_cat.columns = ["Kategori", "Jumlah"]

    fig_cat = px.bar(
        keluhan_cat,
        x="Kategori",
        y="Jumlah",
        title="Distribusi Keluhan berdasarkan Kategori"
    )

    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### 📝 Detail Keluhan")

    st.dataframe(
        keluhan_df.sort_values("Tanggal Keluhan", ascending=False),
        use_container_width=True,
        height=350
    )

    st.markdown("### ⚠️ Keluhan Low Confidence (< 0.45)")
    low_conf = keluhan_df[keluhan_df["Confidence"] < 0.45]

    st.dataframe(
        low_conf.sort_values("Confidence"),
        use_container_width=True,
        height=250
    )

# ----------------- Export utilities -----------------
@st.cache_data
def to_excel(df_in):
    output = BytesIO()
    # using pandas ExcelWriter (openpyxl)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_in.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


st.sidebar.markdown("---")
st.sidebar.subheader("Export Report")


def generate_executive_pdf(
    total_revenue,
    total_transactions,
    success_rate,
    cancelled_value,
    top_product_name,
    impact_df
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    elements = []

    # ===== TITLE =====
    elements.append(Paragraph(
        "<b>Executive Sales Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    # ===== KPI SUMMARY =====
    elements.append(Paragraph("<b>Key Performance Summary</b>", styles["Heading2"]))

    kpi_table = Table([
        ["Metric", "Value"],
        ["Total Revenue", f"Rp {total_revenue:,.0f}"],
        ["Total Transactions", f"{total_transactions}"],
        ["Success Rate", f"{success_rate:.1f}%"],
        ["Cancelled Value", f"Rp {cancelled_value:,.0f}"],
        ["Top Product", top_product_name]
    ])

    elements.append(kpi_table)
    elements.append(Spacer(1, 16))

    # ===== PRODUCT IMPACT =====
    elements.append(Paragraph("<b>Top Product Impact</b>", styles["Heading2"]))

    impact_top = impact_df.head(5).reset_index()

    impact_table_data = [["Product", "Revenue", "Demand", "Success Rate"]]
    for _, r in impact_top.iterrows():
        impact_table_data.append([
            r["Judul Barang"],
            f"Rp {r['revenue']:,.0f}",
            int(r["demand"]),
            f"{r['success_rate']:.0%}"
        ])

    impact_table = Table(impact_table_data)
    elements.append(impact_table)
    elements.append(Spacer(1, 16))

    # ===== EXECUTIVE INSIGHT =====
    elements.append(Paragraph("<b>Executive Insight</b>", styles["Heading2"]))
    elements.append(Paragraph(
        f"""
        Berdasarkan data transaksi:
        <br/>• Produk paling berdampak terhadap revenue adalah <b>{top_product_name}</b>.
        <br/>• Tingkat keberhasilan transaksi berada di <b>{success_rate:.1f}%</b>.
        <br/>• Potensi revenue yang hilang akibat transaksi gagal mencapai <b>Rp {cancelled_value:,.0f}</b>.
        <br/><br/>
        <b>Rekomendasi:</b> Fokuskan scaling pada produk dengan impact tinggi dan mitigasi risiko pada produk dengan cancelled rate tinggi.
        """,
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


if st.sidebar.button("📄 Download Executive PDF"):
    pdf_buffer = generate_executive_pdf(
        total_revenue=total_revenue,
        total_transactions=total_transactions,
        success_rate=success_rate,
        cancelled_value=cancelled_value,
        top_product_name=top_product_name,
        impact_df=impact_df
    )

    st.sidebar.download_button(
        label="⬇️ Download PDF",
        data=pdf_buffer,
        file_name="executive_sales_report.pdf",
        mime="application/pdf"
    )