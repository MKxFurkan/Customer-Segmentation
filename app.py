"""Streamlit UI for the customer segmentation project."""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "marketing_campaign.csv"
MODEL_PATH = ROOT / "models" / "clustering_pipeline.pkl"

# Palette: soft charcoal like the Claude UI, with a warm sand accent.
BG = "#262624"        # page
SURFACE = "#2F2F2C"   # cards
RAISED = "#383835"    # inputs, hover
BORDER = "#43433F"
TEXT = "#ECEBE6"
MUTED = "#A3A29A"
ACCENT = "#E3C48D"

FEATURES = [
    "Income", "Total_Spending", "Total_Purchases",
    "Children", "NumWebVisitsMonth", "NumDealsPurchases",
]
LABELS = {
    "Income": "Yearly income",
    "Total_Spending": "Total spending (2 years)",
    "Total_Purchases": "Purchases (web, catalog, store)",
    "Children": "Children at home",
    "NumWebVisitsMonth": "Web visits last month",
    "NumDealsPurchases": "Purchases with a deal",
}
# Ordered from highest to lowest average spending.
SEGMENTS = ["High-value", "Moderate-value", "Low-spending"]
OFFERS = {
    "High-value": "Premium product offers",
    "Moderate-value": "Discounts and bundle offers",
    "Low-spending": "Entry-level and engagement offers",
}
# One distinct, muted hue per segment so they stay separable on charcoal.
COLORS = {"High-value": "#E3C48D", "Moderate-value": "#8FB8A8", "Low-spending": "#8E9BD1"}
SEGMENT_COLORS = [COLORS[s] for s in SEGMENTS]
FONT = "DM Sans"

st.set_page_config(page_title="Customer Segmentation", page_icon="☕", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;600&display=swap');
:root {{
    --bg: {BG}; --surface: {SURFACE}; --raised: {RAISED}; --border: {BORDER};
    --text: {TEXT}; --muted: {MUTED}; --accent: {ACCENT};
}}
"""
    + """
html, body, .stApp, [class*="st-"], button, input, textarea, select {
    font-family: 'DM Sans', system-ui, sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* Type */
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }
h3 { font-weight: 500 !important; font-size: 1.3rem !important; padding-top: .25rem !important; }
.hero h1 { font-size: 2.6rem; font-weight: 500; margin: 0; padding: 0; line-height: 1.15; }
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 60ch; margin: .6rem 0 0; }
.section-note { color: var(--muted); font-size: .92rem; margin: -.4rem 0 1rem; }

/* Tabs */
[role="tablist"] {
    display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 1rem !important;
    margin-top: 1.75rem; border: none !important; box-shadow: none !important;
}
[role="tablist"] button[role="tab"] {
    height: auto !important; min-height: 4rem; padding: 1rem 1.25rem !important;
    justify-content: flex-start !important;
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; transition: background .15s, border-color .15s;
}
[role="tablist"] button[role="tab"] p {
    font-family: 'Fraunces', Georgia, serif !important; font-size: 1.15rem !important;
    font-weight: 500 !important; color: var(--muted) !important;
}
[role="tablist"] button[role="tab"]:hover { background: var(--raised) !important; }
[role="tablist"] button[role="tab"][aria-selected="true"] {
    background: var(--raised) !important; border-color: var(--accent) !important;
    box-shadow: inset 0 -3px 0 var(--accent);
}
[role="tablist"] button[role="tab"][aria-selected="true"] p { color: var(--text) !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabPanel"] { padding-top: 2rem; }
@media (max-width: 700px) { [role="tablist"] { grid-template-columns: 1fr; } }

/* Containers and forms */
[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .panel-marker) {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem;
}
.group-title { font-weight: 600; font-size: .95rem; margin: 0 0 .75rem; }
.group-gap { height: .75rem; }

/* Inputs */
[data-testid="stNumberInput"] label p, [data-testid="stSelectbox"] label p {
    color: var(--muted); font-size: .88rem; font-weight: 500;
}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
    background: var(--raised) !important; border-color: var(--border) !important; border-radius: 8px;
}
[data-baseweb="input"] input, [data-baseweb="base-input"] input {
    background: var(--raised) !important; color: var(--text) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
}
[data-testid="stNumberInput"] button { background: var(--raised); color: var(--muted); }

/* Primary button */
[data-testid="stFormSubmitButton"] button {
    background: var(--accent); color: #1d1d1b; border: none; font-weight: 600;
    border-radius: 8px; padding: .6rem 1rem; transition: filter .15s;
}
[data-testid="stFormSubmitButton"] button p { color: #1d1d1b; font-weight: 600; }
[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); color: #1d1d1b; }

/* Result and empty state */
.result { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.result .tag { display: inline-flex; align-items: center; gap: .5rem; color: var(--muted); font-size: .9rem; }
.dot { width: .65rem; height: .65rem; border-radius: 50%; display: inline-block; }
.result h2 { font-size: 2rem; font-weight: 500; margin: .5rem 0 .25rem; padding: 0; }
.result .offer { color: var(--muted); margin: 0; }
.result .offer b { color: var(--text); font-weight: 600; }
.empty { border: 1px dashed var(--border); border-radius: 12px; padding: 2rem 1.5rem; color: var(--muted); }
.empty b { color: var(--text); display: block; margin-bottom: .25rem; font-size: 1.05rem; }

/* Segment cards */
.seg-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.4rem; height: 100%; }
.seg-card .name { display: flex; align-items: center; gap: .5rem; font-weight: 600; }
.seg-card .count { font-family: 'Fraunces', Georgia, serif; font-size: 2rem; font-weight: 500; margin: .5rem 0 0; }
.seg-card .meta { color: var(--muted); font-size: .9rem; margin: .1rem 0 .9rem; }
.seg-card .offer { border-top: 1px solid var(--border); padding-top: .75rem; font-size: .9rem; color: var(--muted); }
.seg-card .offer b { color: var(--text); font-weight: 500; }

/* Tables, alerts */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 10px; }

/* About cards */
.note-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1.5rem; height: 100%; }
.note-card h3 { margin: 0 0 .6rem; padding: 0; }
.note-card p, .note-card li { color: var(--muted); line-height: 1.6; }
.note-card ul { padding-left: 1.1rem; margin: 0; }
.note-card li { margin-bottom: .4rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """Apply the same cleaning and feature engineering as the notebook."""
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df[df["Income"].isna() | (df["Income"] < 300000)]
    df = df[df["Year_Birth"] >= 1940]
    df = df[~df["Marital_Status"].isin(["YOLO", "Absurd"])].copy()
    df["Total_Spending"] = df.filter(like="Mnt").sum(axis=1)
    df["Total_Purchases"] = df[
        ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    ].sum(axis=1)
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    return df


missing = [str(p.relative_to(ROOT)) for p in (DATA_PATH, MODEL_PATH) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Run the notebook to create the model.")
    st.stop()

model = load_model()
if model.named_steps["clustering"].n_clusters != len(SEGMENTS):
    st.error("This app expects a 3-cluster model.")
    st.stop()

df = load_data()
df["Cluster"] = model.predict(df[FEATURES])

# Name clusters by average spending, so names don't depend on K-Means numbering.
order = df.groupby("Cluster")["Total_Spending"].mean().sort_values(ascending=False).index
names = dict(zip(order, SEGMENTS))
df["Segment"] = df["Cluster"].map(names)


def dot(segment):
    return f'<span class="dot" style="background:{COLORS[segment]}"></span>'


# Header
st.markdown(
    f"""
<div class="hero">
    <h1>Customer Segmentation</h1>
    <p>Group customers by spending and shopping behaviour, then pick the offer worth testing
    for each group. Based on {len(df):,} customers.</p>
</div>
""",
    unsafe_allow_html=True,
)

find_tab, explore_tab, about_tab = st.tabs(["Find a segment", "Compare segments", "About"])

# Find a segment
with find_tab:
    form_col, result_col = st.columns([1.15, 1], gap="large")

    with form_col:
        st.markdown("### Customer details")
        st.markdown(
            '<p class="section-note">Enter the customer\'s figures to see which segment fits.</p>',
            unsafe_allow_html=True,
        )
        with st.form("customer"):
            st.markdown('<p class="group-title">Value</p>', unsafe_allow_html=True)
            v1, v2 = st.columns(2)
            income = v1.number_input("Yearly income", 0, 300000, 51000, step=1000)
            spending = v2.number_input("Total spending (2 years)", 0, 5000, 400, step=50)
            purchases = v1.number_input("Purchases (web, catalog, store)", 0, 60, 12)
            deals = v2.number_input("Purchases with a deal", 0, 30, 2)

            st.markdown('<div class="group-gap"></div>', unsafe_allow_html=True)
            st.markdown('<p class="group-title">Household and activity</p>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            children = b1.number_input("Children at home", 0, 5, 1)
            visits = b2.number_input("Web visits last month", 0, 31, 5)

            st.markdown('<div class="group-gap"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Find segment", width="stretch")

    with result_col:
        st.markdown("### Result")
        st.markdown(
            '<p class="section-note">The segment, the offer to test and how this customer compares.</p>',
            unsafe_allow_html=True,
        )
        if submitted:
            row = pd.DataFrame(
                [[income, spending, purchases, children, visits, deals]], columns=FEATURES
            )
            segment = names[model.predict(row)[0]]
            outside = [
                LABELS[f] for f in FEATURES
                if not df[f].min() <= row[f].iloc[0] <= df[f].max()
            ]
            if outside:
                st.warning(
                    f"Outside the range of the training data: {', '.join(outside)}. "
                    "Treat this result with caution."
                )
            st.markdown(
                f"""
<div class="result">
    <span class="tag">{dot(segment)} Predicted segment</span>
    <h2>{segment} customer</h2>
    <p class="offer">Offer to test: <b>{OFFERS[segment]}</b></p>
</div>
""",
                unsafe_allow_html=True,
            )
            comparison = pd.DataFrame({
                "This customer": row.iloc[0],
                "Segment average": df.loc[df["Segment"] == segment, FEATURES].mean(),
                "All customers": df[FEATURES].mean(),
            }).rename(index=LABELS)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.dataframe(comparison.style.format("{:,.1f}"), width="stretch")
        else:
            st.markdown(
                """
<div class="empty"><b>No customer checked yet</b>
Fill in the details and select Find segment.</div>
""",
                unsafe_allow_html=True,
            )

# Compare segments
with explore_tab:
    grouped = df.groupby("Segment")

    for column, segment in zip(st.columns(3, gap="medium"), SEGMENTS):
        size = int((df["Segment"] == segment).sum())
        column.markdown(
            f"""
<div class="seg-card">
    <div class="name">{dot(segment)} {segment}</div>
    <div class="count">{size:,}</div>
    <div class="meta">customers, {size / len(df):.0%} of total</div>
    <div class="offer">Offer to test: <b>{OFFERS[segment]}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

    color = alt.Color(
        "Segment:N", scale=alt.Scale(domain=SEGMENTS, range=SEGMENT_COLORS),
        legend=alt.Legend(orient="top", title=None),
    )

    def styled(chart, height=340):
        return (
            chart.properties(
                height=height,
                background=SURFACE,
                padding={"left": 16, "right": 16, "top": 16, "bottom": 16},
            )
            .configure_axis(
                labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domainColor=BORDER,
                tickColor=BORDER, labelFont=FONT, titleFont=FONT, titleFontWeight=500,
            )
            .configure_legend(labelColor=TEXT, labelFont=FONT, symbolType="circle")
            .configure_view(stroke=None)
        )

    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    chart_left, chart_right = st.columns([1.3, 1], gap="large")

    with chart_left:
        st.markdown("### Income and spending")
        st.markdown(
            '<p class="section-note">Each point is one customer.</p>', unsafe_allow_html=True
        )
        scatter = alt.Chart(df.dropna(subset=["Income"])).mark_point(
            filled=True, size=40, opacity=0.75
        ).encode(
            x=alt.X("Income:Q", title="Yearly income"),
            y=alt.Y("Total_Spending:Q", title="Total spending"),
            color=color,
            shape=alt.Shape("Segment:N", scale=alt.Scale(domain=SEGMENTS), legend=None),
            tooltip=["Segment", "Income", "Total_Spending", "Total_Purchases"],
        )
        st.altair_chart(styled(scatter), width="stretch", theme=None)

    with chart_right:
        st.markdown("### Compare one measure")
        choice = st.selectbox(
            "Measure", FEATURES + ["Response"],
            format_func=lambda f: LABELS.get(f, "Response rate, latest campaign (%)"),
        )
        values = grouped[choice].mean().reindex(SEGMENTS)
        if choice == "Response":
            values *= 100
        bars = alt.Chart(values.rename("Average").reset_index()).mark_bar(
            cornerRadiusEnd=5
        ).encode(
            x=alt.X("Average:Q", title="Average"),
            y=alt.Y("Segment:N", sort=SEGMENTS, title=None),
            color=color,
            tooltip=["Segment", alt.Tooltip("Average:Q", format=",.2f")],
        )
        st.altair_chart(styled(bars, height=260), width="stretch", theme=None)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("### Segment averages")
    table = grouped[FEATURES].mean().reindex(SEGMENTS).rename(columns=LABELS)
    table["Response rate (%)"] = grouped["Response"].mean() * 100
    st.dataframe(table.style.format("{:,.1f}"), width="stretch")

# About
with about_tab:
    how, limits = st.columns(2, gap="large")
    how.markdown(
        """
<div class="note-card">
    <h3>How it works</h3>
    <p>A scikit-learn pipeline fills missing income with the median, log-transforms
    spending, scales all six features and assigns one of three K-Means clusters.</p>
    <p>Segments are named by average spending, so the names stay correct even if
    K-Means numbers the clusters differently.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    limits.markdown(
        """
<div class="note-card">
    <h3>Limitations</h3>
    <ul>
        <li>The segments overlap. Customers near a boundary could reasonably fit either group.</li>
        <li>Segment names summarise averages and do not describe every customer.</li>
        <li>Offers are suggestions that have not been tested.</li>
        <li>Response rate was not used for clustering. It describes past behaviour only.</li>
        <li>The model reflects this dataset only.</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )
