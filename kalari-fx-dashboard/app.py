"""
Kalari Fashion FX Bleed Dashboard
Streamlit app — run with: streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import load_and_enrich

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kalari Fashion — FX Bleed Dashboard",
    page_icon="💸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_and_enrich()


df_full = get_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

min_date = df_full["auth_timestamp"].dt.date.min()
max_date = df_full["auth_timestamp"].dt.date.max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
# Unpack safely (user might select a single date)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_filter, end_filter = date_range
else:
    start_filter = end_filter = date_range[0] if date_range else min_date

selected_processors = st.sidebar.multiselect(
    "Processors",
    options=sorted(df_full["processor"].unique()),
    default=sorted(df_full["processor"].unique()),
)

selected_currencies = st.sidebar.multiselect(
    "Currencies",
    options=sorted(df_full["presentment_currency"].unique()),
    default=sorted(df_full["presentment_currency"].unique()),
)

# Apply filters
mask = (
    (df_full["auth_timestamp"].dt.date >= start_filter)
    & (df_full["auth_timestamp"].dt.date <= end_filter)
    & (df_full["processor"].isin(selected_processors))
    & (df_full["presentment_currency"].isin(selected_currencies))
)
df = df_full[mask].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💸 Kalari Fashion — FX Bleed Dashboard")
st.caption(f"Showing **{len(df):,}** transactions · {start_filter} → {end_filter}")

if df.empty:
    st.warning("No transactions match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
total_loss   = df["fx_loss_inr"].sum()
avg_margin   = df["fx_margin_pct"].mean()
num_txns     = len(df)
worst_proc   = df.groupby("processor")["fx_margin_pct"].mean().idxmax()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total FX Loss (₹)", f"₹{total_loss:,.0f}")
k2.metric("Avg FX Margin %", f"{avg_margin:.2f}%")
k3.metric("Transactions", f"{num_txns:,}")
k4.metric("Worst Processor", worst_proc)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Currency Pairs",
    "🏦 Processors",
    "📅 Time Series",
    "🔍 Transactions",
    "🚀 Stretch Goals",
])

# ============================================================
# Tab 1 — Currency Pairs
# ============================================================
with tab1:
    st.subheader("FX Loss by Currency Pair")

    pair_stats = (
        df.groupby("currency_pair")
        .agg(
            avg_margin=("fx_margin_pct", "mean"),
            total_loss=("fx_loss_inr", "sum"),
            count=("fx_loss_inr", "count"),
        )
        .reset_index()
        .sort_values("avg_margin", ascending=True)
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig1a = px.bar(
            pair_stats,
            x="avg_margin",
            y="currency_pair",
            orientation="h",
            title="Avg FX Margin % by Currency Pair",
            labels={"avg_margin": "Avg Margin %", "currency_pair": "Pair"},
            color="avg_margin",
            color_continuous_scale="Reds",
        )
        fig1a.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig1a, use_container_width=True)

    with col_b:
        fig1b = px.bar(
            pair_stats.sort_values("total_loss", ascending=True),
            x="total_loss",
            y="currency_pair",
            orientation="h",
            title="Total FX Loss (₹) by Currency Pair",
            labels={"total_loss": "Total Loss (₹)", "currency_pair": "Pair"},
            color="total_loss",
            color_continuous_scale="Oranges",
        )
        fig1b.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig1b, use_container_width=True)

    # Top 3 problem areas
    st.subheader("Top Problem Areas")
    top3 = (
        df.groupby(["currency_pair", "processor"])
        .agg(total_loss=("fx_loss_inr", "sum"), avg_margin=("fx_margin_pct", "mean"))
        .reset_index()
        .nlargest(3, "total_loss")
    )

    colors = ["🔴", "🟡", "🟡"]
    for idx, (_, row) in enumerate(top3.iterrows()):
        icon = colors[idx] if idx < len(colors) else "⚪"
        st.markdown(
            f"{icon} **{row['currency_pair']} via {row['processor']}**: "
            f"₹{row['total_loss']:,.0f} total loss · {row['avg_margin']:.2f}% avg margin"
        )


# ============================================================
# Tab 2 — Processors
# ============================================================
with tab2:
    st.subheader("FX Loss by Processor")

    proc_stats = (
        df.groupby("processor")
        .agg(
            avg_margin=("fx_margin_pct", "mean"),
            total_loss=("fx_loss_inr", "sum"),
            count=("fx_loss_inr", "count"),
        )
        .reset_index()
        .sort_values("avg_margin", ascending=False)
    )

    col_c, col_d = st.columns(2)

    with col_c:
        fig2a = px.bar(
            proc_stats,
            x="processor",
            y="avg_margin",
            title="Avg FX Margin % by Processor",
            labels={"avg_margin": "Avg Margin %", "processor": "Processor"},
            color="processor",
            text_auto=".2f",
        )
        fig2a.update_traces(textposition="outside")
        fig2a.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2a, use_container_width=True)

    with col_d:
        fig2b = px.scatter(
            df,
            x="presentment_amount",
            y="fx_loss_inr",
            color="processor",
            title="Transaction Size vs FX Loss",
            labels={
                "presentment_amount": "Presentment Amount (local currency)",
                "fx_loss_inr": "FX Loss (₹)",
                "processor": "Processor",
            },
            opacity=0.6,
            hover_data=["currency_pair", "fx_margin_pct"],
        )
        fig2b.update_layout(height=380)
        st.plotly_chart(fig2b, use_container_width=True)

    # Summary table
    st.dataframe(
        proc_stats.rename(columns={
            "processor": "Processor",
            "avg_margin": "Avg Margin %",
            "total_loss": "Total Loss (₹)",
            "count": "Transactions",
        }).style.format({"Avg Margin %": "{:.2f}", "Total Loss (₹)": "₹{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Tab 3 — Time Series
# ============================================================
with tab3:
    st.subheader("FX Loss Over Time")

    df_ts = df.copy()
    df_ts["month"] = df_ts["auth_timestamp"].dt.to_period("M").astype(str)

    monthly_loss = (
        df_ts.groupby("month")
        .agg(total_loss=("fx_loss_inr", "sum"), avg_margin=("fx_margin_pct", "mean"))
        .reset_index()
    )

    col_e, col_f = st.columns(2)

    with col_e:
        fig3a = px.line(
            monthly_loss,
            x="month",
            y="total_loss",
            markers=True,
            title="Monthly Total FX Loss (₹)",
            labels={"month": "Month", "total_loss": "Total FX Loss (₹)"},
        )
        fig3a.update_traces(line_color="#e74c3c", marker_color="#e74c3c")
        fig3a.update_layout(height=360)
        st.plotly_chart(fig3a, use_container_width=True)

    with col_f:
        fig3b = px.line(
            monthly_loss,
            x="month",
            y="avg_margin",
            markers=True,
            title="Monthly Avg FX Margin %",
            labels={"month": "Month", "avg_margin": "Avg Margin %"},
        )
        fig3b.update_traces(line_color="#e67e22", marker_color="#e67e22")
        fig3b.update_layout(height=360)
        st.plotly_chart(fig3b, use_container_width=True)

    # Monthly breakdown by processor
    st.subheader("Monthly Loss by Processor")
    monthly_proc = (
        df_ts.groupby(["month", "processor"])["fx_loss_inr"]
        .sum()
        .reset_index()
    )
    fig3c = px.bar(
        monthly_proc,
        x="month",
        y="fx_loss_inr",
        color="processor",
        barmode="stack",
        title="Monthly FX Loss Stacked by Processor",
        labels={"month": "Month", "fx_loss_inr": "FX Loss (₹)", "processor": "Processor"},
    )
    fig3c.update_layout(height=360)
    st.plotly_chart(fig3c, use_container_width=True)


# ============================================================
# Tab 4 — Transactions Table
# ============================================================
with tab4:
    st.subheader("Transaction Detail")

    display_cols = [
        "transaction_id",
        "auth_timestamp",
        "settlement_date",
        "currency_pair",
        "presentment_amount",
        "settlement_amount",
        "mid_market_rate",
        "expected_inr",
        "fx_loss_inr",
        "fx_margin_pct",
        "processor",
        "is_anomaly",
    ]

    df_display = df[display_cols].sort_values("fx_loss_inr", ascending=False).reset_index(drop=True)

    # Style anomalies in red
    def highlight_anomaly(row):
        color = "background-color: #ffcccc" if row["is_anomaly"] else ""
        return [color] * len(row)

    styled = df_display.style.apply(highlight_anomaly, axis=1).format({
        "presentment_amount": "{:,.2f}",
        "settlement_amount":  "₹{:,.2f}",
        "expected_inr":       "₹{:,.2f}",
        "fx_loss_inr":        "₹{:,.2f}",
        "fx_margin_pct":      "{:.2f}%",
        "mid_market_rate":    "{:.5f}",
    })

    st.dataframe(styled, use_container_width=True, height=450)

    # Export button
    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_bytes,
        file_name="kalari_fx_transactions.csv",
        mime="text/csv",
    )

    anomaly_count = df_display["is_anomaly"].sum()
    if anomaly_count > 0:
        st.warning(f"⚠️ {anomaly_count} anomalous transactions (margin > 5%) highlighted in red above.")


# ============================================================
# Tab 5 — Stretch Goals
# ============================================================
with tab5:
    st.subheader("Stretch Goals")

    # ----------------------------------------------------------
    # Stretch A — Auth-time vs Settlement-time rate locking
    # ----------------------------------------------------------
    with st.expander("📌 Stretch A — Auth-time vs Settlement-time Rate Locking", expanded=True):
        st.markdown(
            """
            **Question:** Is it better for Kalari Fashion if the processor locks the exchange rate
            at authorization time or at settlement time?
            """
        )

        auth_loss       = df["fx_loss_inr"].sum()
        settlement_loss = df["settlement_time_loss_inr"].sum()
        diff            = settlement_loss - auth_loss

        col_sa1, col_sa2, col_sa3 = st.columns(3)
        col_sa1.metric("Loss if locked at Auth", f"₹{auth_loss:,.0f}")
        col_sa2.metric("Loss if locked at Settlement", f"₹{settlement_loss:,.0f}")
        col_sa3.metric(
            "Difference (Settlement − Auth)",
            f"₹{diff:,.0f}",
            delta=f"{'Settlement worse' if diff > 0 else 'Auth worse'}",
            delta_color="inverse",
        )

        fig_sa = go.Figure(go.Bar(
            x=["Auth-time Locking", "Settlement-time Locking"],
            y=[auth_loss, settlement_loss],
            marker_color=["#3498db", "#e74c3c"],
            text=[f"₹{auth_loss:,.0f}", f"₹{settlement_loss:,.0f}"],
            textposition="outside",
        ))
        fig_sa.update_layout(
            title="FX Loss Comparison: Auth vs Settlement Rate Locking",
            yaxis_title="Total FX Loss (₹)",
            height=380,
        )
        st.plotly_chart(fig_sa, use_container_width=True)

        if diff > 0:
            st.info(
                f"💡 Settlement-time locking is **₹{diff:,.0f} more expensive** than auth-time locking "
                f"in this period — suggests INR depreciation benefited processors."
            )
        else:
            st.info(
                f"💡 Auth-time locking is **₹{abs(diff):,.0f} more expensive** than settlement-time "
                f"locking in this period — INR appreciated, so earlier locking was worse."
            )

    # ----------------------------------------------------------
    # Stretch B — Pricing recommendations
    # ----------------------------------------------------------
    with st.expander("💰 Stretch B — Pricing Recommendations by Market", expanded=True):
        st.markdown(
            """
            **Question:** How much should Kalari Fashion increase prices in each market
            to recover the FX margin lost to processors?
            """
        )

        pricing = (
            df.groupby("presentment_currency")
            .agg(
                avg_loss_pct=("fx_margin_pct", "mean"),
                total_loss=("fx_loss_inr", "sum"),
                txn_count=("fx_loss_inr", "count"),
            )
            .reset_index()
            .sort_values("avg_loss_pct", ascending=False)
        )

        def action_label(pct):
            if pct > 2.5:
                return "🔴 Raise prices now"
            elif pct > 1.5:
                return "🟡 Consider increase"
            else:
                return "🟢 Acceptable"

        pricing["recommended_increase_%"] = pricing["avg_loss_pct"].round(2)
        pricing["action"] = pricing["avg_loss_pct"].apply(action_label)

        st.dataframe(
            pricing.rename(columns={
                "presentment_currency":  "Currency",
                "avg_loss_pct":          "Avg FX Loss %",
                "total_loss":            "Total Loss (₹)",
                "txn_count":             "Transactions",
                "recommended_increase_%":"Recommended Price Increase %",
                "action":                "Action",
            }).style.format({
                "Avg FX Loss %":               "{:.2f}%",
                "Total Loss (₹)":              "₹{:,.0f}",
                "Recommended Price Increase %": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig_sb = px.bar(
            pricing,
            x="presentment_currency",
            y="recommended_increase_%",
            color="recommended_increase_%",
            color_continuous_scale="RdYlGn_r",
            title="Recommended Price Increase % by Currency",
            labels={
                "presentment_currency":  "Currency",
                "recommended_increase_%": "Recommended Increase %",
            },
            text_auto=".2f",
        )
        fig_sb.update_traces(textposition="outside")
        fig_sb.update_layout(coloraxis_showscale=False, height=360)
        st.plotly_chart(fig_sb, use_container_width=True)

    # ----------------------------------------------------------
    # Stretch C — Anomaly Detection
    # ----------------------------------------------------------
    with st.expander("🚨 Stretch C — Anomaly Detection (Margin > 5%)", expanded=True):
        st.markdown(
            """
            **Question:** Are there transactions with suspiciously high spreads that warrant
            immediate investigation?
            """
        )

        anomalies = df[df["is_anomaly"]].sort_values("fx_margin_pct", ascending=False)
        total_anomalies = len(anomalies)
        anomaly_pct     = total_anomalies / len(df) * 100 if len(df) > 0 else 0

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Anomalous Transactions", total_anomalies)
        col_c2.metric("% of Volume", f"{anomaly_pct:.1f}%")
        col_c3.metric("Total Anomaly Loss (₹)", f"₹{anomalies['fx_loss_inr'].sum():,.0f}")

        if total_anomalies > 0:
            anom_by_proc = (
                anomalies.groupby("processor")
                .agg(count=("fx_loss_inr", "count"), total_loss=("fx_loss_inr", "sum"))
                .reset_index()
            )
            fig_sc = px.bar(
                anom_by_proc,
                x="processor",
                y="count",
                color="processor",
                title="Anomalous Transactions by Processor",
                labels={"count": "# Anomalies", "processor": "Processor"},
                text_auto=True,
            )
            fig_sc.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_sc, use_container_width=True)

            anom_display = anomalies[[
                "transaction_id", "auth_timestamp", "currency_pair",
                "presentment_amount", "fx_loss_inr", "fx_margin_pct", "processor",
            ]].head(20)
            st.dataframe(
                anom_display.style.format({
                    "presentment_amount": "{:,.2f}",
                    "fx_loss_inr":        "₹{:,.2f}",
                    "fx_margin_pct":      "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No anomalies detected with current filters.")
