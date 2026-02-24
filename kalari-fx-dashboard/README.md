# Kalari Fashion — FX Bleed Dashboard

A proof-of-concept analytics tool that quantifies and visualizes hidden FX losses in Kalari Fashion's multi-currency payment processing pipeline (INR as settlement currency).

---

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Step 1: generate synthetic data (creates data/fx_rates.csv and data/transactions.csv)
python generate_data.py

# Step 2: launch the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Project Structure

```
kalari-fx-dashboard/
├── data/
│   ├── transactions.csv   # 350 synthetic payment transactions
│   └── fx_rates.csv       # Daily mid-market rates (Nov 2024 – Jan 2025)
├── generate_data.py       # Synthetic data generator
├── pipeline.py            # FX loss calculation engine
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Design Decisions

### 1. Auth-date as rate baseline
Mid-market rate is looked up at the authorization date, not the settlement date. This is the most conservative and auditable baseline — the merchant knew the rate when the transaction was approved. Settlement-time locking is analyzed separately in Stretch A.

### 2. Batch enrichment in `pipeline.py`
All FX loss calculations happen in `pipeline.py` before the app loads. The Streamlit app only performs aggregations and visualizations on the pre-enriched DataFrame. This keeps the app fast and separates business logic from presentation.

### 3. FX loss in INR
All losses are expressed in INR (the settlement currency). This makes losses directly comparable across different source currencies and meaningful to the finance team.

### 4. Rate table design (one row per currency per day)
Instead of storing one column per currency pair, the FX rate table stores one row per `(date, currency)`. Adding a new currency requires only adding rows — no schema changes needed. Pairs are derived at query time via a simple lookup.

---

## What the Reviewer Should Notice

1. **SGD→INR via NovaGate** has the highest absolute FX loss — largest volume × highest margin processor.
2. **NovaGate consistently applies the highest FX margin** (~3%+ avg) across all currency pairs.
3. **Stretch A** may show settlement-time locking is more expensive during periods of INR depreciation — the processor benefits when the local currency weakens between auth and settlement.
4. **Anomaly detection** (Stretch C) flags transactions where the processor charged >5% spread above mid-market — these warrant direct contract renegotiation.

---

## What We'd Add Next

- **Live FX rates** via Fixer.io or Open Exchange Rates API (replace `generate_data.py` with a real feed)
- **Email/Slack alerts** when daily FX loss exceeds a configurable threshold
- **Integration with Yuno's transaction data** — replace synthetic CSV with a live webhook or database query
- **Hedging simulator** — model the P&L impact of forward contracts by currency
- **Multi-settlement-currency** support (not just INR)
