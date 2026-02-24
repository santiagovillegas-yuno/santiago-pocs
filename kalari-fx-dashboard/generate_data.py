"""
Generate synthetic FX rates and transaction data for Kalari Fashion FX Bleed Dashboard.

Outputs:
  data/fx_rates.csv       — daily mid-market rates per currency vs INR
  data/transactions.csv   — 300+ payment transactions with settlement details
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_RATES_INR = {
    "SGD": 62.45,
    "AED": 22.83,
    "MYR": 18.12,
    "KES": 0.62,
    "TZS": 0.039,
    "USD": 83.10,
}

CURRENCY_WEIGHTS = {
    "SGD": 0.40,
    "AED": 0.20,
    "MYR": 0.15,
    "KES": 0.10,
    "TZS": 0.10,
    "USD": 0.05,
}

PROCESSORS = {
    "NovaGate":       (2.5, 3.5),
    "PayBridge Asia": (1.5, 2.2),
    "SwiftAcquire":   (1.8, 2.8),
}

# Amount ranges per currency (realistic fashion e-commerce order values in that currency)
AMOUNT_RANGES = {
    "SGD": (80, 600),
    "AED": (150, 1200),
    "MYR": (200, 1800),
    "KES": (5000, 60000),
    "TZS": (80000, 800000),
    "USD": (50, 500),
}

NUM_TRANSACTIONS   = 335   # normal transactions
NUM_ANOMALIES      = 15    # extra high-markup transactions for Stretch C demo
START_DATE = pd.Timestamp("2024-11-01")
END_DATE   = pd.Timestamp("2025-01-31")


# ---------------------------------------------------------------------------
# 1. Generate FX rates
# ---------------------------------------------------------------------------

def generate_fx_rates() -> pd.DataFrame:
    """Daily mid-market rates for each currency vs INR over the full date range."""
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []
    for currency, base_rate in BASE_RATES_INR.items():
        rate = base_rate
        for date in date_range:
            # Small Gaussian daily walk (±0.3%)
            change = np.random.normal(0, 0.003)
            rate = rate * (1 + change)
            rows.append({"date": date.date(), "currency": currency, "rate_vs_inr": round(rate, 5)})

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 2. Generate transactions
# ---------------------------------------------------------------------------

def generate_transactions(df_rates: pd.DataFrame) -> pd.DataFrame:
    # Build a quick lookup: (date, currency) -> rate
    rate_lookup = df_rates.set_index(["date", "currency"])["rate_vs_inr"].to_dict()

    currencies = list(CURRENCY_WEIGHTS.keys())
    weights    = list(CURRENCY_WEIGHTS.values())
    processor_names = list(PROCESSORS.keys())

    rows = []
    total_to_generate = NUM_TRANSACTIONS + NUM_ANOMALIES
    for i in range(total_to_generate):
        is_anomaly_txn = i >= NUM_TRANSACTIONS  # last N are forced anomalies
        txn_id    = f"TXN{10000 + i}"
        currency  = random.choices(currencies, weights=weights, k=1)[0]
        processor = random.choice(processor_names)

        # Auth timestamp: random day in range, random time
        auth_date = START_DATE + pd.Timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
        auth_ts   = auth_date + pd.Timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        # Settlement delay: mostly 1-5 days, ~5% at 7-10 days
        if random.random() < 0.05:
            delay_days = random.randint(7, 10)
        else:
            delay_days = random.randint(1, 5)

        settlement_date = (auth_date + pd.Timedelta(days=delay_days)).date()
        # Cap to our rate table end date
        if settlement_date > END_DATE.date():
            settlement_date = END_DATE.date()

        # Presentment amount
        lo, hi = AMOUNT_RANGES[currency]
        presentment_amount = round(random.uniform(lo, hi), 2)

        # Mid-market rate at auth date
        auth_rate = rate_lookup.get((auth_date.date(), currency))
        if auth_rate is None:
            continue  # skip if out of range (shouldn't happen)

        # Processor applies markup → lower effective rate
        # Anomalous transactions get a high markup (5.5–8%) to trigger Stretch C detection
        if is_anomaly_txn:
            markup_pct = random.uniform(5.5, 8.0)
        else:
            min_markup, max_markup = PROCESSORS[processor]
            markup_pct = random.uniform(min_markup, max_markup)
        applied_rate = auth_rate * (1 - markup_pct / 100)
        settlement_amount = round(presentment_amount * applied_rate, 2)

        rows.append({
            "transaction_id":       txn_id,
            "auth_timestamp":       auth_ts,
            "settlement_date":      settlement_date,
            "presentment_currency": currency,
            "presentment_amount":   presentment_amount,
            "settlement_currency":  "INR",
            "settlement_amount":    settlement_amount,
            "processor":            processor,
            "markup_pct_applied":   round(markup_pct, 4),  # ground truth for validation
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)

    print("Generating FX rates...")
    df_rates = generate_fx_rates()
    df_rates.to_csv("data/fx_rates.csv", index=False)
    print(f"  data/fx_rates.csv — {len(df_rates):,} rows")

    print("Generating transactions...")
    df_txns = generate_transactions(df_rates)
    df_txns.to_csv("data/transactions.csv", index=False)
    print(f"  data/transactions.csv — {len(df_txns):,} rows")

    # Quick sanity check
    print("\nSanity check (first 3 rows of transactions):")
    print(df_txns[["transaction_id", "presentment_currency", "presentment_amount",
                    "settlement_amount", "processor", "markup_pct_applied"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
