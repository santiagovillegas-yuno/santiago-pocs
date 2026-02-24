"""
FX Loss Calculation Pipeline for Kalari Fashion.

Main entry point: enrich_transactions(df_transactions, df_fx_rates) -> DataFrame

Adds columns:
  mid_market_rate          — mid-market rate on auth date (INR per 1 unit of presentment_currency)
  expected_inr             — presentment_amount * mid_market_rate (fair value)
  fx_loss_inr              — expected_inr - settlement_amount (loss due to processor spread)
  fx_margin_pct            — (fx_loss_inr / expected_inr) * 100 (processor markup %)
  settlement_rate          — mid-market rate on settlement_date (for Stretch A)
  settlement_time_loss_inr — loss if rate were locked at settlement time (Stretch A)
  is_anomaly               — True if fx_margin_pct > 5% (Stretch C)
  currency_pair            — e.g. "SGD→INR"
"""

import pandas as pd


ANOMALY_THRESHOLD_PCT = 5.0


def enrich_transactions(
    df_transactions: pd.DataFrame,
    df_fx_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge mid-market rates into transactions and compute all FX loss metrics.

    Parameters
    ----------
    df_transactions : DataFrame
        Raw transactions from transactions.csv.
    df_fx_rates : DataFrame
        Daily rates from fx_rates.csv (one row per currency per day).

    Returns
    -------
    DataFrame with additional FX loss columns (original rows preserved).
    """
    df = df_transactions.copy()

    # Ensure datetime types
    df["auth_timestamp"]  = pd.to_datetime(df["auth_timestamp"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

    df["auth_date"] = df["auth_timestamp"].dt.date

    # Build lookup: (date, currency) -> rate_vs_inr
    rates = df_fx_rates.copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.date
    rate_lookup = rates.set_index(["date", "currency"])["rate_vs_inr"]

    # ---------------------------------------------------------------------------
    # Core FX loss calculation
    # ---------------------------------------------------------------------------
    def compute_row(row):
        currency = row["presentment_currency"]

        # Edge case: same currency (INR→INR)
        if currency == "INR":
            return pd.Series({
                "mid_market_rate":          1.0,
                "expected_inr":             row["presentment_amount"],
                "fx_loss_inr":              0.0,
                "fx_margin_pct":            0.0,
                "settlement_rate":          1.0,
                "settlement_time_loss_inr": 0.0,
            })

        # Auth-date mid-market rate
        auth_rate = rate_lookup.get((row["auth_date"], currency))
        if auth_rate is None:
            # Fallback: use nearest available rate (shouldn't happen in generated data)
            auth_rate = float("nan")

        expected_inr = row["presentment_amount"] * auth_rate if pd.notna(auth_rate) else float("nan")
        fx_loss_inr  = expected_inr - row["settlement_amount"] if pd.notna(expected_inr) else float("nan")
        fx_margin_pct = (fx_loss_inr / expected_inr * 100) if (pd.notna(expected_inr) and expected_inr != 0) else float("nan")

        # Stretch A: settlement-date rate
        settlement_rate = rate_lookup.get((row["settlement_date"], currency))
        if settlement_rate is None:
            settlement_rate = auth_rate  # fallback to auth rate

        settlement_expected    = row["presentment_amount"] * settlement_rate
        settlement_time_loss   = settlement_expected - row["settlement_amount"]

        return pd.Series({
            "mid_market_rate":          round(auth_rate, 5) if pd.notna(auth_rate) else float("nan"),
            "expected_inr":             round(expected_inr, 2) if pd.notna(expected_inr) else float("nan"),
            "fx_loss_inr":              round(fx_loss_inr, 2) if pd.notna(fx_loss_inr) else float("nan"),
            "fx_margin_pct":            round(fx_margin_pct, 4) if pd.notna(fx_margin_pct) else float("nan"),
            "settlement_rate":          round(settlement_rate, 5),
            "settlement_time_loss_inr": round(settlement_time_loss, 2),
        })

    computed = df.apply(compute_row, axis=1)
    df = pd.concat([df, computed], axis=1)

    # Derived columns
    df["currency_pair"] = df["presentment_currency"] + "→INR"
    df["is_anomaly"]    = df["fx_margin_pct"] > ANOMALY_THRESHOLD_PCT

    # Drop helper column
    df.drop(columns=["auth_date"], inplace=True)

    return df


def load_and_enrich(
    transactions_path: str = "data/transactions.csv",
    fx_rates_path: str = "data/fx_rates.csv",
) -> pd.DataFrame:
    """Convenience loader: read CSVs and return enriched DataFrame."""
    df_txns  = pd.read_csv(transactions_path)
    df_rates = pd.read_csv(fx_rates_path)
    return enrich_transactions(df_txns, df_rates)


if __name__ == "__main__":
    df = load_and_enrich()
    print(f"Enriched {len(df)} transactions")
    print(f"Total FX loss (INR): ₹{df['fx_loss_inr'].sum():,.0f}")
    print(f"Avg FX margin:       {df['fx_margin_pct'].mean():.2f}%")
    print(f"Anomalies detected:  {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print("\nFX loss by processor:")
    print(df.groupby("processor")["fx_margin_pct"].mean().round(2).to_string())
