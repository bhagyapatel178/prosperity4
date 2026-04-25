import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_official_log(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        obj = json.load(f)

    activities = obj.get("activitiesLog")
    if not activities:
        raise ValueError("No activitiesLog found in JSON file")

    df = pd.read_csv(StringIO(activities), sep=";")

    numeric_cols = [
        "day",
        "timestamp",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
        "mid_price",
        "profit_and_loss",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summarise(df: pd.DataFrame):
    df = df.copy()
    df["valid_book"] = df["bid_price_1"].notna() & df["ask_price_1"].notna()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    print("\n=== BASIC SUMMARY ===")
    print("Products:", df["product"].unique())

    for product in df["product"].unique():
        pdf = df[df["product"] == product].sort_values("timestamp").copy()

        final_pnl = pdf["profit_and_loss"].dropna().iloc[-1]
        nonzero = pdf[pdf["profit_and_loss"].fillna(0) != 0]
        first_nonzero_ts = None if nonzero.empty else int(nonzero["timestamp"].iloc[0])

        valid_rows = pdf["valid_book"].sum()
        total_rows = len(pdf)
        pct_valid = 100 * valid_rows / total_rows if total_rows else 0

        print(f"\nProduct: {product}")
        print(f"  final pnl: {final_pnl:.4f}")
        print(f"  first nonzero pnl timestamp: {first_nonzero_ts}")
        print(f"  valid top-of-book rows: {valid_rows}/{total_rows} ({pct_valid:.2f}%)")

        if pdf["spread"].notna().any():
            print(f"  avg spread: {pdf['spread'].mean():.4f}")
            print(f"  median spread: {pdf['spread'].median():.4f}")

        print("  mid-price summary:")
        print(pdf["mid_price"].describe().to_string())

    # final pnl by product
    finals = (
        df.sort_values(["product", "timestamp"])
          .groupby("product", as_index=False)
          .tail(1)[["product", "profit_and_loss"]]
          .rename(columns={"profit_and_loss": "final_pnl"})
    )

    print("\n=== FINAL PNL BY PRODUCT ===")
    print(finals.to_string(index=False))


def plot_pnl(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    for product in df["product"].unique():
        pdf = df[df["product"] == product].sort_values("timestamp")
        plt.plot(pdf["timestamp"], pdf["profit_and_loss"], label=product)
    plt.title("Official Run - PnL by Product")
    plt.xlabel("timestamp")
    plt.ylabel("profit_and_loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mid(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    for product in df["product"].unique():
        pdf = df[df["product"] == product].sort_values("timestamp")
        plt.plot(pdf["timestamp"], pdf["mid_price"], label=product)
    plt.title("Official Run - Mid Price by Product")
    plt.xlabel("timestamp")
    plt.ylabel("mid_price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python research/analyse_official_log.py <path_to_downloaded_json_log>")
        sys.exit(1)

    path = sys.argv[1]
    df = load_official_log(path)
    summarise(df)
    plot_pnl(df)
    plot_mid(df)


if __name__ == "__main__":
    main()