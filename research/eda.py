import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Change this to your actual filename
PRICE_FILE = "prices_round_1_day_-1.csv"

df = pd.read_csv(DATA_DIR / PRICE_FILE, sep=";")

print("Columns:")
print(df.columns.tolist())
print("\nProducts:")
print(df["product"].unique())

products = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]

for product in products:
    pdf = df[df["product"] == product].copy()

    if pdf.empty:
        print(f"\nNo data found for {product}")
        continue

    # Spread from best bid / ask
    pdf["spread"] = pdf["ask_price_1"] - pdf["bid_price_1"]

    # Rolling behaviour
    pdf["mid_ma_20"] = pdf["mid_price"].rolling(20).mean()
    pdf["mid_ma_100"] = pdf["mid_price"].rolling(100).mean()
    pdf["mid_std_20"] = pdf["mid_price"].rolling(20).std()

    print(f"\n===== {product} =====")
    print(pdf[["timestamp", "bid_price_1", "ask_price_1", "mid_price", "spread"]].head())
    print("\nSummary stats:")
    print(pdf[["mid_price", "spread"]].describe())

    # Mid price + moving averages
    plt.figure(figsize=(12, 5))
    plt.plot(pdf["timestamp"], pdf["mid_price"], label="mid_price")
    plt.plot(pdf["timestamp"], pdf["mid_ma_20"], label="ma_20")
    plt.plot(pdf["timestamp"], pdf["mid_ma_100"], label="ma_100")
    plt.title(f"{product} - Mid Price")
    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Spread
    plt.figure(figsize=(12, 4))
    plt.plot(pdf["timestamp"], pdf["spread"], label="spread")
    plt.title(f"{product} - Spread")
    plt.xlabel("timestamp")
    plt.ylabel("spread")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Short-term volatility proxy
    plt.figure(figsize=(12, 4))
    plt.plot(pdf["timestamp"], pdf["mid_std_20"], label="rolling std 20")
    plt.title(f"{product} - Rolling Volatility")
    plt.xlabel("timestamp")
    plt.ylabel("std")
    plt.legend()
    plt.tight_layout()
    plt.show()