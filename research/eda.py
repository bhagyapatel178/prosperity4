import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_FILE = "prices_round_1_day_-2.csv"   # change if needed

df = pd.read_csv(DATA_DIR / PRICE_FILE, sep=";")

for col in ["bid_price_1", "ask_price_1", "mid_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

products = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]

for product in products:
    pdf = df[df["product"] == product].copy()

    # valid top-of-book rows only
    pdf["valid_book"] = pdf["bid_price_1"].notna() & pdf["ask_price_1"].notna()

    # rebuild mid from best bid/ask when both exist
    pdf["mid_clean"] = np.where(
        pdf["valid_book"],
        (pdf["bid_price_1"] + pdf["ask_price_1"]) / 2,
        np.nan
    )

    pdf["spread"] = np.where(
        pdf["valid_book"],
        pdf["ask_price_1"] - pdf["bid_price_1"],
        np.nan
    )

    # for plotting smooth behaviour
    pdf["mid_ffill"] = pd.Series(pdf["mid_clean"]).ffill()
    pdf["ma_20"] = pdf["mid_ffill"].rolling(20).mean()
    pdf["ma_100"] = pdf["mid_ffill"].rolling(100).mean()
    pdf["ret_1"] = pdf["mid_ffill"].diff()
    pdf["rolling_std_20"] = pdf["ret_1"].rolling(20).std()

    clean = pdf[pdf["valid_book"]].copy()

    print(f"\n===== {product} CLEANED =====")
    print(clean[["mid_clean", "spread"]].describe())

    # zoomed mid plot
    plt.figure(figsize=(12, 5))
    plt.plot(pdf["timestamp"], pdf["mid_ffill"], label="mid_ffill")
    plt.plot(pdf["timestamp"], pdf["ma_20"], label="ma_20")
    plt.plot(pdf["timestamp"], pdf["ma_100"], label="ma_100")
    plt.title(f"{product} - Cleaned Mid Price")
    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # spread plot
    plt.figure(figsize=(12, 4))
    plt.plot(clean["timestamp"], clean["spread"], label="spread")
    plt.title(f"{product} - Cleaned Spread")
    plt.xlabel("timestamp")
    plt.ylabel("spread")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # returns volatility
    plt.figure(figsize=(12, 4))
    plt.plot(pdf["timestamp"], pdf["rolling_std_20"], label="rolling std of returns")
    plt.title(f"{product} - Rolling Volatility of Returns")
    plt.xlabel("timestamp")
    plt.ylabel("std")
    plt.legend()
    plt.tight_layout()
    plt.show()