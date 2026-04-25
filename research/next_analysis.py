import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRICE_FILES = sorted(DATA_DIR.glob("prices_round_1_day_*.csv"))

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"


def load_prices() -> pd.DataFrame:
    dfs = []
    for file in PRICE_FILES:
        df = pd.read_csv(file, sep=";")
        df["source_file"] = file.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No price files found in data/")

    df = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        "day",
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # valid top of book only
    df["valid_book"] = df["bid_price_1"].notna() & df["ask_price_1"].notna()
    df["mid_clean"] = np.where(
        df["valid_book"],
        (df["bid_price_1"] + df["ask_price_1"]) / 2,
        np.nan,
    )
    df["spread"] = np.where(
        df["valid_book"],
        df["ask_price_1"] - df["bid_price_1"],
        np.nan,
    )

    return df


def fit_linear_trend(x: np.ndarray, y: np.ndarray):
    """
    Fit y = slope * x + intercept
    Returns slope, intercept, y_hat, residuals, r2, rmse
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    residuals = y - y_hat

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals ** 2))

    return slope, intercept, y_hat, residuals, r2, rmse


def analyse_pepper(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("PEPPER ROOT TREND ANALYSIS")
    print("=" * 80)

    pepper = df[(df["product"] == PEPPER) & (df["valid_book"])].copy()
    pepper = pepper.sort_values(["day", "timestamp"])

    summary_rows = []

    for day, day_df in pepper.groupby("day"):
        day_df = day_df.dropna(subset=["timestamp", "mid_clean"]).copy()
        x = day_df["timestamp"].to_numpy(dtype=float)
        y = day_df["mid_clean"].to_numpy(dtype=float)

        slope, intercept, y_hat, residuals, r2, rmse = fit_linear_trend(x, y)

        summary_rows.append(
            {
                "day": int(day),
                "n_obs": len(day_df),
                "slope_per_timestamp": slope,
                "intercept": intercept,
                "r2": r2,
                "rmse": rmse,
                "residual_std": np.std(residuals),
                "residual_mean": np.mean(residuals),
                "spread_mean": day_df["spread"].mean(),
                "spread_median": day_df["spread"].median(),
            }
        )

        print(f"\nDay {int(day)}")
        print(f"  slope per timestamp: {slope:.6f}")
        print(f"  intercept:          {intercept:.3f}")
        print(f"  R^2:                {r2:.6f}")
        print(f"  RMSE:               {rmse:.6f}")
        print(f"  residual std:       {np.std(residuals):.6f}")
        print(f"  avg spread:         {day_df['spread'].mean():.6f}")

        # Plot actual vs fitted trend
        plt.figure(figsize=(12, 5))
        plt.plot(day_df["timestamp"], day_df["mid_clean"], label="mid_clean")
        plt.plot(day_df["timestamp"], y_hat, label="linear_fit")
        plt.title(f"{PEPPER} - Day {int(day)}: Actual vs Linear Trend")
        plt.xlabel("timestamp")
        plt.ylabel("price")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot residuals
        plt.figure(figsize=(12, 4))
        plt.plot(day_df["timestamp"], residuals, label="residual")
        plt.axhline(0, linestyle="--")
        plt.title(f"{PEPPER} - Day {int(day)}: Residuals Around Trend")
        plt.xlabel("timestamp")
        plt.ylabel("actual - fitted")
        plt.legend()
        plt.tight_layout()
        plt.show()

    summary_df = pd.DataFrame(summary_rows).sort_values("day")
    print("\nPepper trend summary:")
    print(summary_df.to_string(index=False))


def analyse_osmium(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("OSMIUM SHORT-TERM BEHAVIOUR ANALYSIS")
    print("=" * 80)

    osmium = df[(df["product"] == OSMIUM) & (df["valid_book"])].copy()
    osmium = osmium.sort_values(["day", "timestamp"])

    summary_rows = []

    for day, day_df in osmium.groupby("day"):
        day_df = day_df.dropna(subset=["timestamp", "mid_clean"]).copy()
        day_df["ret_1"] = day_df["mid_clean"].diff()
        day_df["next_ret_1"] = day_df["ret_1"].shift(-1)

        valid = day_df.dropna(subset=["ret_1", "next_ret_1"]).copy()

        if len(valid) < 2:
            print(f"\nDay {int(day)}: not enough valid return data")
            continue

        # Lag-1 autocorrelation of returns
        lag1_autocorr = valid["ret_1"].corr(valid["next_ret_1"])

        up_now = valid[valid["ret_1"] > 0]
        down_now = valid[valid["ret_1"] < 0]
        flat_now = valid[valid["ret_1"] == 0]

        avg_next_after_up = up_now["next_ret_1"].mean() if not up_now.empty else np.nan
        avg_next_after_down = down_now["next_ret_1"].mean() if not down_now.empty else np.nan
        avg_next_after_flat = flat_now["next_ret_1"].mean() if not flat_now.empty else np.nan

        # Sign continuation / reversal
        nonzero = valid[(valid["ret_1"] != 0) & (valid["next_ret_1"] != 0)].copy()
        if not nonzero.empty:
            same_sign = np.mean(np.sign(nonzero["ret_1"]) == np.sign(nonzero["next_ret_1"]))
            opposite_sign = np.mean(np.sign(nonzero["ret_1"]) == -np.sign(nonzero["next_ret_1"]))
        else:
            same_sign = np.nan
            opposite_sign = np.nan

        summary_rows.append(
            {
                "day": int(day),
                "n_obs": len(valid),
                "lag1_return_autocorr": lag1_autocorr,
                "avg_next_ret_after_up": avg_next_after_up,
                "avg_next_ret_after_down": avg_next_after_down,
                "avg_next_ret_after_flat": avg_next_after_flat,
                "same_sign_rate": same_sign,
                "opposite_sign_rate": opposite_sign,
                "mid_std": day_df["mid_clean"].std(),
                "spread_mean": day_df["spread"].mean(),
            }
        )

        print(f"\nDay {int(day)}")
        print(f"  lag-1 return autocorr:     {lag1_autocorr:.6f}")
        print(f"  avg next ret after UP:     {avg_next_after_up:.6f}")
        print(f"  avg next ret after DOWN:   {avg_next_after_down:.6f}")
        print(f"  avg next ret after FLAT:   {avg_next_after_flat:.6f}")
        print(f"  same sign rate:            {same_sign:.6f}")
        print(f"  opposite sign rate:        {opposite_sign:.6f}")
        print(f"  avg spread:                {day_df['spread'].mean():.6f}")

        # Plot price
        plt.figure(figsize=(12, 5))
        plt.plot(day_df["timestamp"], day_df["mid_clean"], label="mid_clean")
        plt.title(f"{OSMIUM} - Day {int(day)} Mid Price")
        plt.xlabel("timestamp")
        plt.ylabel("price")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot returns
        plt.figure(figsize=(12, 4))
        plt.plot(valid["timestamp"], valid["ret_1"], label="ret_1")
        plt.axhline(0, linestyle="--")
        plt.title(f"{OSMIUM} - Day {int(day)} One-Step Returns")
        plt.xlabel("timestamp")
        plt.ylabel("return")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Scatter return_t vs return_t+1
        plt.figure(figsize=(6, 6))
        plt.scatter(valid["ret_1"], valid["next_ret_1"], alpha=0.4)
        plt.axhline(0, linestyle="--")
        plt.axvline(0, linestyle="--")
        plt.title(f"{OSMIUM} - Day {int(day)} Return_t vs Return_t+1")
        plt.xlabel("ret_t")
        plt.ylabel("ret_t+1")
        plt.tight_layout()
        plt.show()

    summary_df = pd.DataFrame(summary_rows).sort_values("day")
    print("\nOsmium short-term summary:")
    print(summary_df.to_string(index=False))


def main():
    df = load_prices()

    print("Loaded files:")
    for f in PRICE_FILES:
        print(f" - {f.name}")

    analyse_pepper(df)
    analyse_osmium(df)


if __name__ == "__main__":
    main()