
#!/usr/bin/env python
import argparse
import datetime as dt
import os

import pandas as pd
import requests
import yfinance as yf

TODAY = dt.date.today()


def fetch_gold_prices(lookback_years: int) -> pd.DataFrame:
    start = TODAY - dt.timedelta(days=365 * lookback_years + 30)
    ticker = "XAUUSD=X"
    df = yf.download(ticker, start=start, end=TODAY + dt.timedelta(days=1), interval="1d")
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_dxy(lookback_years: int) -> pd.DataFrame:
    start = TODAY - dt.timedelta(days=365 * lookback_years + 30)
    dxy_ticker = "DX-Y.NYB"
    df = yf.download(dxy_ticker, start=start, end=TODAY + dt.timedelta(days=1), interval="1d")
    df = df.rename(columns=lambda c: f"dxy_{c.lower()}")
    df.index = pd.to_datetime(df.index)
    return df


def fetch_real_yield_series() -> pd.DataFrame:
    """Fetch real 10Y yield series from a public CSV provider."""
    url = "https://eco3min.fr/dataset/real-10y-treasury-yield.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    # Ensure there is a column called 'real_10y'; adjust if provider changes schema.
    if "real_10y" not in df.columns:
        raise RuntimeError("Expected column 'real_10y' in real-yield dataset")
    return df[["real_10y"]]


def compute_fundamental_score(df_gold: pd.DataFrame,
                              df_dxy: pd.DataFrame,
                              df_real: pd.DataFrame) -> pd.Series:
    df = pd.concat([
        df_gold[["close"]].rename(columns={"close": "xau_close"}),
        df_dxy[["dxy_close"]],
        df_real[["real_10y"]],
    ], axis=1).dropna()

    real = df["real_10y"]
    real_score = pd.Series(0.0, index=df.index)
    real_score[real <= -0.5] = 2.0
    real_score[(real > -0.5) & (real <= 0.5)] = 1.0
    real_score[(real > 0.5) & (real <= 1.5)] = 0.0
    real_score[real > 1.5] = -1.5

    dxy_ret_20 = df["dxy_close"].pct_change(20)
    dxy_score = pd.Series(0.0, index=df.index)
    dxy_score[dxy_ret_20 < 0] = 1.0
    dxy_score[dxy_ret_20 > 0] = -1.0

    roll_corr = df["xau_close"].rolling(60).corr(df["dxy_close"])
    corr_penalty = roll_corr.abs() < 0.3
    dxy_score[corr_penalty] *= 0.5

    cb_score = pd.Series(0.5, index=df.index)
    growth_score = pd.Series(0.0, index=df.index)
    geo_score = pd.Series(0.0, index=df.index)
    supply_score = pd.Series(0.0, index=df.index)

    F = (
        0.35 * real_score +
        0.25 * dxy_score +
        0.15 * cb_score +
        0.10 * growth_score +
        0.10 * geo_score +
        0.05 * supply_score
    )

    F = F.clip(-2.0, 2.0)
    return F


def compute_technical_and_regime(df_gold: pd.DataFrame) -> pd.DataFrame:
    df = df_gold.copy()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["trend"] = "neutral"
    df.loc[(df["close"] > df["ma200"]) & (df["ma50"] > df["ma200"]), "trend"] = "bull"
    df.loc[(df["close"] < df["ma200"]) & (df["ma50"] < df["ma200"]), "trend"] = "bear"

    atr = df["atr14"]
    rolling = atr.rolling(60)
    pct = rolling.rank(pct=True)
    df["atr_pct_60d"] = pct

    df["vol_regime"] = "range"
    df.loc[df["atr_pct_60d"] >= 0.7, "vol_regime"] = "trend"
    df.loc[(df["atr_pct_60d"] > 0.4) & (df["atr_pct_60d"] < 0.7), "vol_regime"] = "transition"

    return df


def generate_daily_decision_table(df_gold: pd.DataFrame,
                                  F: pd.Series,
                                  tech: pd.DataFrame) -> pd.DataFrame:
    today = df_gold.index.max().normalize()
    # Use last available index at or before today
    idx = df_gold.index.get_loc(today, method="pad") if today in df_gold.index else -1

    F = F.reindex(df_gold.index).ffill()
    tech = tech.reindex(df_gold.index)

    date_idx = df_gold.index[idx]
    df = pd.DataFrame(index=[date_idx])
    df["close"] = df_gold.loc[date_idx, "close"]
    df["fundamental_score"] = F.loc[date_idx]
    df["trend"] = tech.loc[date_idx, "trend"]
    df["vol_regime"] = tech.loc[date_idx, "vol_regime"]
    df["atr14"] = tech.loc[date_idx, "atr14"]

    F_today = df["fundamental_score"].iloc[0]
    trend_today = df["trend"].iloc[0]

    direction = "flat"
    if F_today >= 0.5 and trend_today == "bull":
        direction = "long-only"
    elif F_today <= -0.5 and trend_today == "bear":
        direction = "short-only"

    df["direction_bias"] = direction
    return df


def run_daily_report(lookback_years: int, out_html: str, out_csv: str) -> None:
    gold = fetch_gold_prices(lookback_years)
    dxy = fetch_dxy(lookback_years)
    real = fetch_real_yield_series()

    F = compute_fundamental_score(gold, dxy, real)
    tech = compute_technical_and_regime(gold)
    table = generate_daily_decision_table(gold, F, tech)

    table.to_csv(out_csv, index_label="date")

    html = table.to_html(float_format=lambda x: f"{x:.4f}")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>XAUUSD Daily Strategy Report</h1>")
        f.write(html)
        f.write("</body></html>")


def run_backtest(lookback_years: int, out_html: str, out_csv: str) -> None:
    gold = fetch_gold_prices(lookback_years)
    dxy = fetch_dxy(lookback_years)
    real = fetch_real_yield_series()

    F = compute_fundamental_score(gold, dxy, real)
    tech = compute_technical_and_regime(gold)

    tables = []
    for dt_idx in gold.index[250:]:
        sub_gold = gold.loc[:dt_idx]
        sub_F = F.loc[:dt_idx]
        sub_tech = tech.loc[:dt_idx]
        row = generate_daily_decision_table(sub_gold, sub_F, sub_tech)
        tables.append(row)

    decisions = pd.concat(tables)
    prices = gold["close"].reindex(decisions.index)

    pos = decisions["direction_bias"].map({"long-only": 1.0, "short-only": -1.0, "flat": 0.0}).fillna(0.0)
    ret = prices.pct_change().fillna(0.0)
    decisions["strategy_ret"] = pos.shift(1).fillna(0.0) * ret
    decisions["cum_return"] = (1 + decisions["strategy_ret"]).cumprod() - 1

    decisions.to_csv(out_csv, index_label="date")

    final_cum = decisions["cum_return"].iloc[-1]
    html = decisions.tail(60).to_html(float_format=lambda x: f"{x:.4f}")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>XAUUSD 15Y Backtest Summary</h1>")
        f.write(f"<p>Final cumulative return: {final_cum:.2%}</p>")
        f.write("<h2>Last 60 days of stats</h2>")
        f.write(html)
        f.write("</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily-report", "backtest"], required=True)
    parser.add_argument("--lookback-years", type=int, default=15)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    if args.mode == "daily-report":
        run_daily_report(args.lookback_years, args.out_html, args.out_csv)
    elif args.mode == "backtest":
        run_backtest(args.lookback_years, args.out_html, args.out_csv)
    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
