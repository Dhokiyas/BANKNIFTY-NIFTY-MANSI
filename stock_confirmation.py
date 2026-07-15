from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

def analyse_stock(
    dataframe: pd.DataFrame,
    selected_date,
) -> dict[str, Any]:
    """Stock ke 5-minute candles se direction calculate karta hai."""

    df = dataframe.copy()

    if df.empty:
        raise ValueError("Stock candle data empty hai.")

    df = df.sort_values("Datetime").reset_index(drop=True)

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["SessionDate"] = df["Datetime"].dt.date

    current_df = df[
        df["SessionDate"] == selected_date
    ].copy()

    if current_df.empty:
        raise ValueError(
            "Selected date ke liye stock candles nahi mili."
        )

    current_df["TypicalPrice"] = (
        current_df["High"]
        + current_df["Low"]
        + current_df["Close"]
    ) / 3

    current_df["PriceVolume"] = (
        current_df["TypicalPrice"]
        * current_df["Volume"]
    )

    cumulative_volume = current_df["Volume"].cumsum()

    current_df["VWAP"] = (
        current_df["PriceVolume"].cumsum()
        / cumulative_volume.replace(0, np.nan)
    )

    latest = current_df.iloc[-1]

    current_price = float(latest["Close"])
    day_open = float(current_df.iloc[0]["Open"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])

    latest_vwap = latest["VWAP"]

    if pd.isna(latest_vwap):
        latest_vwap = float(current_df["Close"].mean())
    else:
        latest_vwap = float(latest_vwap)

    recent_closes = current_df["Close"].tail(4)

    if len(recent_closes) >= 2:
        recent_change_percentage = (
            (
                float(recent_closes.iloc[-1])
                / float(recent_closes.iloc[0])
            )
            - 1
        ) * 100
    else:
        recent_change_percentage = 0.0

    day_change_percentage = (
        (current_price / day_open) - 1
    ) * 100

    score = 0
    reasons: list[str] = []

    if current_price > latest_vwap:
        score += 1
        reasons.append("Price above VWAP")
    elif current_price < latest_vwap:
        score -= 1
        reasons.append("Price below VWAP")

    if ema20 > ema50:
        score += 1
        reasons.append("EMA20 above EMA50")
    elif ema20 < ema50:
        score -= 1
        reasons.append("EMA20 below EMA50")

    if current_price > day_open:
        score += 1
        reasons.append("Price above day open")
    elif current_price < day_open:
        score -= 1
        reasons.append("Price below day open")

    if recent_change_percentage > 0.05:
        score += 1
        reasons.append("Recent momentum positive")
    elif recent_change_percentage < -0.05:
        score -= 1
        reasons.append("Recent momentum negative")

    if score >= 2:
        signal = "Bullish"
    elif score <= -2:
        signal = "Bearish"
    else:
        signal = "Sideways"

    return {
        "current_price": current_price,
        "day_open": day_open,
        "day_change_percentage": day_change_percentage,
        "recent_change_percentage": recent_change_percentage,
        "vwap": latest_vwap,
        "ema20": ema20,
        "ema50": ema50,
        "score": score,
        "signal": signal,
        "reason": ", ".join(reasons),
    }

def calculate_majority(
    stock_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_stocks = len(stock_results)

    bullish_count = sum(
        result["signal"] == "Bullish"
        for result in stock_results
    )

    bearish_count = sum(
        result["signal"] == "Bearish"
        for result in stock_results
    )

    sideways_count = sum(
        result["signal"] == "Sideways"
        for result in stock_results
    )

    if bullish_count > total_stocks / 2:
        stock_view = "Bullish"
    elif bearish_count > total_stocks / 2:
        stock_view = "Bearish"
    else:
        stock_view = "Sideways"

    return {
        "stock_view": stock_view,
        "total": total_stocks,
        "bullish": bullish_count,
        "bearish": bearish_count,
        "sideways": sideways_count,
        "bullish_percentage": (
            bullish_count / total_stocks
        ) * 100,
        "bearish_percentage": (
            bearish_count / total_stocks
        ) * 100,
        "sideways_percentage": (
            sideways_count / total_stocks
        ) * 100,
    }

def combine_final_sentiment(
    chart_view: str,
    stock_view: str,
) -> str:
    sentiment_matrix = {
        ("Bullish", "Bullish"): "Bullish",
        ("Bullish", "Sideways"): "Neutral Bullish",
        ("Bullish", "Bearish"): "Sideways",
        ("Sideways", "Bullish"): "Neutral Bullish",
        ("Sideways", "Sideways"): "Sideways",
        ("Sideways", "Bearish"): "Neutral Bearish",
        ("Bearish", "Bullish"): "Sideways",
        ("Bearish", "Sideways"): "Neutral Bearish",
        ("Bearish", "Bearish"): "Bearish",
    }

    return sentiment_matrix.get(
        (chart_view, stock_view),
        "Sideways",
    )
