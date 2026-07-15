from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=1,
    ).mean()


def find_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> tuple[list[float], list[float]]:
    swing_highs: list[float] = []
    swing_lows: list[float] = []

    if len(df) < left + right + 1:
        return swing_highs, swing_lows

    for index in range(left, len(df) - right):
        current_high = float(df.iloc[index]["High"])
        current_low = float(df.iloc[index]["Low"])

        nearby_highs = df.iloc[
            index - left : index + right + 1
        ]["High"]

        nearby_lows = df.iloc[
            index - left : index + right + 1
        ]["Low"]

        if current_high >= float(nearby_highs.max()):
            swing_highs.append(current_high)

        if current_low <= float(nearby_lows.min()):
            swing_lows.append(current_low)

    return swing_highs, swing_lows


def cluster_levels(
    levels: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    if not levels:
        return []

    sorted_levels = sorted(
        levels,
        key=lambda item: float(item["price"]),
    )

    clusters: list[list[dict[str, Any]]] = []

    for level in sorted_levels:
        if not clusters:
            clusters.append([level])
            continue

        latest_cluster = clusters[-1]

        cluster_average = float(
            np.average(
                [item["price"] for item in latest_cluster],
                weights=[
                    item.get("weight", 1)
                    for item in latest_cluster
                ],
            )
        )

        if abs(float(level["price"]) - cluster_average) <= tolerance:
            latest_cluster.append(level)
        else:
            clusters.append([level])

    result: list[dict[str, Any]] = []

    for cluster in clusters:
        prices = [float(item["price"]) for item in cluster]
        weights = [
            float(item.get("weight", 1))
            for item in cluster
        ]

        result.append(
            {
                "price": float(
                    np.average(prices, weights=weights)
                ),
                "strength": float(sum(weights)),
                "sources": list(
                    dict.fromkeys(
                        item["source"] for item in cluster
                    )
                ),
            }
        )

    return result


def analyse_index(
    dataframe: pd.DataFrame,
    selected_date,
) -> dict[str, Any]:
    df = dataframe.copy()

    if df.empty:
        raise ValueError("Analysis ke liye candle data empty hai.")

    df = df.sort_values("Datetime").reset_index(drop=True)

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ATR"] = calculate_atr(df)

    df["SessionDate"] = df["Datetime"].dt.date

    current_df = df[
        df["SessionDate"] == selected_date
    ].copy()

    if current_df.empty:
        raise ValueError(
            "Selected date ke liye index candles nahi mili."
        )

    previous_dates = sorted(
        date_value
        for date_value in df["SessionDate"].unique()
        if date_value < selected_date
    )

    previous_df = pd.DataFrame()

    if previous_dates:
        previous_date = previous_dates[-1]

        previous_df = df[
            df["SessionDate"] == previous_date
        ].copy()

    latest = current_df.iloc[-1]

    current_price = float(latest["Close"])
    day_open = float(current_df.iloc[0]["Open"])
    day_high = float(current_df["High"].max())
    day_low = float(current_df["Low"].min())

    atr = float(latest["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        atr = max(current_price * 0.002, 1.0)

    opening_range = current_df.head(3)

    opening_high = float(opening_range["High"].max())
    opening_low = float(opening_range["Low"].min())

    swing_highs, swing_lows = find_swings(current_df)

    candidates: list[dict[str, Any]] = [
        {
            "price": day_high,
            "source": "Current Day High",
            "weight": 2.0,
        },
        {
            "price": day_low,
            "source": "Current Day Low",
            "weight": 2.0,
        },
        {
            "price": opening_high,
            "source": "Opening Range High",
            "weight": 2.0,
        },
        {
            "price": opening_low,
            "source": "Opening Range Low",
            "weight": 2.0,
        },
    ]

    if not previous_df.empty:
        candidates.extend(
            [
                {
                    "price": float(previous_df["High"].max()),
                    "source": "Previous Day High",
                    "weight": 3.0,
                },
                {
                    "price": float(previous_df["Low"].min()),
                    "source": "Previous Day Low",
                    "weight": 3.0,
                },
                {
                    "price": float(
                        previous_df.iloc[-1]["Close"]
                    ),
                    "source": "Previous Day Close",
                    "weight": 2.0,
                },
            ]
        )

    for swing_high in swing_highs:
        candidates.append(
            {
                "price": swing_high,
                "source": "Swing High",
                "weight": 1.5,
            }
        )

    for swing_low in swing_lows:
        candidates.append(
            {
                "price": swing_low,
                "source": "Swing Low",
                "weight": 1.5,
            }
        )

    tolerance = max(
        atr * 0.25,
        current_price * 0.0005,
    )

    clustered_levels = cluster_levels(
        candidates,
        tolerance,
    )

    minimum_distance = max(
        atr * 0.08,
        current_price * 0.0001,
    )

    supports = sorted(
        [
            level
            for level in clustered_levels
            if level["price"] < current_price - minimum_distance
        ],
        key=lambda level: level["price"],
        reverse=True,
    )

    resistances = sorted(
        [
            level
            for level in clustered_levels
            if level["price"] > current_price + minimum_distance
        ],
        key=lambda level: level["price"],
    )

    while len(supports) < 3:
        fallback_number = len(supports) + 1

        supports.append(
            {
                "price": current_price - atr * fallback_number,
                "strength": 0,
                "sources": ["ATR Projection"],
            }
        )

    while len(resistances) < 3:
        fallback_number = len(resistances) + 1

        resistances.append(
            {
                "price": current_price + atr * fallback_number,
                "strength": 0,
                "sources": ["ATR Projection"],
            }
        )

    supports = supports[:3]
    resistances = resistances[:3]

    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])

    trend_score = 0

    if current_price > ema20:
        trend_score += 1
    else:
        trend_score -= 1

    if ema20 > ema50:
        trend_score += 1
    else:
        trend_score -= 1

    if current_price > day_open:
        trend_score += 1
    else:
        trend_score -= 1

    recent_closes = current_df["Close"].tail(5)

    if len(recent_closes) >= 3:
        close_slope = float(
            np.polyfit(
                range(len(recent_closes)),
                recent_closes,
                1,
            )[0]
        )

        if close_slope > atr * 0.03:
            trend_score += 1
        elif close_slope < -(atr * 0.03):
            trend_score -= 1

    recent_high = float(
        current_df["High"].tail(5).max()
    )

    recent_low = float(
        current_df["Low"].tail(5).min()
    )

    recent_position = (
        (current_price - recent_low)
        / max(recent_high - recent_low, 0.01)
    )

    if recent_position >= 0.70:
        trend_score += 1
    elif recent_position <= 0.30:
        trend_score -= 1

    if trend_score >= 2:
        chart_view = "Bullish"
    elif trend_score <= -2:
        chart_view = "Bearish"
    else:
        chart_view = "Sideways"

    resistance_1 = float(resistances[0]["price"])
    resistance_2 = float(resistances[1]["price"])
    resistance_3 = float(resistances[2]["price"])

    support_1 = float(supports[0]["price"])
    support_2 = float(supports[1]["price"])
    support_3 = float(supports[2]["price"])

    bullish_minimum_target = max(
        resistance_2,
        resistance_1 + atr * 0.75,
    )

    bullish_maximum_target = max(
        resistance_3,
        resistance_1 + atr * 1.75,
    )

    bearish_minimum_target = min(
        support_2,
        support_1 - atr * 0.75,
    )

    bearish_maximum_target = min(
        support_3,
        support_1 - atr * 1.75,
    )

    return {
        "current_price": current_price,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "ema20": ema20,
        "ema50": ema50,
        "atr": atr,
        "chart_score": trend_score,
        "chart_view": chart_view,
        "support_1": support_1,
        "support_2": support_2,
        "support_3": support_3,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "resistance_3": resistance_3,
        "bullish_trigger": resistance_1,
        "bullish_minimum_target": bullish_minimum_target,
        "bullish_maximum_target": bullish_maximum_target,
        "bearish_trigger": support_1,
        "bearish_minimum_target": bearish_minimum_target,
        "bearish_maximum_target": bearish_maximum_target,
        "support_sources": supports[0]["sources"],
        "resistance_sources": resistances[0]["sources"],
    }
