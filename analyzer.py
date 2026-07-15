from __future__ import annotations

from typing import Any

import math
import numpy as np
import pandas as pd

def calculate_atr(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    previous_close = dataframe["Close"].shift(1)

    true_range = pd.concat(
        [
            dataframe["High"] - dataframe["Low"],
            (
                dataframe["High"] - previous_close
            ).abs(),
            (
                dataframe["Low"] - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=1,
    ).mean()

def round_to_step(
    price: float,
    step: float,
) -> float:
    return round(price / step) * step

def normalize_trigger_level(
    raw_price: float,
    current_price: float,
    level_type: str,
) -> float:
    """
    Final chart level ko manual chart-analysis style mein normalize karta hai.

    Nifty:
      Support    -> lower 10 point
      Resistance -> upper 10 point

    Bank Nifty:
      Support    -> nearest 50 point psychological level
      Resistance -> nearest 10 point actual chart level
    """

    is_bank_nifty = current_price > 40000

    if is_bank_nifty:
        if level_type == "support":
            normalized = round_to_step(
                raw_price,
                50.0,
            )

            if normalized >= current_price:
                normalized -= 50.0

        else:
            normalized = round_to_step(
                raw_price,
                10.0,
            )

            if normalized <= current_price:
                normalized += 10.0

    else:
        if level_type == "support":
            normalized = (
                math.floor(
                    (raw_price + 0.000001) / 10.0
                )
                * 10.0
            )

            if normalized >= current_price:
                normalized -= 10.0

        else:
            normalized = (
                math.ceil(
                    (raw_price - 0.000001) / 10.0
                )
                * 10.0
            )

            if normalized <= current_price:
                normalized += 10.0

    return float(normalized)

def find_swings(
    dataframe: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    swing_highs: list[float] = []
    swing_lows: list[float] = []

    if len(dataframe) < 3:
        return swing_highs, swing_lows

    for index in range(1, len(dataframe) - 1):
        previous_candle = dataframe.iloc[index - 1]
        current_candle = dataframe.iloc[index]
        next_candle = dataframe.iloc[index + 1]

        if (
            current_candle["High"]
            >= previous_candle["High"]
            and current_candle["High"]
            >= next_candle["High"]
        ):
            swing_highs.append(
                float(current_candle["High"])
            )

        if (
            current_candle["Low"]
            <= previous_candle["Low"]
            and current_candle["Low"]
            <= next_candle["Low"]
        ):
            swing_lows.append(
                float(current_candle["Low"])
            )

    return swing_highs, swing_lows

def cluster_candidates(
    candidates: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: candidate["price"],
    )

    clusters: list[list[dict[str, Any]]] = []

    for candidate in sorted_candidates:
        if not clusters:
            clusters.append([candidate])
            continue

        current_cluster = clusters[-1]

        cluster_price = float(
            np.average(
                [
                    item["price"]
                    for item in current_cluster
                ],
                weights=[
                    item["weight"]
                    for item in current_cluster
                ],
            )
        )

        if (
            abs(candidate["price"] - cluster_price)
            <= tolerance
        ):
            current_cluster.append(candidate)
        else:
            clusters.append([candidate])

    final_clusters: list[dict[str, Any]] = []

    for cluster in clusters:
        prices = [
            float(item["price"])
            for item in cluster
        ]

        weights = [
            float(item["weight"])
            for item in cluster
        ]

        final_clusters.append(
            {
                "price": float(
                    np.average(
                        prices,
                        weights=weights,
                    )
                ),
                "strength": float(sum(weights)),
                "sources": list(
                    dict.fromkeys(
                        item["source"]
                        for item in cluster
                    )
                ),
            }
        )

    return final_clusters

def choose_immediate_level(
    clusters: list[dict[str, Any]],
    current_dataframe: pd.DataFrame,
    current_price: float,
    atr: float,
    step: float,
    tolerance: float,
    level_type: str,
) -> dict[str, Any]:
    minimum_distance = max(
        atr * 0.18,
        step * 0.30,
    )

    maximum_distance = max(
        atr * 3.5,
        step * 5,
    )

    valid_levels = []

    for cluster in clusters:
        level_price = float(cluster["price"])

        if level_type == "support":
            distance = current_price - level_price

            if distance < minimum_distance:
                continue

            candle_touch_count = int(
                (
                    (
                        current_dataframe["Low"]
                        - level_price
                    ).abs()
                    <= tolerance
                ).sum()
            )

        else:
            distance = level_price - current_price

            if distance < minimum_distance:
                continue

            candle_touch_count = int(
                (
                    (
                        current_dataframe["High"]
                        - level_price
                    ).abs()
                    <= tolerance
                ).sum()
            )

        if distance > maximum_distance:
            continue

        distance_in_atr = (
            distance / max(atr, 0.01)
        )

        proximity_bonus = max(
            0,
            3.0 - distance_in_atr,
        )

        final_score = (
            cluster["strength"]
            + candle_touch_count * 1.25
            + proximity_bonus
            - distance_in_atr * 0.80
        )

        valid_levels.append(
            {
                **cluster,
                "distance": distance,
                "touches": candle_touch_count,
                "score": final_score,
            }
        )

    if valid_levels:
        selected_level = max(
            valid_levels,
            key=lambda item: item["score"],
        )

        return {
            **selected_level,
            "price": float(
                selected_level["price"]
            ),
        }

    fallback_price = (
        current_price - atr
        if level_type == "support"
        else current_price + atr
    )

    return {
        "price": float(fallback_price),
        "strength": 0,
        "touches": 0,
        "score": 0,
        "sources": ["ATR Fallback"],
    }

def analyse_index(
    dataframe: pd.DataFrame,
    selected_date,
) -> dict[str, Any]:
    df = dataframe.copy()

    if df.empty:
        raise ValueError(
            "Analysis ke liye candle data empty hai."
        )

    df = (
        df
        .sort_values("Datetime")
        .drop_duplicates(subset=["Datetime"])
        .reset_index(drop=True)
    )

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
            "Selected date ke liye candles nahi mili."
        )

    previous_dates = sorted(
        session_date
        for session_date
        in df["SessionDate"].unique()
        if session_date < selected_date
    )

    previous_df = pd.DataFrame()

    if previous_dates:
        previous_df = df[
            df["SessionDate"]
            == previous_dates[-1]
        ].copy()

    latest = current_df.iloc[-1]

    current_price = float(latest["Close"])
    day_open = float(current_df.iloc[0]["Open"])
    day_high = float(current_df["High"].max())
    day_low = float(current_df["Low"].min())

    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])

    atr = float(latest["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        atr = max(
            current_price * 0.002,
            1.0,
        )

    # Nifty levels 10 points aur Bank Nifty
    # levels 50 points par round honge.
    step = 50.0 if current_price > 40000 else 10.0

    tolerance = max(
        atr * 0.20,
        step * 0.35,
    )

    opening_range = current_df.head(
        min(3, len(current_df))
    )

    opening_high = float(
        opening_range["High"].max()
    )

    opening_low = float(
        opening_range["Low"].min()
    )

    typical_price = (
        current_df["High"]
        + current_df["Low"]
        + current_df["Close"]
    ) / 3

    session_average = float(
        typical_price.mean()
    )

    swing_highs, swing_lows = find_swings(
        current_df
    )

    candidates: list[dict[str, Any]] = []

    def add_candidate(
        price: float,
        source: str,
        weight: float,
    ) -> None:
        if np.isfinite(price):
            candidates.append(
                {
                    "price": float(price),
                    "source": source,
                    "weight": float(weight),
                }
            )

    # Recent candles ko sabse zyada priority.
    recent_df = current_df.tail(8)

    for position, (_, candle) in enumerate(
        recent_df.iterrows(),
        start=1,
    ):
        recency_weight = (
            2.2
            + position / max(len(recent_df), 1)
        )

        add_candidate(
            candle["Low"],
            "Recent Candle Low",
            recency_weight,
        )

        add_candidate(
            candle["High"],
            "Recent Candle High",
            recency_weight,
        )

    add_candidate(
        opening_low,
        "Opening Range Low",
        3.0,
    )

    add_candidate(
        opening_high,
        "Opening Range High",
        3.0,
    )

    add_candidate(
        day_open,
        "Day Open",
        2.5,
    )

    add_candidate(
        session_average,
        "Session Average",
        2.5,
    )

    add_candidate(
        ema20,
        "EMA 20",
        2.4,
    )

    add_candidate(
        ema50,
        "EMA 50",
        1.5,
    )

    add_candidate(
        day_low,
        "Current Day Low",
        2.0,
    )

    add_candidate(
        day_high,
        "Current Day High",
        2.0,
    )

    for swing_low in swing_lows:
        add_candidate(
            swing_low,
            "Swing Low",
            2.2,
        )

    for swing_high in swing_highs:
        add_candidate(
            swing_high,
            "Swing High",
            2.2,
        )

    # Psychological round-number levels.
    nearest_round_level = round_to_step(
        current_price,
        step,
    )

    for offset in range(-5, 6):
        round_level = (
            nearest_round_level
            + offset * step
        )

        round_weight = 2.4

        if (
            step == 50
            and round_level % 100 == 0
        ):
            round_weight = 3.4

        add_candidate(
            round_level,
            "Psychological Round Level",
            round_weight,
        )

    # Previous-day levels tabhi include honge
    # jab current price ke reasonably paas hon.
    if not previous_df.empty:
        previous_levels = [
            (
                float(previous_df["High"].max()),
                "Previous Day High",
                2.5,
            ),
            (
                float(previous_df["Low"].min()),
                "Previous Day Low",
                2.5,
            ),
            (
                float(
                    previous_df.iloc[-1]["Close"]
                ),
                "Previous Day Close",
                2.2,
            ),
        ]

        for (
            previous_price,
            previous_source,
            previous_weight,
        ) in previous_levels:
            if (
                abs(
                    previous_price
                    - current_price
                )
                <= max(
                    atr * 4,
                    step * 6,
                )
            ):
                add_candidate(
                    previous_price,
                    previous_source,
                    previous_weight,
                )

    clusters = cluster_candidates(
        candidates,
        tolerance,
    )

    support = choose_immediate_level(
        clusters=clusters,
        current_dataframe=current_df,
        current_price=current_price,
        atr=atr,
        step=step,
        tolerance=tolerance,
        level_type="support",
    )

    resistance = choose_immediate_level(
        clusters=clusters,
        current_dataframe=current_df,
        current_price=current_price,
        atr=atr,
        step=step,
        tolerance=tolerance,
        level_type="resistance",
    )

    support_raw_level = float(
        support["price"]
    )

    resistance_raw_level = float(
        resistance["price"]
    )

    support_level = normalize_trigger_level(
        raw_price=support_raw_level,
        current_price=current_price,
        level_type="support",
    )

    resistance_level = normalize_trigger_level(
        raw_price=resistance_raw_level,
        current_price=current_price,
        level_type="resistance",
    )

    chart_score = 0

    if current_price > ema20:
        chart_score += 1
    else:
        chart_score -= 1

    if ema20 > ema50:
        chart_score += 1
    else:
        chart_score -= 1

    if current_price > day_open:
        chart_score += 1
    else:
        chart_score -= 1

    if current_price > session_average:
        chart_score += 1
    else:
        chart_score -= 1

    recent_closes = current_df["Close"].tail(5)

    if len(recent_closes) >= 3:
        close_slope = float(
            np.polyfit(
                range(len(recent_closes)),
                recent_closes,
                1,
            )[0]
        )

        if close_slope > atr * 0.025:
            chart_score += 1
        elif close_slope < -(atr * 0.025):
            chart_score -= 1

    if chart_score >= 2:
        chart_view = "Bullish"
    elif chart_score <= -2:
        chart_view = "Bearish"
    else:
        chart_view = "Sideways"

    bullish_minimum_distance = max(
        atr * 0.80,
        step,
    )

    bullish_maximum_distance = max(
        atr * 2.00,
        step * 3,
    )

    bearish_minimum_distance = max(
        atr * 0.80,
        step,
    )

    bearish_maximum_distance = max(
        atr * 2.50,
        step * 3,
    )

    bullish_minimum_target = round_to_step(
        resistance_level
        + bullish_minimum_distance,
        step,
    )

    bullish_maximum_target = round_to_step(
        resistance_level
        + bullish_maximum_distance,
        step,
    )

    bearish_minimum_target = round_to_step(
        support_level
        - bearish_minimum_distance,
        step,
    )

    bearish_maximum_target = round_to_step(
        support_level
        - bearish_maximum_distance,
        step,
    )

    return {
        "current_price": current_price,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "ema20": ema20,
        "ema50": ema50,
        "atr": atr,
        "chart_score": chart_score,
        "chart_view": chart_view,

        "support_level": support_level,
        "resistance_level": resistance_level,

        "support_raw_level": support_raw_level,
        "resistance_raw_level": resistance_raw_level,

        "support_sources": support["sources"],
        "resistance_sources": resistance["sources"],

        "support_touches": support["touches"],
        "resistance_touches": resistance["touches"],

        "bullish_trigger": resistance_level,
        "bullish_minimum_target": (
            bullish_minimum_target
        ),
        "bullish_maximum_target": (
            bullish_maximum_target
        ),

        "bearish_trigger": support_level,
        "bearish_minimum_target": (
            bearish_minimum_target
        ),
        "bearish_maximum_target": (
            bearish_maximum_target
        ),
    }
