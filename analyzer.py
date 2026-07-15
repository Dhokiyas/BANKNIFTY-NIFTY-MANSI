from __future__ import annotations

from datetime import datetime
from typing import Any

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
            (dataframe["High"] - previous_close).abs(),
            (dataframe["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=1,
    ).mean()

def align_datetime(
    datetime_series: pd.Series,
    selected_datetime: datetime,
) -> pd.Timestamp:
    cutoff = pd.Timestamp(selected_datetime)
    timezone = datetime_series.dt.tz

    if timezone is not None:
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(timezone)
        else:
            cutoff = cutoff.tz_convert(timezone)

    elif cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    return cutoff

def get_completed_candles(
    dataframe: pd.DataFrame,
    selected_date,
    selected_datetime: datetime | None,
) -> pd.DataFrame:
    session_df = dataframe[
        dataframe["Datetime"].dt.date == selected_date
    ].copy()

    if session_df.empty:
        raise ValueError(
            "Selected date ke liye candle data nahi mila."
        )

    if selected_datetime is not None:
        cutoff = align_datetime(
            session_df["Datetime"],
            selected_datetime,
        )

        # Angel One timestamp candle ka start time hota hai.
        # Selected time ki running candle include nahi hogi.
        completed_df = session_df[
            session_df["Datetime"] < cutoff
        ].copy()

        if completed_df.empty:
            raise ValueError(
                "Analysis ke liye kam se kam ek completed "
                "5-minute candle chahiye."
            )

        session_df = completed_df

    return (
        session_df
        .sort_values("Datetime")
        .drop_duplicates(subset=["Datetime"])
        .reset_index(drop=True)
    )

def find_pivots(
    dataframe: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    if len(dataframe) < 3:
        return pivot_highs, pivot_lows

    for index in range(1, len(dataframe) - 1):
        previous_row = dataframe.iloc[index - 1]
        current_row = dataframe.iloc[index]
        next_row = dataframe.iloc[index + 1]

        if (
            current_row["High"] >= previous_row["High"]
            and current_row["High"] >= next_row["High"]
        ):
            pivot_highs.append(
                float(current_row["High"])
            )

        if (
            current_row["Low"] <= previous_row["Low"]
            and current_row["Low"] <= next_row["Low"]
        ):
            pivot_lows.append(
                float(current_row["Low"])
            )

    return pivot_highs, pivot_lows

def cluster_levels(
    candidates: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: item["price"],
    )

    groups: list[list[dict[str, Any]]] = []

    for candidate in candidates:
        if not groups:
            groups.append([candidate])
            continue

        latest_group = groups[-1]

        average_price = float(
            np.average(
                [
                    item["price"]
                    for item in latest_group
                ],
                weights=[
                    item["weight"]
                    for item in latest_group
                ],
            )
        )

        if (
            abs(candidate["price"] - average_price)
            <= tolerance
        ):
            latest_group.append(candidate)
        else:
            groups.append([candidate])

    clusters: list[dict[str, Any]] = []

    for group in groups:
        prices = [
            float(item["price"])
            for item in group
        ]

        weights = [
            float(item["weight"])
            for item in group
        ]

        clusters.append(
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
                        for item in group
                    )
                ),
            }
        )

    return clusters

def calculate_level_statistics(
    dataframe: pd.DataFrame,
    level_price: float,
    tolerance: float,
    level_type: str,
) -> dict[str, int]:
    touches = 0
    recent_touches = 0
    rejections = 0

    recent_start = max(
        len(dataframe) - 6,
        0,
    )

    for position, row in dataframe.iterrows():
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])

        candle_crossed_level = (
            low <= level_price + tolerance
            and high >= level_price - tolerance
        )

        if candle_crossed_level:
            touches += 1

            if position >= recent_start:
                recent_touches += 1

        if level_type == "support":
            rejected = (
                low <= level_price + tolerance
                and close > level_price + tolerance * 0.15
            )

        else:
            rejected = (
                high >= level_price - tolerance
                and close < level_price - tolerance * 0.15
            )

        if rejected:
            rejections += 1

    return {
        "touches": touches,
        "recent_touches": recent_touches,
        "rejections": rejections,
    }

def score_levels(
    clusters: list[dict[str, Any]],
    current_df: pd.DataFrame,
    current_price: float,
    atr: float,
    tolerance: float,
    level_type: str,
    is_bank_nifty: bool,
) -> list[dict[str, Any]]:
    minimum_gap = max(
        atr * 0.12,
        5.0 if not is_bank_nifty else 15.0,
    )

    maximum_gap = max(
        atr * 4.5,
        180.0 if not is_bank_nifty else 600.0,
    )

    results: list[dict[str, Any]] = []

    for cluster in clusters:
        level_price = float(cluster["price"])

        if level_type == "support":
            distance = current_price - level_price
        else:
            distance = level_price - current_price

        if distance < minimum_gap:
            continue

        if distance > maximum_gap:
            continue

        statistics = calculate_level_statistics(
            dataframe=current_df,
            level_price=level_price,
            tolerance=tolerance,
            level_type=level_type,
        )

        distance_in_atr = (
            distance / max(atr, 0.01)
        )

        proximity_bonus = max(
            0.0,
            4.0 - distance_in_atr,
        )

        score = (
            float(cluster["strength"])
            + statistics["touches"] * 1.10
            + statistics["recent_touches"] * 1.60
            + statistics["rejections"] * 1.35
            + proximity_bonus
            - distance_in_atr * 0.65
        )

        results.append(
            {
                **cluster,
                **statistics,
                "distance": distance,
                "score": score,
            }
        )

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

def round_actionable_level(
    price: float,
    is_bank_nifty: bool,
) -> float:
    step = 10.0 if is_bank_nifty else 5.0

    return float(
        round(price / step) * step
    )

def calculate_chart_levels(
    full_dataframe: pd.DataFrame,
    current_df: pd.DataFrame,
    selected_date,
    current_price: float,
    ema20: float,
    ema50: float,
    atr: float,
) -> dict[str, Any]:
    is_bank_nifty = current_price > 40000

    tolerance = max(
        atr * 0.18,
        6.0 if not is_bank_nifty else 18.0,
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

    # Recent structure
    recent_df = current_df.tail(8)
    recent_count = max(len(recent_df), 1)

    for number, (_, candle) in enumerate(
        recent_df.iterrows(),
        start=1,
    ):
        recency_weight = (
            2.2 + number / recent_count
        )

        add_candidate(
            float(candle["Low"]),
            "Recent Candle Low",
            recency_weight,
        )

        add_candidate(
            float(candle["High"]),
            "Recent Candle High",
            recency_weight,
        )

    # Opening 15-minute range
    opening_range = current_df.head(
        min(3, len(current_df))
    )

    add_candidate(
        float(opening_range["Low"].min()),
        "Opening Range Low",
        3.4,
    )

    add_candidate(
        float(opening_range["High"].max()),
        "Opening Range High",
        3.4,
    )

    # Session structure
    add_candidate(
        float(current_df["Low"].min()),
        "Current Day Low",
        2.4,
    )

    add_candidate(
        float(current_df["High"].max()),
        "Current Day High",
        2.4,
    )

    add_candidate(
        float(current_df.iloc[0]["Open"]),
        "Day Open",
        2.4,
    )

    session_average = float(
        (
            current_df["High"]
            + current_df["Low"]
            + current_df["Close"]
        ).mean()
        / 3
    )

    add_candidate(
        session_average,
        "Session Average",
        2.2,
    )

    add_candidate(
        ema20,
        "EMA 20",
        2.1,
    )

    add_candidate(
        ema50,
        "EMA 50",
        1.5,
    )

    # Swing structure
    pivot_highs, pivot_lows = find_pivots(
        current_df
    )

    for pivot_high in pivot_highs:
        add_candidate(
            pivot_high,
            "Swing High",
            3.0,
        )

    for pivot_low in pivot_lows:
        add_candidate(
            pivot_low,
            "Swing Low",
            3.0,
        )

    # Previous trading session
    full_dataframe = full_dataframe.copy()

    full_dataframe["SessionDate"] = (
        full_dataframe["Datetime"].dt.date
    )

    previous_dates = sorted(
        session_date
        for session_date
        in full_dataframe["SessionDate"].unique()
        if session_date < selected_date
    )

    if previous_dates:
        previous_df = full_dataframe[
            full_dataframe["SessionDate"]
            == previous_dates[-1]
        ].copy()

        previous_high = float(
            previous_df["High"].max()
        )

        previous_low = float(
            previous_df["Low"].min()
        )

        previous_close = float(
            previous_df.iloc[-1]["Close"]
        )

        previous_pivot = (
            previous_high
            + previous_low
            + previous_close
        ) / 3

        previous_levels = [
            (
                previous_high,
                "Previous Day High",
                3.2,
            ),
            (
                previous_low,
                "Previous Day Low",
                3.2,
            ),
            (
                previous_close,
                "Previous Day Close",
                2.6,
            ),
            (
                previous_pivot,
                "Previous Day Pivot",
                2.5,
            ),
        ]

        maximum_distance = max(
            atr * 5,
            250.0 if not is_bank_nifty else 700.0,
        )

        for price, source, weight in previous_levels:
            if (
                abs(price - current_price)
                <= maximum_distance
            ):
                add_candidate(
                    price,
                    source,
                    weight,
                )

    # Psychological levels
    if is_bank_nifty:
        lower_level = (
            int(current_price // 50)
            * 50
        )

        for offset in range(-10, 11):
            level = lower_level + offset * 50

            weight = (
                3.3
                if level % 100 == 0
                else 2.5
            )

            add_candidate(
                level,
                (
                    "100 Point Psychological Level"
                    if level % 100 == 0
                    else "50 Point Psychological Level"
                ),
                weight,
            )

    else:
        lower_level = (
            int(current_price // 20)
            * 20
        )

        for offset in range(-10, 11):
            add_candidate(
                lower_level + offset * 20,
                "20 Point Psychological Level",
                2.4,
            )

    clusters = cluster_levels(
        candidates=candidates,
        tolerance=tolerance,
    )

    support_candidates = score_levels(
        clusters=clusters,
        current_df=current_df,
        current_price=current_price,
        atr=atr,
        tolerance=tolerance,
        level_type="support",
        is_bank_nifty=is_bank_nifty,
    )

    resistance_candidates = score_levels(
        clusters=clusters,
        current_df=current_df,
        current_price=current_price,
        atr=atr,
        tolerance=tolerance,
        level_type="resistance",
        is_bank_nifty=is_bank_nifty,
    )

    if support_candidates:
        support = support_candidates[0]
    else:
        support = {
            "price": float(recent_df["Low"].min()),
            "sources": ["Recent Structure Low"],
            "touches": 1,
        }

    if resistance_candidates:
        resistance = resistance_candidates[0]
    else:
        resistance = {
            "price": float(recent_df["High"].max()),
            "sources": ["Recent Structure High"],
            "touches": 1,
        }

    support_level = round_actionable_level(
        float(support["price"]),
        is_bank_nifty,
    )

    resistance_level = round_actionable_level(
        float(resistance["price"]),
        is_bank_nifty,
    )

    display_step = (
        10.0 if is_bank_nifty else 5.0
    )

    if support_level >= current_price:
        support_level -= display_step

    if resistance_level <= current_price:
        resistance_level += display_step

    # Next levels for targets
    minimum_target_gap = max(
        atr * 0.55,
        20.0 if not is_bank_nifty else 50.0,
    )

    higher_levels = sorted(
        {
            round_actionable_level(
                float(item["price"]),
                is_bank_nifty,
            )
            for item in resistance_candidates
            if float(item["price"])
            > resistance_level + minimum_target_gap
        }
    )

    lower_levels = sorted(
        {
            round_actionable_level(
                float(item["price"]),
                is_bank_nifty,
            )
            for item in support_candidates
            if float(item["price"])
            < support_level - minimum_target_gap
        },
        reverse=True,
    )

    upside_minimum_fallback = (
        resistance_level
        + max(
            atr * 0.80,
            25.0 if not is_bank_nifty else 70.0,
        )
    )

    upside_maximum_fallback = (
        resistance_level
        + max(
            atr * 1.80,
            60.0 if not is_bank_nifty else 160.0,
        )
    )

    downside_minimum_fallback = (
        support_level
        - max(
            atr * 0.80,
            25.0 if not is_bank_nifty else 70.0,
        )
    )

    downside_maximum_fallback = (
        support_level
        - max(
            atr * 1.80,
            60.0 if not is_bank_nifty else 160.0,
        )
    )

    bullish_minimum_target = (
        higher_levels[0]
        if len(higher_levels) >= 1
        else round_actionable_level(
            upside_minimum_fallback,
            is_bank_nifty,
        )
    )

    bullish_maximum_target = (
        higher_levels[1]
        if len(higher_levels) >= 2
        else round_actionable_level(
            upside_maximum_fallback,
            is_bank_nifty,
        )
    )

    bearish_minimum_target = (
        lower_levels[0]
        if len(lower_levels) >= 1
        else round_actionable_level(
            downside_minimum_fallback,
            is_bank_nifty,
        )
    )

    bearish_maximum_target = (
        lower_levels[1]
        if len(lower_levels) >= 2
        else round_actionable_level(
            downside_maximum_fallback,
            is_bank_nifty,
        )
    )

    return {
        "support_level": float(support_level),
        "resistance_level": float(resistance_level),

        "support_raw_level": float(
            support["price"]
        ),
        "resistance_raw_level": float(
            resistance["price"]
        ),

        "support_sources": support["sources"],
        "resistance_sources": resistance["sources"],

        "support_touches": int(
            support.get("touches", 0)
        ),
        "resistance_touches": int(
            resistance.get("touches", 0)
        ),

        "bullish_minimum_target": float(
            bullish_minimum_target
        ),
        "bullish_maximum_target": float(
            bullish_maximum_target
        ),
        "bearish_minimum_target": float(
            bearish_minimum_target
        ),
        "bearish_maximum_target": float(
            bearish_maximum_target
        ),
    }

def calculate_chart_view(
    current_df: pd.DataFrame,
    current_price: float,
    day_open: float,
    ema20: float,
    ema50: float,
    session_average: float,
    opening_high: float,
    opening_low: float,
    atr: float,
) -> tuple[str, int]:
    score = 0

    score += 1 if current_price > ema20 else -1
    score += 1 if ema20 > ema50 else -1
    score += 1 if current_price > day_open else -1
    score += (
        1
        if current_price > session_average
        else -1
    )

    if current_price > opening_high:
        score += 2

    elif current_price < opening_low:
        score -= 2

    recent_closes = current_df["Close"].tail(5)

    if len(recent_closes) >= 3:
        slope = float(
            np.polyfit(
                range(len(recent_closes)),
                recent_closes,
                1,
            )[0]
        )

        if slope > atr * 0.025:
            score += 1

        elif slope < -(atr * 0.025):
            score -= 1

    recent_high = float(
        current_df["High"].tail(6).max()
    )

    recent_low = float(
        current_df["Low"].tail(6).min()
    )

    range_position = (
        (current_price - recent_low)
        / max(
            recent_high - recent_low,
            0.01,
        )
    )

    if range_position >= 0.70:
        score += 1

    elif range_position <= 0.30:
        score -= 1

    pivot_highs, pivot_lows = find_pivots(
        current_df
    )

    if len(pivot_highs) >= 2:
        if pivot_highs[-1] > pivot_highs[-2]:
            score += 1
        elif pivot_highs[-1] < pivot_highs[-2]:
            score -= 1

    if len(pivot_lows) >= 2:
        if pivot_lows[-1] > pivot_lows[-2]:
            score += 1
        elif pivot_lows[-1] < pivot_lows[-2]:
            score -= 1

    if score >= 3:
        return "Bullish", score

    if score <= -3:
        return "Bearish", score

    return "Sideways", score

def analyse_index(
    dataframe: pd.DataFrame,
    selected_date,
    selected_datetime: datetime | None = None,
) -> dict[str, Any]:
    df = dataframe.copy()

    if df.empty:
        raise ValueError(
            "Analysis ke liye candle data empty hai."
        )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    df[numeric_columns] = df[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )
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

    current_df = get_completed_candles(
        dataframe=df,
        selected_date=selected_date,
        selected_datetime=selected_datetime,
    )

    latest_datetime = current_df.iloc[-1][
        "Datetime"
    ]

    latest_row = df[
        df["Datetime"] == latest_datetime
    ].iloc[-1]

    current_price = float(
        latest_row["Close"]
    )

    day_open = float(
        current_df.iloc[0]["Open"]
    )

    day_high = float(
        current_df["High"].max()
    )

    day_low = float(
        current_df["Low"].min()
    )

    ema20 = float(
        latest_row["EMA20"]
    )

    ema50 = float(
        latest_row["EMA50"]
    )

    atr = float(
        latest_row["ATR"]
    )

    if not np.isfinite(atr) or atr <= 0:
        atr = max(
            current_price * 0.002,
            1.0,
        )

    session_average = float(
        (
            current_df["High"]
            + current_df["Low"]
            + current_df["Close"]
        ).mean()
        / 3
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

    level_analysis = calculate_chart_levels(
        full_dataframe=df,
        current_df=current_df,
        selected_date=selected_date,
        current_price=current_price,
        ema20=ema20,
        ema50=ema50,
        atr=atr,
    )

    chart_view, chart_score = (
        calculate_chart_view(
            current_df=current_df,
            current_price=current_price,
            day_open=day_open,
            ema20=ema20,
            ema50=ema50,
            session_average=session_average,
            opening_high=opening_high,
            opening_low=opening_low,
            atr=atr,
        )
    )

    support_level = level_analysis[
        "support_level"
    ]

    resistance_level = level_analysis[
        "resistance_level"
    ]

    breakout_buffer = max(
        atr * 0.12,
        5.0 if current_price < 40000 else 15.0,
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

        "support_raw_level": level_analysis[
            "support_raw_level"
        ],
        "resistance_raw_level": level_analysis[
            "resistance_raw_level"
        ],

        "support_sources": level_analysis[
            "support_sources"
        ],
        "resistance_sources": level_analysis[
            "resistance_sources"
        ],

        "support_touches": level_analysis[
            "support_touches"
        ],
        "resistance_touches": level_analysis[
            "resistance_touches"
        ],

        "bullish_trigger": resistance_level,
        "bearish_trigger": support_level,

        "breakout_confirmation_above": (
            resistance_level + breakout_buffer
        ),
        "breakdown_confirmation_below": (
            support_level - breakout_buffer
        ),

        "bullish_minimum_target": (
            level_analysis[
                "bullish_minimum_target"
            ]
        ),
        "bullish_maximum_target": (
            level_analysis[
                "bullish_maximum_target"
            ]
        ),

        "bearish_minimum_target": (
            level_analysis[
                "bearish_minimum_target"
            ]
        ),
        "bearish_maximum_target": (
            level_analysis[
                "bearish_maximum_target"
            ]
        ),

        "analysis_candle_time": latest_datetime,
    }
