from __future__ import annotations

import math
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

def floor_to_step(
    value: float,
    step: float,
) -> float:
    return float(
        math.floor((value + 0.000001) / step)
        * step
    )

def ceil_to_step(
    value: float,
    step: float,
) -> float:
    return float(
        math.ceil((value - 0.000001) / step)
        * step
    )

def align_cutoff_timezone(
    datetime_series: pd.Series,
    selected_datetime: datetime,
) -> pd.Timestamp:
    cutoff = pd.Timestamp(selected_datetime)
    series_timezone = datetime_series.dt.tz

    if series_timezone is not None:
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(
                series_timezone
            )
        else:
            cutoff = cutoff.tz_convert(
                series_timezone
            )

    elif cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    return cutoff

def get_completed_session_candles(
    dataframe: pd.DataFrame,
    selected_date,
    selected_datetime: datetime | None,
) -> pd.DataFrame:
    session_df = dataframe[
        dataframe["Datetime"].dt.date
        == selected_date
    ].copy()

    if session_df.empty:
        raise ValueError(
            "Selected date ke liye candles nahi mili."
        )

    if selected_datetime is not None:
        cutoff = align_cutoff_timezone(
            session_df["Datetime"],
            selected_datetime,
        )

        # Selected timestamp wali candle running candle hai.
        # Sirf usse pehle ki completed candles use hongi.
        completed_df = session_df[
            session_df["Datetime"] < cutoff
        ].copy()

        if not completed_df.empty:
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
            current_row["High"]
            >= previous_row["High"]
            and current_row["High"]
            >= next_row["High"]
        ):
            pivot_highs.append(
                float(current_row["High"])
            )

        if (
            current_row["Low"]
            <= previous_row["Low"]
            and current_row["Low"]
            <= next_row["Low"]
        ):
            pivot_lows.append(
                float(current_row["Low"])
            )

    return pivot_highs, pivot_lows

def cluster_candidates(
    candidates: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: item["price"],
    )

    grouped: list[list[dict[str, Any]]] = []

    for candidate in candidates:
        if not grouped:
            grouped.append([candidate])
            continue

        latest_group = grouped[-1]

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
            abs(
                candidate["price"]
                - average_price
            )
            <= tolerance
        ):
            latest_group.append(candidate)
        else:
            grouped.append([candidate])

    clusters: list[dict[str, Any]] = []

    for group in grouped:
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

def calculate_touch_stats(
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
        is_recent = position >= recent_start

        if level_type == "support":
            touched = (
                abs(
                    float(row["Low"])
                    - level_price
                )
                <= tolerance
                or (
                    float(row["Low"])
                    <= level_price + tolerance
                    and float(row["High"])
                    >= level_price - tolerance
                )
            )

            rejected = (
                float(row["Low"])
                <= level_price + tolerance
                and float(row["Close"])
                > level_price
            )

        else:
            touched = (
                abs(
                    float(row["High"])
                    - level_price
                )
                <= tolerance
                or (
                    float(row["Low"])
                    <= level_price + tolerance
                    and float(row["High"])
                    >= level_price - tolerance
                )
            )

            rejected = (
                float(row["High"])
                >= level_price - tolerance
                and float(row["Close"])
                < level_price
            )

        if touched:
            touches += 1

            if is_recent:
                recent_touches += 1

        if rejected:
            rejections += 1

    return {
        "touches": touches,
        "recent_touches": recent_touches,
        "rejections": rejections,
    }

def select_best_cluster(
    clusters: list[dict[str, Any]],
    current_df: pd.DataFrame,
    current_price: float,
    atr: float,
    tolerance: float,
    level_type: str,
) -> dict[str, Any] | None:
    minimum_gap = max(
        atr * 0.18,
        10.0,
    )

    maximum_gap = max(
        atr * 4.5,
        120.0 if current_price < 40000 else 450.0,
    )

    scored_levels: list[dict[str, Any]] = []

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

        statistics = calculate_touch_stats(
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
            + statistics["recent_touches"] * 1.55
            + statistics["rejections"] * 1.40
            + proximity_bonus
            - distance_in_atr * 0.45
        )

        scored_levels.append(
            {
                **cluster,
                **statistics,
                "distance": distance,
                "score": score,
            }
        )

    if not scored_levels:
        return None

    return max(
        scored_levels,
        key=lambda item: item["score"],
    )

def calculate_actionable_levels(
    full_dataframe: pd.DataFrame,
    current_df: pd.DataFrame,
    current_price: float,
    ema20: float,
    atr: float,
    selected_date,
) -> dict[str, Any]:
    is_bank_nifty = current_price > 40000

    tolerance = max(
        atr * 0.25,
        7.0 if not is_bank_nifty else 18.0,
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

    recent_df = current_df.tail(8)

    total_recent = max(
        len(recent_df),
        1,
    )

    for position, (_, candle) in enumerate(
        recent_df.iterrows(),
        start=1,
    ):
        recency_weight = (
            2.4
            + position / total_recent
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

    opening_range = current_df.head(
        min(3, len(current_df))
    )

    add_candidate(
        float(opening_range["Low"].min()),
        "Opening Range Low",
        3.2,
    )

    add_candidate(
        float(opening_range["High"].max()),
        "Opening Range High",
        3.2,
    )

    add_candidate(
        float(current_df.iloc[0]["Open"]),
        "Day Open",
        2.5,
    )

    typical_price = (
        current_df["High"]
        + current_df["Low"]
        + current_df["Close"]
    ) / 3

    add_candidate(
        float(typical_price.mean()),
        "Session Average",
        2.2,
    )

    add_candidate(
        ema20,
        "EMA 20",
        2.3,
    )

    pivot_highs, pivot_lows = find_pivots(
        current_df
    )

    for pivot_high in pivot_highs:
        add_candidate(
            pivot_high,
            "Swing High",
            2.8,
        )

    for pivot_low in pivot_lows:
        add_candidate(
            pivot_low,
            "Swing Low",
            2.8,
        )

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

        previous_levels = [
            (
                float(previous_df["High"].max()),
                "Previous Day High",
                3.0,
            ),
            (
                float(previous_df["Low"].min()),
                "Previous Day Low",
                3.0,
            ),
            (
                float(
                    previous_df.iloc[-1]["Close"]
                ),
                "Previous Day Close",
                2.4,
            ),
        ]

        maximum_previous_distance = max(
            atr * 5,
            180 if not is_bank_nifty else 500,
        )

        for price, source, weight in previous_levels:
            if (
                abs(price - current_price)
                <= maximum_previous_distance
            ):
                add_candidate(
                    price,
                    source,
                    weight,
                )

    # Psychological levels
    # Nifty: 20-point trading levels
    # Bank Nifty: 20, 50 and 100-point levels
    base_twenty = floor_to_step(
        current_price,
        20.0,
    )

    for offset in range(-8, 9):
        add_candidate(
            base_twenty + offset * 20.0,
            "20 Point Psychological Level",
            2.3,
        )

    if is_bank_nifty:
        base_fifty = floor_to_step(
            current_price,
            50.0,
        )

        base_hundred = floor_to_step(
            current_price,
            100.0,
        )

        for offset in range(-5, 6):
            add_candidate(
                base_fifty + offset * 50.0,
                "50 Point Psychological Level",
                3.2,
            )

            add_candidate(
                base_hundred + offset * 100.0,
                "100 Point Psychological Level",
                4.4,
            )

    clusters = cluster_candidates(
        candidates=candidates,
        tolerance=tolerance,
    )

    selected_support = select_best_cluster(
        clusters=clusters,
        current_df=current_df,
        current_price=current_price,
        atr=atr,
        tolerance=tolerance,
        level_type="support",
    )

    selected_resistance = select_best_cluster(
        clusters=clusters,
        current_df=current_df,
        current_price=current_price,
        atr=atr,
        tolerance=tolerance,
        level_type="resistance",
    )

    if selected_support is None:
        selected_support = {
            "price": float(
                recent_df["Low"].min()
            ),
            "sources": [
                "Recent Structure Low"
            ],
            "touches": 1,
            "score": 0,
        }

    if selected_resistance is None:
        selected_resistance = {
            "price": float(
                recent_df["High"].max()
            ),
            "sources": [
                "Recent Structure High"
            ],
            "touches": 1,
            "score": 0,
        }

    support_raw = float(
        selected_support["price"]
    )

    resistance_raw = float(
        selected_resistance["price"]
    )

    # Manual chart-view mein trigger levels
    # clean 20-point actionable levels par hote hain.
    support_level = floor_to_step(
        support_raw,
        20.0,
    )

    resistance_level = ceil_to_step(
        resistance_raw,
        20.0,
    )

    if support_level >= current_price:
        support_level -= 20.0

    if resistance_level <= current_price:
        resistance_level += 20.0

    # Bank Nifty mein strong 100-point level hold
    # immediate support ko priority deta hai.
    if is_bank_nifty:
        hundred_support = floor_to_step(
            current_price,
            100.0,
        )

        hundred_distance = (
            current_price
            - hundred_support
        )

        last_closes = current_df[
            "Close"
        ].tail(3)

        recent_low = float(
            current_df["Low"].tail(4).min()
        )

        hundred_is_held = (
            hundred_distance >= 10
            and hundred_distance
            <= max(120.0, atr * 1.50)
            and bool(
                (
                    last_closes
                    >= hundred_support
                ).sum()
                >= max(1, len(last_closes) - 1)
            )
            and recent_low
            <= hundred_support + max(
                45.0,
                tolerance,
            )
        )

        if hundred_is_held:
            support_level = float(
                hundred_support
            )

            selected_support = {
                **selected_support,
                "sources": [
                    "100 Point Psychological Level",
                    "Recent Price Hold",
                    "Wick Rejection",
                ],
            }

    return {
        "support_level": float(
            support_level
        ),
        "resistance_level": float(
            resistance_level
        ),
        "support_raw_level": support_raw,
        "resistance_raw_level": resistance_raw,
        "support_sources": selected_support[
            "sources"
        ],
        "resistance_sources": selected_resistance[
            "sources"
        ],
        "support_touches": int(
            selected_support.get(
                "touches",
                0,
            )
        ),
        "resistance_touches": int(
            selected_resistance.get(
                "touches",
                0,
            )
        ),
    }

def calculate_chart_view(
    current_df: pd.DataFrame,
    current_price: float,
    day_open: float,
    ema20: float,
    ema50: float,
    session_average: float,
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
            score += 1

        elif close_slope < -(atr * 0.025):
            score -= 1

    recent_high = float(
        current_df["High"].tail(5).max()
    )

    recent_low = float(
        current_df["Low"].tail(5).min()
    )

    recent_position = (
        (current_price - recent_low)
        / max(
            recent_high - recent_low,
            0.01,
        )
    )

    if recent_position >= 0.70:
        score += 1

    elif recent_position <= 0.30:
        score -= 1

    if score >= 2:
        return "Bullish", score

    if score <= -2:
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

    df = (
        df
        .sort_values("Datetime")
        .drop_duplicates(
            subset=["Datetime"]
        )
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

    current_df = get_completed_session_candles(
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

    ema20 = float(latest_row["EMA20"])
    ema50 = float(latest_row["EMA50"])
    atr = float(latest_row["ATR"])

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

    levels = calculate_actionable_levels(
        full_dataframe=df,
        current_df=current_df,
        current_price=current_price,
        ema20=ema20,
        atr=atr,
        selected_date=selected_date,
    )

    chart_view, chart_score = (
        calculate_chart_view(
            current_df=current_df,
            current_price=current_price,
            day_open=day_open,
            ema20=ema20,
            ema50=ema50,
            session_average=session_average,
            atr=atr,
        )
    )

    support_level = levels[
        "support_level"
    ]

    resistance_level = levels[
        "resistance_level"
    ]

    is_bank_nifty = current_price > 40000

    if is_bank_nifty:
        bullish_minimum_target = (
            resistance_level + 70
        )

        bullish_maximum_target = (
            resistance_level + 170
        )

        bearish_minimum_target = (
            support_level - 100
        )

        bearish_maximum_target = (
            support_level - 250
        )

    else:
        bullish_minimum_target = (
            resistance_level + 30
        )

        bullish_maximum_target = (
            resistance_level + 80
        )

        bearish_minimum_target = (
            support_level - 60
        )

        bearish_maximum_target = (
            support_level - 130
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

        "support_raw_level": levels[
            "support_raw_level"
        ],
        "resistance_raw_level": levels[
            "resistance_raw_level"
        ],

        "support_sources": levels[
            "support_sources"
        ],
        "resistance_sources": levels[
            "resistance_sources"
        ],

        "support_touches": levels[
            "support_touches"
        ],
        "resistance_touches": levels[
            "resistance_touches"
        ],

        "bullish_trigger": resistance_level,
        "bullish_minimum_target": float(
            bullish_minimum_target
        ),
        "bullish_maximum_target": float(
            bullish_maximum_target
        ),

        "bearish_trigger": support_level,
        "bearish_minimum_target": float(
            bearish_minimum_target
        ),
        "bearish_maximum_target": float(
            bearish_maximum_target
        ),

        "analysis_candle_time": (
            latest_datetime
        ),
    }
