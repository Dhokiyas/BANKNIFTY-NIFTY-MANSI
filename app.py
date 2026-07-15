from datetime import date, datetime, time, timedelta
import time as time_module

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from analyzer import analyse_index
from angel_data import (
    AngelAPIError,
    fetch_candle_data,
    login_and_test,
)
from config import (
    BANKNIFTY_TOP_STOCKS,
    INDEX_CONFIG,
    NIFTY_TOP_STOCKS,
)
from stock_confirmation import (
    analyse_stock,
    calculate_majority,
    combine_final_sentiment,
)

st.set_page_config(
    page_title="Nifty Bank Nifty Market View",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Nifty & Bank Nifty Market View")
st.caption(
    "5-Minute Chart Analysis + Top Weightage Stocks Confirmation"
)

def candles_to_dataframe(candles):
    dataframe = pd.DataFrame(
        candles,
        columns=[
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    )

    dataframe["Datetime"] = pd.to_datetime(
        dataframe["Datetime"]
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    dataframe[numeric_columns] = dataframe[
        numeric_columns
    ].apply(pd.to_numeric, errors="coerce")

    return (
        dataframe
        .dropna()
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

def fetch_dataframe_with_retry(
    smart_api,
    symbol_token,
    from_date_string,
    to_date_string,
):
    last_error = None

    for attempt in range(3):
        try:
            candles = fetch_candle_data(
                smart_api=smart_api,
                exchange="NSE",
                symbol_token=symbol_token,
                from_date=from_date_string,
                to_date=to_date_string,
            )

            return candles_to_dataframe(candles)

        except AngelAPIError as exc:
            last_error = exc
            error_text = str(exc).lower()

            rate_limit_error = any(
                phrase in error_text
                for phrase in [
                    "rate",
                    "exceed",
                    "access denied",
                    "too many",
                ]
            )

            if rate_limit_error and attempt < 2:
                time_module.sleep(1.5 * (attempt + 1))
                continue

            raise

    raise AngelAPIError(str(last_error))

def create_chart(
    dataframe,
    analysis,
    selected_date,
    title,
):
    chart_df = dataframe[
        dataframe["Datetime"].dt.date == selected_date
    ].copy()

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=chart_df["Datetime"],
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name=title,
            )
        ]
    )

    levels = [
        (analysis["resistance_2"], "R2"),
        (analysis["resistance_1"], "R1"),
        (analysis["support_1"], "S1"),
        (analysis["support_2"], "S2"),
    ]

    for level_price, label in levels:
        figure.add_hline(
            y=level_price,
            line_dash="dash",
            annotation_text=(
                f"{label}: {level_price:,.2f}"
            ),
            annotation_position="right",
        )

    figure.update_layout(
        title=title,
        height=390,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=20,
            t=45,
            b=10,
        ),
    )

    return figure


def show_sentiment_meter(sentiment):
    sentiment_config = {
        "Bearish": {
            "angle": -72,
            "score": 10,
            "color": "#B91C1C",
        },
        "Neutral Bearish": {
            "angle": -36,
            "score": 30,
            "color": "#F97316",
        },
        "Sideways": {
            "angle": 0,
            "score": 50,
            "color": "#64748B",
        },
        "Neutral Bullish": {
            "angle": 36,
            "score": 70,
            "color": "#84CC16",
        },
        "Bullish": {
            "angle": 72,
            "score": 90,
            "color": "#16A34A",
        },
    }

    selected = sentiment_config.get(
        sentiment,
        sentiment_config["Sideways"],
    )

    angle = selected["angle"]
    score = selected["score"]
    active_color = selected["color"]

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;
            }

            .sentiment-card {
                width: 100%;
                padding: 8px 4px 0 4px;
                text-align: center;
            }

            .gauge-title {
                color: #111827;
                font-size: 17px;
                font-weight: 700;
                margin-bottom: 2px;
            }

            .gauge-subtitle {
                color: #64748B;
                font-size: 12px;
                margin-bottom: 4px;
            }

            .gauge-wrap {
                position: relative;
                width: 520px;
                max-width: 100%;
                height: 255px;
                margin: 0 auto;
                overflow: hidden;
            }

            .gauge-arc {
                position: absolute;
                z-index: 1;
                width: 460px;
                height: 460px;
                left: 50%;
                top: 20px;
                transform: translateX(-50%);
                border-radius: 50%;

                background: conic-gradient(
                    from 270deg,
                    #B91C1C 0deg,
                    #B91C1C 36deg,

                    #F97316 36deg,
                    #F97316 72deg,

                    #94A3B8 72deg,
                    #94A3B8 108deg,

                    #84CC16 108deg,
                    #84CC16 144deg,

                    #16A34A 144deg,
                    #16A34A 180deg,

                    transparent 180deg,
                    transparent 360deg
                );

                box-shadow:
                    0 2px 8px rgba(15, 23, 42, 0.15);
            }

            .gauge-inner {
                position: absolute;
                z-index: 2;
                width: 330px;
                height: 330px;
                left: 50%;
                top: 85px;
                transform: translateX(-50%);
                border-radius: 50%;
                background: #FFFFFF;
            }

            .needle {
                position: absolute;
                z-index: 4;
                width: 7px;
                height: 142px;
                left: calc(50% - 3.5px);
                bottom: 5px;
                border-radius: 8px;
                background: linear-gradient(
                    to top,
                    #111827,
                    __ACTIVE_COLOR__
                );
                transform-origin: 50% 100%;
                transform: rotate(__ANGLE__deg);
                transition: transform 0.8s ease-in-out;
                box-shadow:
                    0 2px 5px rgba(15, 23, 42, 0.35);
            }

            .needle-hub {
                position: absolute;
                z-index: 5;
                width: 30px;
                height: 30px;
                left: calc(50% - 15px);
                bottom: -10px;
                border: 6px solid #FFFFFF;
                border-radius: 50%;
                background: #111827;
                box-shadow:
                    0 2px 7px rgba(15, 23, 42, 0.30);
            }

            .zone-label {
                position: absolute;
                z-index: 6;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 0.3px;
                white-space: nowrap;
                color: #FFFFFF;
                text-shadow:
                    0 1px 3px rgba(0, 0, 0, 0.45);
            }

            .zone-bearish {
                left: 5%;
                top: 163px;
                transform: rotate(-68deg);
            }

            .zone-neutral-bearish {
                left: 15%;
                top: 70px;
                transform: rotate(-38deg);
            }

            .zone-sideways {
                left: 50%;
                top: 31px;
                transform: translateX(-50%);
            }

            .zone-neutral-bullish {
                right: 13%;
                top: 70px;
                transform: rotate(38deg);
            }

            .zone-bullish {
                right: 5%;
                top: 163px;
                transform: rotate(68deg);
            }

            .score-box {
                margin-top: 5px;
                font-size: 13px;
                color: #64748B;
            }

            .score-value {
                color: __ACTIVE_COLOR__;
                font-size: 32px;
                font-weight: 800;
                line-height: 35px;
            }

            .sentiment-result {
                display: inline-block;
                margin-top: 7px;
                padding: 8px 24px;
                color: #FFFFFF;
                background: __ACTIVE_COLOR__;
                border-radius: 999px;
                font-size: 15px;
                font-weight: 800;
                box-shadow:
                    0 3px 8px rgba(15, 23, 42, 0.16);
            }

            @media (max-width: 600px) {
                .gauge-wrap {
                    transform: scale(0.88);
                    transform-origin: top center;
                    margin-bottom: -30px;
                }

                .zone-label {
                    font-size: 9px;
                }
            }
        </style>
    </head>

    <body>
        <div class="sentiment-card">
            <div class="gauge-title">
                Market Sentiment Meter
            </div>

            <div class="gauge-subtitle">
                Chart View + Weightage Stocks Confirmation
            </div>

            <div class="gauge-wrap">
                <div class="gauge-arc"></div>
                <div class="gauge-inner"></div>

                <div class="zone-label zone-bearish">
                    BEARISH
                </div>

                <div class="zone-label zone-neutral-bearish">
                    NEUTRAL BEARISH
                </div>

                <div class="zone-label zone-sideways">
                    SIDEWAYS
                </div>

                <div class="zone-label zone-neutral-bullish">
                    NEUTRAL BULLISH
                </div>

                <div class="zone-label zone-bullish">
                    BULLISH
                </div>

                <div class="needle"></div>
                <div class="needle-hub"></div>
            </div>

            <div class="score-box">
                Sentiment Score
            </div>

            <div class="score-value">
                __SCORE__/100
            </div>

            <div class="sentiment-result">
                __SENTIMENT__
            </div>
        </div>
    </body>
    </html>
    """

    html_code = (
        html_code
        .replace("__ANGLE__", str(angle))
        .replace("__SCORE__", str(score))
        .replace("__SENTIMENT__", sentiment.upper())
        .replace("__ACTIVE_COLOR__", active_color)
    )

    components.html(
        html_code,
        height=385,
        scrolling=False,
    )


def show_market_panel(
    title,
    dataframe,
    chart_analysis,
    stock_confirmation,
    final_sentiment,
    selected_date,
):
    st.subheader(title)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Current Price",
            f'{chart_analysis["current_price"]:,.2f}',
        )

    with metric2:
        st.metric(
            "Chart View",
            chart_analysis["chart_view"],
        )

    with metric3:
        st.metric(
            "Stocks View",
            stock_confirmation["stock_view"],
        )

    st.markdown("#### Final Sentiment")

    show_sentiment_meter(final_sentiment)

    st.info(
        f"Final Sentiment: **{final_sentiment}**"
    )

    st.plotly_chart(
        create_chart(
            dataframe=dataframe,
            analysis=chart_analysis,
            selected_date=selected_date,
            title=title,
        ),
        use_container_width=True,
    )

    support_column, resistance_column = st.columns(2)

    with support_column:
        st.markdown("#### Support")

        st.write(
            f'**S1:** {chart_analysis["support_1"]:,.2f}'
        )
        st.write(
            f'**S2:** {chart_analysis["support_2"]:,.2f}'
        )
        st.write(
            f'**S3:** {chart_analysis["support_3"]:,.2f}'
        )

    with resistance_column:
        st.markdown("#### Resistance")

        st.write(
            f'**R1:** {chart_analysis["resistance_1"]:,.2f}'
        )
        st.write(
            f'**R2:** {chart_analysis["resistance_2"]:,.2f}'
        )
        st.write(
            f'**R3:** {chart_analysis["resistance_3"]:,.2f}'
        )

    st.markdown("#### Breakout / Breakdown Plan")

    bullish_column, bearish_column = st.columns(2)

    with bullish_column:
        st.success(
            f'**Above:** '
            f'{chart_analysis["bullish_trigger"]:,.2f}\n\n'
            f'**Minimum:** '
            f'{chart_analysis["bullish_minimum_target"]:,.2f}\n\n'
            f'**Maximum:** '
            f'{chart_analysis["bullish_maximum_target"]:,.2f}'
        )

    with bearish_column:
        st.error(
            f'**Below:** '
            f'{chart_analysis["bearish_trigger"]:,.2f}\n\n'
            f'**Minimum:** '
            f'{chart_analysis["bearish_minimum_target"]:,.2f}\n\n'
            f'**Maximum:** '
            f'{chart_analysis["bearish_maximum_target"]:,.2f}'
        )

def show_stock_table(
    title,
    stock_results,
    confirmation,
):
    st.subheader(title)

    count1, count2, count3, count4 = st.columns(4)

    with count1:
        st.metric(
            "Final Stock View",
            confirmation["stock_view"],
        )

    with count2:
        st.metric(
            "Bullish",
            confirmation["bullish"],
        )

    with count3:
        st.metric(
            "Bearish",
            confirmation["bearish"],
        )

    with count4:
        st.metric(
            "Sideways",
            confirmation["sideways"],
        )

    table_rows = []

    for result in stock_results:
        table_rows.append(
            {
                "Stock": result["name"],
                "Symbol": result["symbol"],
                "LTP": round(
                    result["current_price"],
                    2,
                ),
                "Day %": round(
                    result["day_change_percentage"],
                    2,
                ),
                "VWAP": round(
                    result["vwap"],
                    2,
                ),
                "Score": result["score"],
                "Signal": result["signal"],
            }
        )

    table_dataframe = pd.DataFrame(table_rows)

    st.dataframe(
        table_dataframe,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LTP": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "Day %": st.column_config.NumberColumn(
                format="%.2f%%",
            ),
            "VWAP": st.column_config.NumberColumn(
                format="%.2f",
            ),
        },
    )

    st.caption(
        "Bullish majority tabhi maana jayega jab "
        "50% se zyada stocks Bullish hon. "
        "Bearish ke liye bhi same rule hai. "
        "Tie ya mixed result Sideways rahega."
    )

control1, control2 = st.columns(2)

with control1:
    selected_date = st.date_input(
        "Select Date",
        value=date.today(),
    )

with control2:
    selected_time = st.time_input(
        "Select Time",
        value=datetime.now().time().replace(
            second=0,
            microsecond=0,
        ),
        step=300,
    )

if st.button(
    "🔍 Generate Complete Market View",
    type="primary",
    use_container_width=True,
):
    try:
        selected_datetime = datetime.combine(
            selected_date,
            selected_time,
        )

        market_open = datetime.combine(
            selected_date,
            time(9, 15),
        )

        market_close = datetime.combine(
            selected_date,
            time(15, 30),
        )

        if selected_datetime < market_open:
            raise AngelAPIError(
                "Selected time 09:15 AM ke baad hona chahiye."
            )

        selected_datetime = min(
            selected_datetime,
            market_close,
        )

        lookback_start = datetime.combine(
            selected_date - timedelta(days=7),
            time(9, 15),
        )

        from_date_string = lookback_start.strftime(
            "%Y-%m-%d %H:%M"
        )

        to_date_string = selected_datetime.strftime(
            "%Y-%m-%d %H:%M"
        )

        progress = st.progress(
            0,
            text="Angel One login ho raha hai...",
        )

        login_result = login_and_test()
        smart_api = login_result["smart_api"]

        index_results = {}

        index_items = list(INDEX_CONFIG.items())

        for index_number, (
            index_name,
            index_config,
        ) in enumerate(index_items):
            progress.progress(
                5 + (index_number * 5),
                text=f"{index_name} candles fetch ho rahe hain...",
            )

            dataframe = fetch_dataframe_with_retry(
                smart_api=smart_api,
                symbol_token=index_config[
                    "symbol_token"
                ],
                from_date_string=from_date_string,
                to_date_string=to_date_string,
            )

            chart_analysis = analyse_index(
                dataframe=dataframe,
                selected_date=selected_date,
            )

            index_results[index_name] = {
                "dataframe": dataframe,
                "analysis": chart_analysis,
            }

            time_module.sleep(0.50)

        unique_stocks = {}

        for stock in (
            NIFTY_TOP_STOCKS
            + BANKNIFTY_TOP_STOCKS
        ):
            unique_stocks[stock["symbol"]] = stock

        stock_signal_cache = {}
        unique_stock_list = list(unique_stocks.values())

        for stock_number, stock in enumerate(
            unique_stock_list,
            start=1,
        ):
            progress_value = int(
                15
                + (
                    stock_number
                    / len(unique_stock_list)
                )
                * 75
            )

            progress.progress(
                progress_value,
                text=(
                    f'{stock["name"]} scan ho raha hai '
                    f'({stock_number}/{len(unique_stock_list)})'
                ),
            )

            stock_dataframe = fetch_dataframe_with_retry(
                smart_api=smart_api,
                symbol_token=stock["token"],
                from_date_string=from_date_string,
                to_date_string=to_date_string,
            )

            stock_analysis = analyse_stock(
                dataframe=stock_dataframe,
                selected_date=selected_date,
            )

            stock_signal_cache[stock["symbol"]] = {
                **stock,
                **stock_analysis,
            }

            time_module.sleep(0.50)

        nifty_stock_results = [
            stock_signal_cache[stock["symbol"]]
            for stock in NIFTY_TOP_STOCKS
        ]

        banknifty_stock_results = [
            stock_signal_cache[stock["symbol"]]
            for stock in BANKNIFTY_TOP_STOCKS
        ]

        nifty_confirmation = calculate_majority(
            nifty_stock_results
        )

        banknifty_confirmation = calculate_majority(
            banknifty_stock_results
        )

        nifty_final_sentiment = combine_final_sentiment(
            chart_view=index_results[
                "NIFTY 50"
            ]["analysis"]["chart_view"],
            stock_view=nifty_confirmation["stock_view"],
        )

        banknifty_final_sentiment = combine_final_sentiment(
            chart_view=index_results[
                "BANK NIFTY"
            ]["analysis"]["chart_view"],
            stock_view=banknifty_confirmation[
                "stock_view"
            ],
        )

        progress.progress(
            100,
            text="Complete analysis ready hai.",
        )

        time_module.sleep(0.25)
        progress.empty()

        st.success(
            "✅ Chart + stocks confirmation successfully generated"
        )

        st.caption(
            f'Analysis Time: '
            f'{selected_datetime.strftime("%d-%m-%Y %I:%M %p")}'
        )

        nifty_column, banknifty_column = st.columns(2)

        with nifty_column:
            show_market_panel(
                title="NIFTY 50",
                dataframe=index_results[
                    "NIFTY 50"
                ]["dataframe"],
                chart_analysis=index_results[
                    "NIFTY 50"
                ]["analysis"],
                stock_confirmation=nifty_confirmation,
                final_sentiment=nifty_final_sentiment,
                selected_date=selected_date,
            )

        with banknifty_column:
            show_market_panel(
                title="BANK NIFTY",
                dataframe=index_results[
                    "BANK NIFTY"
                ]["dataframe"],
                chart_analysis=index_results[
                    "BANK NIFTY"
                ]["analysis"],
                stock_confirmation=banknifty_confirmation,
                final_sentiment=banknifty_final_sentiment,
                selected_date=selected_date,
            )

        st.divider()
        st.header("📋 Stocks Double Confirmation")

        nifty_table_column, bank_table_column = st.columns(2)

        with nifty_table_column:
            show_stock_table(
                title="Nifty Top 10 Stocks",
                stock_results=nifty_stock_results,
                confirmation=nifty_confirmation,
            )

        with bank_table_column:
            show_stock_table(
                title="Bank Nifty Top 6 Stocks",
                stock_results=banknifty_stock_results,
                confirmation=banknifty_confirmation,
            )

        st.warning(
            "Breakout/breakdown tabhi confirmed maana jayega "
            "jab completed 5-minute candle trigger level ke "
            "upar ya neeche close ho."
        )

    except AngelAPIError as exc:
        st.error(f"❌ {exc}")

    except ValueError as exc:
        st.error(f"❌ {exc}")

    except Exception as exc:
        st.exception(exc)
