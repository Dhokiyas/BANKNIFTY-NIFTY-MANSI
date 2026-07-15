from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import analyse_index
from angel_data import (
    AngelAPIError,
    fetch_candle_data,
    login_and_test,
)
from config import INDEX_CONFIG


st.set_page_config(
    page_title="Nifty Bank Nifty Market View",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Nifty & Bank Nifty Market View")
st.caption("5-Minute Chart Analysis — Angel One SmartAPI")


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

    return dataframe.dropna().sort_values(
        "Datetime"
    ).reset_index(drop=True)


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

    level_lines = [
        (
            analysis["resistance_2"],
            "R2",
        ),
        (
            analysis["resistance_1"],
            "R1",
        ),
        (
            analysis["support_1"],
            "S1",
        ),
        (
            analysis["support_2"],
            "S2",
        ),
    ]

    for price, label in level_lines:
        figure.add_hline(
            y=price,
            line_dash="dash",
            annotation_text=f"{label}: {price:,.2f}",
            annotation_position="right",
        )

    figure.update_layout(
        title=title,
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
    )

    return figure


def show_analysis(
    title,
    dataframe,
    analysis,
    selected_date,
):
    st.subheader(title)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Current Price",
            f'{analysis["current_price"]:,.2f}',
        )

    with metric2:
        st.metric(
            "Chart View",
            analysis["chart_view"],
        )

    with metric3:
        st.metric(
            "ATR",
            f'{analysis["atr"]:,.2f}',
        )

    st.plotly_chart(
        create_chart(
            dataframe,
            analysis,
            selected_date,
            title,
        ),
        use_container_width=True,
    )

    support_column, resistance_column = st.columns(2)

    with support_column:
        st.markdown("#### Support Levels")

        st.write(
            f'**S1:** {analysis["support_1"]:,.2f}'
        )

        st.write(
            f'**S2:** {analysis["support_2"]:,.2f}'
        )

        st.write(
            f'**S3:** {analysis["support_3"]:,.2f}'
        )

        st.caption(
            "S1 Source: "
            + ", ".join(
                analysis["support_sources"]
            )
        )

    with resistance_column:
        st.markdown("#### Resistance Levels")

        st.write(
            f'**R1:** {analysis["resistance_1"]:,.2f}'
        )

        st.write(
            f'**R2:** {analysis["resistance_2"]:,.2f}'
        )

        st.write(
            f'**R3:** {analysis["resistance_3"]:,.2f}'
        )

        st.caption(
            "R1 Source: "
            + ", ".join(
                analysis["resistance_sources"]
            )
        )

    st.markdown("#### Breakout Plan")

    bullish_column, bearish_column = st.columns(2)

    with bullish_column:
        st.success(
            f'Above {analysis["bullish_trigger"]:,.2f}\n\n'
            f'Minimum Target: '
            f'{analysis["bullish_minimum_target"]:,.2f}\n\n'
            f'Maximum Target: '
            f'{analysis["bullish_maximum_target"]:,.2f}'
        )

    with bearish_column:
        st.error(
            f'Below {analysis["bearish_trigger"]:,.2f}\n\n'
            f'Minimum Target: '
            f'{analysis["bearish_minimum_target"]:,.2f}\n\n'
            f'Maximum Target: '
            f'{analysis["bearish_maximum_target"]:,.2f}'
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
    "🔍 Generate Nifty & Bank Nifty View",
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

        if selected_datetime < market_open:
            raise AngelAPIError(
                "Selected time 09:15 AM ke baad hona chahiye."
            )

        with st.spinner(
            "Candles fetch aur chart analysis ho raha hai..."
        ):
            login_result = login_and_test()
            smart_api = login_result["smart_api"]

            lookback_start = datetime.combine(
                selected_date - timedelta(days=5),
                time(9, 15),
            )

            from_date_string = lookback_start.strftime(
                "%Y-%m-%d %H:%M"
            )

            to_date_string = selected_datetime.strftime(
                "%Y-%m-%d %H:%M"
            )

            analysis_results = {}

            for index_name, index_config in INDEX_CONFIG.items():
                candles = fetch_candle_data(
                    smart_api=smart_api,
                    exchange=index_config["exchange"],
                    symbol_token=index_config["symbol_token"],
                    from_date=from_date_string,
                    to_date=to_date_string,
                )

                dataframe = candles_to_dataframe(candles)

                analysis = analyse_index(
                    dataframe,
                    selected_date,
                )

                analysis_results[index_name] = {
                    "dataframe": dataframe,
                    "analysis": analysis,
                }

        st.success(
            "✅ Chart analysis successfully generated"
        )

        nifty_column, banknifty_column = st.columns(2)

        with nifty_column:
            show_analysis(
                "NIFTY 50",
                analysis_results["NIFTY 50"]["dataframe"],
                analysis_results["NIFTY 50"]["analysis"],
                selected_date,
            )

        with banknifty_column:
            show_analysis(
                "BANK NIFTY",
                analysis_results["BANK NIFTY"]["dataframe"],
                analysis_results["BANK NIFTY"]["analysis"],
                selected_date,
            )

        st.caption(
            "Breakout/breakdown tabhi confirm maana jayega "
            "jab 5-minute candle trigger level ke upar ya neeche close ho."
        )

    except AngelAPIError as exc:
        st.error(f"❌ {exc}")

    except ValueError as exc:
        st.error(f"❌ {exc}")

    except Exception as exc:
        st.exception(exc)
