from datetime import date, datetime, time

import pandas as pd
import streamlit as st

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
st.caption("Angel One 5-Minute Candle Data Test")

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

    return dataframe

if st.button(
    "📥 Fetch Nifty & Bank Nifty Candles",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Angel One se 5-minute candles fetch ho rahe hain..."
        ):
            login_result = login_and_test()
            smart_api = login_result["smart_api"]

            from_datetime = datetime.combine(
                selected_date,
                time(9, 15),
            )

            to_datetime = datetime.combine(
                selected_date,
                selected_time,
            )

            if to_datetime < from_datetime:
                raise AngelAPIError(
                    "Selected time 09:15 AM ke baad hona chahiye."
                )

            from_date_string = from_datetime.strftime(
                "%Y-%m-%d %H:%M"
            )

            to_date_string = to_datetime.strftime(
                "%Y-%m-%d %H:%M"
            )

            nifty_config = INDEX_CONFIG["NIFTY 50"]
            banknifty_config = INDEX_CONFIG["BANK NIFTY"]

            nifty_candles = fetch_candle_data(
                smart_api=smart_api,
                exchange=nifty_config["exchange"],
                symbol_token=nifty_config["symbol_token"],
                from_date=from_date_string,
                to_date=to_date_string,
            )

            banknifty_candles = fetch_candle_data(
                smart_api=smart_api,
                exchange=banknifty_config["exchange"],
                symbol_token=banknifty_config["symbol_token"],
                from_date=from_date_string,
                to_date=to_date_string,
            )

            nifty_df = candles_to_dataframe(nifty_candles)
            banknifty_df = candles_to_dataframe(
                banknifty_candles
            )

        st.success(
            "✅ Nifty aur Bank Nifty candle data received"
        )

        nifty_column, banknifty_column = st.columns(2)

        with nifty_column:
            st.subheader("NIFTY 50")

            latest_nifty = nifty_df.iloc[-1]

            st.metric(
                "Latest Close",
                f'{latest_nifty["Close"]:,.2f}',
            )

            st.write(
                f'Latest Candle: '
                f'{latest_nifty["Datetime"]}'
            )

            st.dataframe(
                nifty_df.tail(10),
                use_container_width=True,
                hide_index=True,
            )

        with banknifty_column:
            st.subheader("BANK NIFTY")

            latest_banknifty = banknifty_df.iloc[-1]

            st.metric(
                "Latest Close",
                f'{latest_banknifty["Close"]:,.2f}',
            )

            st.write(
                f'Latest Candle: '
                f'{latest_banknifty["Datetime"]}'
            )

            st.dataframe(
                banknifty_df.tail(10),
                use_container_width=True,
                hide_index=True,
            )

    except AngelAPIError as exc:
        st.error(f"❌ {exc}")

    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")
