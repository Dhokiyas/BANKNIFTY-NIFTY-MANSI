import os
import time
from typing import Any

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv()

class AngelAPIError(Exception):
    """Angel One connection related error."""

def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise AngelAPIError(
            f"{name} missing hai. .env file ya Codespaces Secret mein add karo."
        )

    return value

def login_and_test() -> dict[str, Any]:
    api_key = get_required_env("ANGEL_API_KEY")
    client_code = get_required_env("ANGEL_CLIENT_CODE")
    pin = get_required_env("ANGEL_PIN")
    totp_secret = get_required_env("ANGEL_TOTP_SECRET")

    try:
        current_totp = pyotp.TOTP(totp_secret).now()
    except Exception as exc:
        raise AngelAPIError("TOTP secret invalid hai.") from exc

    try:
        smart_api = SmartConnect(api_key)

        session = smart_api.generateSession(
            client_code,
            pin,
            current_totp,
        )

        if not session or session.get("status") is not True:
            message = (
                session.get("message", "Angel One login failed")
                if isinstance(session, dict)
                else "Angel One login failed"
            )
            raise AngelAPIError(message)

        session_data = session.get("data") or {}
        refresh_token = session_data.get("refreshToken")

        if not refresh_token:
            raise AngelAPIError("Refresh token receive nahi hua.")

        feed_token = smart_api.getfeedToken()

        profile_response = smart_api.getProfile(refresh_token)

        if not profile_response or profile_response.get("status") is not True:
            message = (
                profile_response.get("message", "Profile fetch failed")
                if isinstance(profile_response, dict)
                else "Profile fetch failed"
            )
            raise AngelAPIError(message)

        profile = profile_response.get("data") or {}

        return {
            "connected": True,
            "client_code": profile.get("clientcode", client_code),
            "name": profile.get("name", "Angel One User"),
            "exchanges": profile.get("exchanges", []),
            "feed_token_received": bool(feed_token),
            "smart_api": smart_api,
        }

    except AngelAPIError:
        raise

    except Exception as exc:
        raise AngelAPIError(f"Angel One connection error: {exc}") from exc

def fetch_candle_data(
    smart_api,
    exchange: str,
    symbol_token: str,
    from_date: str,
    to_date: str,
    interval: str = "FIVE_MINUTE",
    max_retries: int = 3,
    retry_delay_seconds: float = 1.5,
):
    """Angel One se historical candle data fetch karta hai."""

    candle_params = {
        "exchange": exchange,
        "symboltoken": symbol_token,
        "interval": interval,
        "fromdate": from_date,
        "todate": to_date,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = smart_api.getCandleData(candle_params)

            if not response or response.get("status") is not True:
                message = (
                    response.get("message", "Candle data fetch failed")
                    if isinstance(response, dict)
                    else "Candle data fetch failed"
                )

                normalized_message = str(message).lower()
                is_rate_limited = (
                    "access denied" in normalized_message
                    and "access rate" in normalized_message
                )

                if is_rate_limited and attempt < max_retries:
                    time.sleep(retry_delay_seconds * attempt)
                    continue

                if is_rate_limited:
                    raise AngelAPIError(
                        "Angel One API rate limit hit hua. "
                        "30-60 seconds wait karke dubara try karo."
                    )

                raise AngelAPIError(message)

            candles = response.get("data") or []

            if not candles:
                raise AngelAPIError(
                    "Selected date/time ke liye candle data nahi mila."
                )

            return candles

        except AngelAPIError:
            raise

        except Exception as exc:
            normalized_error = str(exc).lower()
            is_rate_limited = (
                "access denied" in normalized_error
                and "access rate" in normalized_error
            ) or "couldn't parse the json response" in normalized_error

            if is_rate_limited and attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)
                continue

            if is_rate_limited:
                raise AngelAPIError(
                    "Angel One API rate limit hit hua. "
                    "30-60 seconds wait karke dubara try karo."
                ) from exc

            raise AngelAPIError(
                f"Candle data connection error: {exc}"
            ) from exc

    raise AngelAPIError(
        "Candle data retry limit exceed ho gayi. "
        "Please kuch der baad dubara try karo."
    )
