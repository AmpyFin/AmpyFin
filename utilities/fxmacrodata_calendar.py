"""FXMacroData release-calendar utility for strategy features."""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


class FXMacroDataError(RuntimeError):
    """Raised when the FXMacroData API returns an error or an unexpected payload."""


def fetch_fxmacrodata_calendar(
    currency: str = "usd",
    *,
    limit: int = 50,
    min_tier: Optional[int] = 2,
    api_key: Optional[str] = None,
    base_url: str = FXMACRODATA_BASE_URL,
) -> list[dict[str, Any]]:
    """Fetch official macro release events for AmpyFin strategies."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    headers: dict[str, str] = {}
    token = (api_key or os.getenv("FXMACRODATA_API_KEY") or "").strip()
    if token:
        if any(ord(char) < 33 or ord(char) == 127 for char in token):
            raise FXMacroDataError(
                "FXMacroData API key contains whitespace or control characters"
            )
        headers["X-API-Key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}"
    try:
        # Redirects are not followed so the key is never sent to another host.
        response = requests.get(
            url, params=params, headers=headers, timeout=20, allow_redirects=False
        )
    except requests.RequestException as exc:
        raise FXMacroDataError(
            f"FXMacroData request failed: {type(exc).__name__}"
        ) from None
    if response.status_code != 200:
        raise FXMacroDataError(
            f"FXMacroData request failed with HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError:
        raise FXMacroDataError("FXMacroData returned a non-JSON response") from None
    events = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise FXMacroDataError(
            "FXMacroData returned an unexpected response: "
            f"{detail or 'missing data list'}"
        )
    events = [event for event in events if isinstance(event, dict)]
    if min_tier is None:
        return events[:limit_count]

    return [
        event for event in events if int(event.get("market_tier") or 99) <= min_tier
    ][:limit_count]


def release_dates(events: list[dict[str, Any]]) -> set[str]:
    """Return the release-date set for calendar-aware entry filters."""

    return {event["date"] for event in events if event.get("date")}
