import pytest
import requests

import utilities.fxmacrodata_calendar as fx


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def test_fxmacrodata_calendar_does_not_follow_redirects(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse({}, status_code=302)

    monkeypatch.setattr(fx.requests, "get", fake_get)
    with pytest.raises(fx.FXMacroDataError, match="HTTP 302"):
        fx.fetch_fxmacrodata_calendar("usd", api_key="test-key")
    assert captured["allow_redirects"] is False
    assert captured["headers"] == {"X-API-Key": "test-key"}


def test_fxmacrodata_calendar_errors_never_include_the_key(monkeypatch):
    def fake_get(url, **kwargs):
        raise requests.ConnectionError(f"failed with {kwargs['headers']}")

    monkeypatch.setattr(fx.requests, "get", fake_get)
    with pytest.raises(fx.FXMacroDataError) as excinfo:
        fx.fetch_fxmacrodata_calendar("usd", api_key="test-key")
    assert "test-key" not in str(excinfo.value)

    with pytest.raises(fx.FXMacroDataError) as excinfo:
        fx.fetch_fxmacrodata_calendar("usd", api_key="test\nkey")
    assert "test" not in str(excinfo.value)


@pytest.mark.parametrize(
    "payload", [{"detail": "Invalid API key"}, [1, 2], {"data": "oops"}, None]
)
def test_fxmacrodata_calendar_rejects_malformed_payloads(monkeypatch, payload):
    monkeypatch.setattr(fx.requests, "get", lambda url, **kwargs: FakeResponse(payload))
    with pytest.raises(fx.FXMacroDataError, match="unexpected response"):
        fx.fetch_fxmacrodata_calendar("usd")
