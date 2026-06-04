from unittest.mock import MagicMock

from app.adapters.schwab.schwab_auth import SchwabAuth


def _successful_response():
    response = MagicMock()
    response.status_code = 200
    response.text = '{"access_token":"ok"}'
    return response


def test_refresh_token_request_does_not_print_secrets(monkeypatch, capsys):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return _successful_response()

    monkeypatch.setattr("app.adapters.schwab.schwab_auth.requests.post", fake_post)

    auth = SchwabAuth(
        client_id="client-id-secret",
        client_secret="client-secret-value",
        redirect_uri="https://example.com/callback",
    )

    auth.get_refreshed_access_token("refresh-token-secret")

    captured = capsys.readouterr()
    output = f"{captured.out}{captured.err}"

    assert "refresh-token-secret" not in output
    assert "client-secret-value" not in output
    assert "Authorization" not in output
    assert "Basic" not in output
    assert calls


def test_refresh_token_request_uses_timeout(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return _successful_response()

    monkeypatch.setattr("app.adapters.schwab.schwab_auth.requests.post", fake_post)

    auth = SchwabAuth(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.com/callback",
    )

    auth.get_refreshed_access_token("refresh-token")

    assert calls[0][1]["timeout"] == SchwabAuth.TOKEN_REQUEST_TIMEOUT_SECONDS


def test_authorization_code_request_uses_timeout(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return _successful_response()

    monkeypatch.setattr("app.adapters.schwab.schwab_auth.requests.post", fake_post)

    auth = SchwabAuth(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.com/callback",
    )

    auth.get_access_token("auth-code")

    assert calls[0][1]["timeout"] == SchwabAuth.TOKEN_REQUEST_TIMEOUT_SECONDS
