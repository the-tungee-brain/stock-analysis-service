import os

from app.core.access_control import max_active_users


def test_max_active_users_defaults_to_five(monkeypatch):
    monkeypatch.delenv("MAX_ACTIVE_USERS", raising=False)
    assert max_active_users() == 5


def test_max_active_users_reads_env(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_USERS", "12")
    assert max_active_users() == 12


def test_max_active_users_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_USERS", "not-a-number")
    assert max_active_users() == 5


def test_max_active_users_never_below_one(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_USERS", "0")
    assert max_active_users() == 1
