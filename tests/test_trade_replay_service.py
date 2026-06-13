from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user, get_current_user_id
from app.dependencies.service_dependencies import get_trade_replay_service
from app.main import app
from app.models.intraday_trading_bias_models import (
    IntradayTradingBiasAlignment,
    IntradayTradingBiasLevels,
    IntradayTradingBiasResponse,
)
from app.models.trade_replay_models import TradeReplayEvent
from app.services import trade_replay_service as replay_module
from app.services.trade_replay_service import (
    InMemoryTradeReplayStore,
    MissedMoveRecord,
    ReplayBar,
    TradePlanRecord,
    TradeReplayService,
    generate_replay_events,
    plan_signature,
)

ET = ZoneInfo("America/New_York")
SESSION_DATE = date(2026, 6, 5)


class _FakeYFinanceAdapter:
    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars

    def get_history(
        self,
        symbol: str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool = True,
        prepost: bool = False,
    ) -> pd.DataFrame:
        del symbol, period, interval, auto_adjust, prepost
        return self.bars


def _bias_response() -> IntradayTradingBiasResponse:
    return IntradayTradingBiasResponse(
        bias="Neutral",
        confidence="Medium",
        setup_type="RangeDay",
        action="Watch",
        levels=IntradayTradingBiasLevels(
            open_range_high=210.39,
            open_range_low=206.18,
            vwap=208.38,
        ),
        alignment=IntradayTradingBiasAlignment(
            market="mixed",
            intraday_trend="mixed",
            vwap="above",
            volume="neutral",
            catalyst="none",
        ),
        reasons=[],
        warnings=["Delayed/polled yfinance bars; not real-time."],
        data_gaps=[],
        last_updated=datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc),
        staleness_seconds=900,
    )


def _frame(rows: list[tuple[datetime, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [1000 for _ in rows],
        },
        index=pd.DatetimeIndex([row[0] for row in rows]),
    )


def _service(monkeypatch, bars: pd.DataFrame) -> tuple[TradeReplayService, InMemoryTradeReplayStore]:
    monkeypatch.setattr(
        replay_module,
        "build_intraday_trading_bias",
        lambda *args, **kwargs: _bias_response(),
    )
    store = InMemoryTradeReplayStore()
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(bars),  # type: ignore[arg-type]
    )
    return service, store


def _event(
    *,
    event_id: str,
    event_date: date,
    event_type: str,
    event_time: datetime,
    level_price: float | None = None,
    observed_price: float | None = None,
    dedupe_key: str | None = None,
) -> TradeReplayEvent:
    return TradeReplayEvent(
        id=event_id,
        plan_id="plan-1",
        symbol="NVDA",
        event_date=event_date,
        workflow="day_trade",
        event_type=event_type,
        event_time=event_time,
        level_price=level_price,
        observed_price=observed_price,
        message=f"{event_type} message",
        severity="important" if event_type != "stop_hit" else "warning",
        actionability="missed" if event_type != "long_trigger_activated" else "active",
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        dedupe_key=dedupe_key or event_id,
        created_at=datetime(2026, 6, 12, 20, 0, tzinfo=timezone.utc),
    )


def _missed_move(
    *,
    missed_move_id: str,
    event_date: date,
    outcome: str = "target_hit",
    max_move: float | None = 0.04,
    quality: float | None = 2.0,
) -> MissedMoveRecord:
    trigger = _event(
        event_id=f"{missed_move_id}-trigger",
        event_date=event_date,
        event_type="long_trigger_activated",
        event_time=datetime.combine(event_date, datetime.min.time(), tzinfo=ET).replace(hour=10),
        level_price=100.0,
        observed_price=100.0,
    )
    terminal_type = {
        "target_hit": "target_1_hit",
        "extended": "setup_extended",
        "invalidated": "setup_invalidated",
        "stopped": "stop_hit",
    }[outcome]
    terminal = _event(
        event_id=f"{missed_move_id}-terminal",
        event_date=event_date,
        event_type=terminal_type,
        event_time=datetime.combine(event_date, datetime.min.time(), tzinfo=ET).replace(hour=11),
        level_price=104.0,
        observed_price=104.0,
    )
    return MissedMoveRecord(
        missed_move_id=missed_move_id,
        symbol="NVDA",
        workflow="day_trade",
        event_date=event_date,
        setup_type="Long opening range breakout",
        trigger_price=100.0,
        outcome=outcome,  # type: ignore[arg-type]
        max_move_after_trigger_pct=max_move,
        setup_quality_score=quality,
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        trigger_event_id=trigger.id,
        terminal_event_id=terminal.id,
        replay_events=[trigger, terminal],
        created_at=datetime.combine(event_date, datetime.min.time(), tzinfo=timezone.utc),
    )


def test_plan_snapshot_not_duplicated_on_repeated_refresh(monkeypatch) -> None:
    service, store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 210.5, 214.7, 210.4, 214.6),
            ]
        ),
    )

    first = service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    second = service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    third = service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)

    assert first.plan_created is True
    assert second.plan_created is False
    assert third.plan_created is False
    assert len(store.plans) == 1
    assert first.plan_id == second.plan_id
    assert first.plan_id == third.plan_id


def test_trigger_and_target_events_are_emitted_once(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 210.5, 214.7, 210.4, 214.6),
            ]
        ),
    )

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert [event.event_type for event in replay.events].count("long_trigger_activated") == 1
    assert [event.event_type for event in replay.events].count("target_1_hit") == 1


def test_completed_missed_move_is_persisted(monkeypatch) -> None:
    service, store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 210.5, 214.7, 210.4, 214.6),
            ]
        ),
    )
    service.now = datetime(2026, 6, 5, 16, 5, tzinfo=ET)

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)

    assert len(store.missed_moves) == 1
    missed_move = store.missed_moves[0]
    assert missed_move.symbol == "NVDA"
    assert missed_move.outcome == "target_hit"
    assert missed_move.trigger_price == 210.4
    assert missed_move.max_move_after_trigger_pct is not None
    assert [event.event_type for event in missed_move.replay_events] == [
        "long_trigger_activated",
        "target_1_hit",
    ]


def test_etf_day_trade_replay_and_missed_move_generation(monkeypatch) -> None:
    service, store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 210.5, 214.7, 210.4, 214.6),
            ]
        ),
    )
    service.now = datetime(2026, 6, 5, 16, 5, tzinfo=ET)

    for symbol in ("SPY", "QQQ", "IWM", "DIA"):
        refresh = service.refresh(
            symbol=symbol,
            workflow="day_trade",
            event_date=SESSION_DATE,
        )
        replay = service.get_replay(
            symbol=symbol,
            workflow="day_trade",
            event_date=SESSION_DATE,
        )

        assert refresh.plan_created is True
        assert replay.symbol == symbol
        assert replay.events
        assert [event.event_type for event in replay.events] == [
            "long_trigger_activated",
            "target_1_hit",
        ]

    assert [move.symbol for move in store.missed_moves] == ["SPY", "QQQ", "IWM", "DIA"]
    assert all(move.outcome == "target_hit" for move in store.missed_moves)


def test_missed_moves_today_range_uses_eastern_trading_date() -> None:
    store = InMemoryTradeReplayStore()
    store.save_missed_moves(
        [
            _missed_move(missed_move_id="today", event_date=date(2026, 6, 5)),
            _missed_move(missed_move_id="yesterday", event_date=date(2026, 6, 4)),
        ]
    )
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(pd.DataFrame()),  # type: ignore[arg-type]
        now=datetime(2026, 6, 6, 0, 30, tzinfo=timezone.utc),
    )

    summary = service.list_missed_moves(
        symbol="NVDA",
        workflow="day_trade",
        range_="today",
        sort="most_recent",
    )

    assert [row.id for row in summary.rows] == ["today"]


def test_missed_moves_last_5_trading_days_excludes_weekends_and_holidays() -> None:
    store = InMemoryTradeReplayStore()
    store.save_missed_moves(
        [
            _missed_move(missed_move_id="jun29", event_date=date(2026, 6, 29)),
            _missed_move(missed_move_id="jun26", event_date=date(2026, 6, 26)),
            _missed_move(missed_move_id="jun30", event_date=date(2026, 6, 30)),
            _missed_move(missed_move_id="jul01", event_date=date(2026, 7, 1)),
            _missed_move(missed_move_id="jul02", event_date=date(2026, 7, 2)),
            _missed_move(missed_move_id="jul03-holiday", event_date=date(2026, 7, 3)),
            _missed_move(missed_move_id="jul06", event_date=date(2026, 7, 6)),
        ]
    )
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(pd.DataFrame()),  # type: ignore[arg-type]
        now=datetime(2026, 7, 6, 16, 5, tzinfo=ET),
    )

    summary = service.list_missed_moves(
        symbol="NVDA",
        workflow="day_trade",
        range_="last_5_trading_days",
        sort="most_recent",
    )

    assert [row.id for row in summary.rows] == [
        "jul06",
        "jul02",
        "jul01",
        "jun30",
        "jun29",
    ]


def test_missed_moves_sorting() -> None:
    store = InMemoryTradeReplayStore()
    store.save_missed_moves(
        [
            _missed_move(
                missed_move_id="small-high-quality",
                event_date=date(2026, 6, 5),
                max_move=0.02,
                quality=3.0,
            ),
            _missed_move(
                missed_move_id="big-low-quality",
                event_date=date(2026, 6, 4),
                max_move=0.09,
                quality=1.0,
            ),
        ]
    )
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(pd.DataFrame()),  # type: ignore[arg-type]
        now=datetime(2026, 6, 5, 16, 5, tzinfo=ET),
    )

    biggest = service.list_missed_moves(
        symbol="NVDA",
        workflow="day_trade",
        range_="last_5_trading_days",
        sort="biggest_move",
    )
    highest_quality = service.list_missed_moves(
        symbol="NVDA",
        workflow="day_trade",
        range_="last_5_trading_days",
        sort="highest_setup_quality",
    )

    assert [row.id for row in biggest.rows] == [
        "big-low-quality",
        "small-high-quality",
    ]
    assert [row.id for row in highest_quality.rows] == [
        "small-high-quality",
        "big-low-quality",
    ]


def test_replay_timeline_can_be_fetched_by_missed_move_id() -> None:
    store = InMemoryTradeReplayStore()
    store.save_missed_moves(
        [_missed_move(missed_move_id="selected", event_date=SESSION_DATE)]
    )
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(pd.DataFrame()),  # type: ignore[arg-type]
    )

    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
        missed_move_id="selected",
    )

    assert replay.symbol == "NVDA"
    assert [event.event_type for event in replay.events] == [
        "long_trigger_activated",
        "target_1_hit",
    ]


def test_session_label_uses_eastern_market_time() -> None:
    assert replay_module._session_label(  # noqa: SLF001
        datetime(2026, 6, 12, 7, 10, tzinfo=timezone.utc)
    ) == "Pre-market"
    assert replay_module._session_label(  # noqa: SLF001
        datetime(2026, 6, 12, 13, 30, tzinfo=timezone.utc)
    ) == "Regular session"
    assert replay_module._session_label(  # noqa: SLF001
        datetime(2026, 6, 12, 20, 0, tzinfo=timezone.utc)
    ) == "After-hours"


def test_stop_event_is_emitted_once(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 210.4, 210.6, 208.0, 208.1),
            ]
        ),
    )

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert [event.event_type for event in replay.events].count("stop_hit") == 1


def test_delayed_source_label_is_preserved(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
            ]
        ),
    )

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert replay.source == "delayed"
    assert replay.source_freshness_label == replay_module.SOURCE_FRESHNESS_DELAYED
    assert all(event.source_freshness_label == replay_module.SOURCE_FRESHNESS_DELAYED for event in replay.events)


def test_get_returns_chronological_events(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 15, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.5, 214.7, 210.4, 214.6),
            ]
        ),
    )

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert replay.events == sorted(replay.events, key=lambda event: event.event_time)


def test_get_replay_sorts_same_timestamp_target_1_before_target_2() -> None:
    store = InMemoryTradeReplayStore()
    timestamp = datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc)
    store.append_events(
        [
            TradeReplayEvent(
                id="evt-target-2",
                plan_id="plan-1",
                symbol="NVDA",
                event_date=SESSION_DATE,
                workflow="day_trade",
                event_type="target_2_hit",
                event_time=timestamp,
                level_price=108.0,
                observed_price=108.0,
                message="Target 2",
                severity="important",
                actionability="missed",
                source="delayed",
                source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
                dedupe_key="target-2",
            ),
            TradeReplayEvent(
                id="evt-target-1",
                plan_id="plan-1",
                symbol="NVDA",
                event_date=SESSION_DATE,
                workflow="day_trade",
                event_type="target_1_hit",
                event_time=timestamp,
                level_price=104.0,
                observed_price=104.0,
                message="Target 1",
                severity="important",
                actionability="missed",
                source="delayed",
                source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
                dedupe_key="target-1",
            ),
        ]
    )
    service = TradeReplayService(
        store=store,
        yfinance_adapter=_FakeYFinanceAdapter(pd.DataFrame()),  # type: ignore[arg-type]
    )

    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert [event.event_type for event in replay.events] == [
        "target_1_hit",
        "target_2_hit",
    ]


def test_trade_replay_routes_match_frontend_contract(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.8, 210.4),
            ]
        ),
    )

    class _FakeUser:
        identity_sub = "user-1"

    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_trade_replay_service] = lambda: service
    client = TestClient(app)
    try:
        refresh = client.post(
            "/api/v1/research/trade-replay/refresh",
            json={
                "symbol": "nvda",
                "workflow": "day_trade",
                "date": "2026-06-05",
            },
        )
        assert refresh.status_code == 200
        assert refresh.json()["success"] is True

        response = client.get(
            "/api/v1/research/trade-replay",
            params={
                "symbol": "NVDA",
                "workflow": "day_trade",
                "date": "2026-06-05",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "NVDA"
        assert body["workflow"] == "day_trade"
        assert body["source_freshness_label"] == replay_module.SOURCE_FRESHNESS_DELAYED
        assert isinstance(body["events"], list)
    finally:
        app.dependency_overrides.clear()


def test_day_trade_live_replay_long_only_returns_only_long_events(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.5, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 214.0, 215.0, 213.0, 214.8),
                (datetime(2026, 6, 5, 10, 20, tzinfo=ET), 207.0, 207.5, 205.8, 206.0),
                (datetime(2026, 6, 5, 10, 25, tzinfo=ET), 202.0, 203.0, 201.5, 202.0),
            ]
        ),
    )

    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
        direction_mode="long_only",
    )

    assert replay.direction_mode == "long_only"
    assert [event.event_type for event in replay.events] == [
        "long_trigger_activated",
        "target_1_hit",
    ]
    assert all("short" not in event.dedupe_key for event in replay.events)


def test_day_trade_live_replay_short_only_returns_only_short_events(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.5, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 214.0, 215.0, 213.0, 214.8),
                (datetime(2026, 6, 5, 10, 20, tzinfo=ET), 207.0, 207.5, 205.8, 206.0),
                (datetime(2026, 6, 5, 10, 25, tzinfo=ET), 202.0, 203.0, 201.5, 202.0),
            ]
        ),
    )

    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
        direction_mode="short_only",
    )

    assert replay.direction_mode == "short_only"
    assert [event.event_type for event in replay.events] == [
        "short_trigger_activated",
        "target_1_hit",
    ]
    assert all("long" not in event.dedupe_key for event in replay.events)


def test_day_trade_live_replay_long_and_short_can_evaluate_both_sides(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 210.0, 210.5, 209.5, 210.4),
                (datetime(2026, 6, 5, 10, 10, tzinfo=ET), 214.0, 215.0, 213.0, 214.8),
                (datetime(2026, 6, 5, 10, 20, tzinfo=ET), 207.0, 207.5, 205.8, 206.0),
                (datetime(2026, 6, 5, 10, 25, tzinfo=ET), 202.0, 203.0, 201.5, 202.0),
            ]
        ),
    )

    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
        direction_mode="long_and_short",
    )

    assert replay.direction_mode == "long_and_short"
    assert "long_trigger_activated" in [event.event_type for event in replay.events]
    assert "short_trigger_activated" in [event.event_type for event in replay.events]
    assert any("long" in event.dedupe_key for event in replay.events)
    assert any("short" in event.dedupe_key for event in replay.events)


def test_trade_replay_route_rejects_invalid_direction_mode(monkeypatch) -> None:
    service, _store = _service(monkeypatch, _frame([]))

    class _FakeUser:
        identity_sub = "user-1"

    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_trade_replay_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.get(
            "/api/v1/research/trade-replay",
            params={
                "symbol": "NVDA",
                "workflow": "day_trade",
                "date": "2026-06-05",
                "direction_mode": "sideways",
            },
        )
        assert response.status_code == 400
        assert "direction_mode must be one of" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_swing_trade_entry_and_target_events_are_generated() -> None:
    levels = {
        "side": "Long",
        "entry": 100.0,
        "stop": 95.0,
        "target_1": 110.0,
    }
    payload = {"workflow": "swing_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-swing-1",
        symbol="NVDA",
        workflow="swing_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=99.0,
            high=101.0,
            low=98.0,
            close=100.5,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 20, 0, tzinfo=timezone.utc),
            open=106.0,
            high=111.0,
            low=105.0,
            close=110.5,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert [event.event_type for event in events] == [
        "entry_triggered",
        "target_hit",
    ]


def test_day_trade_target_2_event_is_generated_when_level_exists() -> None:
    levels = {
        "long_entry": 100.0,
        "long_stop": 98.0,
        "long_target_1": 104.0,
        "long_target_2": 108.0,
        "open_range_high": 99.99,
        "open_range_low": 96.0,
        "vwap": 98.0,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-day-1",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=99.0,
            high=100.5,
            low=98.5,
            close=100.2,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            open=104.0,
            high=108.5,
            low=103.0,
            close=108.2,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert [event.event_type for event in events] == [
        "long_trigger_activated",
        "target_1_hit",
        "target_2_hit",
    ]
    assert events[1].event_time < events[2].event_time


def test_vwap_flips_are_not_emitted_as_timeline_events() -> None:
    levels = {
        "long_entry": 100.0,
        "long_stop": 98.0,
        "long_target_1": 106.0,
        "open_range_high": 99.99,
        "open_range_low": 96.0,
        "vwap": 101.0,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-day-vwap-1",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=99.0,
            high=100.5,
            low=98.5,
            close=102.0,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            open=102.0,
            high=102.5,
            low=100.0,
            close=100.5,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 30, tzinfo=timezone.utc),
            open=100.5,
            high=103.0,
            low=100.0,
            close=102.5,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert "vwap_lost" not in [event.event_type for event in events]
    assert "vwap_reclaimed" not in [event.event_type for event in events]


def test_day_trade_short_target_uses_low_after_short_trigger() -> None:
    levels = {
        "short_entry": 99.0,
        "short_stop": 101.0,
        "short_target_1": 95.0,
        "open_range_high": 102.0,
        "open_range_low": 99.01,
        "vwap": 101.0,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-day-short-1",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=100.5,
            low=98.5,
            close=98.8,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            open=98.0,
            high=96.0,
            low=94.8,
            close=95.2,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert [event.event_type for event in events] == [
        "short_trigger_activated",
        "target_1_hit",
    ]


def test_target_and_stop_do_not_fire_before_plan_generated_at(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (datetime(2026, 6, 5, 9, 45, tzinfo=ET), 214.0, 215.0, 207.0, 214.5),
                (datetime(2026, 6, 5, 10, 5, tzinfo=ET), 209.0, 209.5, 208.5, 209.2),
            ]
        ),
    )

    service.refresh(symbol="NVDA", workflow="day_trade", event_date=SESSION_DATE)
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=SESSION_DATE,
    )

    assert replay.events == []


def test_same_bar_stop_and_target_is_marked_ambiguous() -> None:
    levels = {
        "long_entry": 100.0,
        "long_stop": 98.0,
        "long_target_1": 104.0,
        "open_range_high": 99.99,
        "open_range_low": 96.0,
        "vwap": 98.0,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-day-ambiguous-1",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=97.5,
            close=101.0,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert [event.event_type for event in events] == [
        "long_trigger_activated",
        "setup_invalidated",
    ]
    assert "ambiguous" in events[-1].message
    assert events[-1].actionability == "invalidated"


def test_duplicate_invalidations_are_collapsed_to_one_story_event() -> None:
    levels = {
        "long_entry": 100.0,
        "long_stop": 98.0,
        "long_target_1": 106.0,
        "open_range_high": 99.99,
        "open_range_low": 96.0,
        "vwap": 98.0,
    }
    payload = {"workflow": "day_trade", "levels": levels}
    plan = TradePlanRecord(
        plan_id="plan-day-invalid-1",
        symbol="NVDA",
        workflow="day_trade",
        plan_date=SESSION_DATE,
        generated_at=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        source="delayed",
        source_freshness_label=replay_module.SOURCE_FRESHNESS_DELAYED,
        signature=plan_signature(levels, payload),
        levels=levels,
        payload=payload,
    )
    bars = [
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            open=100.5,
            high=101.0,
            low=97.5,
            close=97.8,
        ),
        ReplayBar(
            timestamp=datetime(2026, 6, 5, 15, 30, tzinfo=timezone.utc),
            open=97.8,
            high=98.5,
            low=96.0,
            close=97.0,
        ),
    ]

    events = generate_replay_events(plan, bars)

    assert [event.event_type for event in events].count("stop_hit") == 1
    assert [event.event_type for event in events].count("setup_invalidated") == 0


def test_market_date_uses_eastern_session_date(monkeypatch) -> None:
    service, _store = _service(
        monkeypatch,
        _frame(
            [
                (
                    datetime(2026, 6, 9, 0, 30, tzinfo=timezone.utc),
                    210.0,
                    210.5,
                    209.8,
                    210.4,
                ),
            ]
        ),
    )

    service.refresh(
        symbol="NVDA",
        workflow="day_trade",
        event_date=date(2026, 6, 8),
    )
    replay = service.get_replay(
        symbol="NVDA",
        workflow="day_trade",
        event_date=date(2026, 6, 8),
    )

    assert [event.event_type for event in replay.events] == [
        "long_trigger_activated",
    ]


def test_plan_signature_includes_material_metadata_not_runtime_fields() -> None:
    levels = {"entry": 100.0, "stop": 95.0}
    payload = {"workflow": "swing_trade", "risk": {"rMultipleTarget1": 2.0}}
    metadata = {
        "workflow": "swing_trade",
        "plan_date": SESSION_DATE,
        "source": "delayed",
        "source_freshness_label": replay_module.SOURCE_FRESHNESS_DELAYED,
    }

    original = plan_signature(levels, payload, metadata)
    same = plan_signature(dict(levels), dict(payload), dict(metadata))
    changed_date = plan_signature(
        levels,
        payload,
        {**metadata, "plan_date": date(2026, 6, 6)},
    )
    changed_risk = plan_signature(
        levels,
        {"workflow": "swing_trade", "risk": {"rMultipleTarget1": 1.5}},
        metadata,
    )

    assert original == same
    assert original != changed_date
    assert original != changed_risk
