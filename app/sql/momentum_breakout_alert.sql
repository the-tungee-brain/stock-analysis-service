create table momentum_breakout_alert (
    alert_id                 varchar2(36) primary key,
    user_id                  varchar2(36) not null,
    symbol                   varchar2(16) not null,
    setup_name               varchar2(64) not null,
    created_at               timestamp with time zone default systimestamp not null,
    signal_date              date not null,
    entry_price              number(12, 4) not null,
    stop_price               number(12, 4) not null,
    target_price             number(12, 4) not null,
    entry_is_stop            number(1) default 1 not null,
    status                   varchar2(32) not null,
    expires_at               timestamp with time zone not null,
    triggered_at             timestamp with time zone,
    exit_at                  timestamp with time zone,
    exit_price               number(12, 4),
    outcome_return_pct       number(12, 6),
    risk_gate_action         varchar2(16),
    risk_gate_reasons        clob,
    historical_win_rate      number(8, 4),
    historical_profit_factor number(12, 4),
    historical_total_trades  number(10),
    updated_at               timestamp with time zone default systimestamp not null
);

create index idx_mb_alert_user_status
    on momentum_breakout_alert (user_id, status, created_at desc);

create index idx_mb_alert_active_symbol
    on momentum_breakout_alert (user_id, symbol, setup_name, status);

create table momentum_breakout_alert_event (
    event_id       varchar2(36) primary key,
    alert_id       varchar2(36) not null,
    user_id        varchar2(36) not null,
    event_type     varchar2(32) not null,
    from_status    varchar2(32),
    to_status      varchar2(32) not null,
    price          number(12, 4),
    recorded_at    timestamp with time zone not null,
    message        varchar2(2000)
);

create index idx_mb_alert_event_alert
    on momentum_breakout_alert_event (alert_id, recorded_at);

alter table momentum_breakout_alert_event
    add constraint fk_mb_alert_event_alert
    foreign key (alert_id) references momentum_breakout_alert (alert_id);
