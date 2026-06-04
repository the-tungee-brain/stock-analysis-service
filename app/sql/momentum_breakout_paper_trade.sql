create table momentum_breakout_paper_trade (
    alert_id                 varchar2(36) primary key,
    user_id                  varchar2(36) not null,
    symbol                   varchar2(16) not null,
    setup_name               varchar2(64) not null,
    signal_date              date not null,
    entry_triggered_at       timestamp with time zone,
    entry_price              number(12, 4) not null,
    stop_price               number(12, 4) not null,
    target_price             number(12, 4) not null,
    exit_at                  timestamp with time zone,
    exit_price               number(12, 4),
    status                   varchar2(32) not null,
    outcome_return_pct       number(12, 6),
    holding_days             number(8),
    risk_gate_action         varchar2(16),
    market_regime            varchar2(16),
    volume_ratio             number(12, 4),
    rs_percentile            number(8, 2),
    created_at               timestamp with time zone default systimestamp not null,
    updated_at               timestamp with time zone default systimestamp not null
);

create index idx_mb_paper_trade_user_created
    on momentum_breakout_paper_trade (user_id, created_at desc);

create index idx_mb_paper_trade_user_status
    on momentum_breakout_paper_trade (user_id, status);

alter table momentum_breakout_alert add (
    market_regime   varchar2(16),
    volume_ratio    number(12, 4),
    rs_percentile   number(8, 2)
);
