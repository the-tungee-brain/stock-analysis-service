create table portfolio_snapshot (
   id                varchar2(36) default sys_guid() primary key,
   user_id           varchar2(36) not null,
   snapshot_date     date not null,
   account_number    varchar2(32),
   liquidation_value number,
   cash_balance      number,
   positions_json    clob not null,
   summary_json      clob,
   created_at        timestamp default systimestamp not null,
   constraint uq_portfolio_snapshot_user_date unique (user_id, snapshot_date)
);

create index idx_portfolio_snapshot_user_date
   on portfolio_snapshot (user_id, snapshot_date desc);
