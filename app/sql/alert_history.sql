create table alert_history (
   id                varchar2(36) default sys_guid() primary key,
   user_id           varchar2(36) not null,
   fingerprint       varchar2(64) not null,
   action            varchar2(64) not null,
   symbol            varchar2(16),
   reason            varchar2(2000) not null,
   priority          number not null,
   status            varchar2(20) default 'active' not null,
   first_seen_at     timestamp default systimestamp not null,
   last_seen_at      timestamp default systimestamp not null,
   resolved_at       timestamp,
   created_at        timestamp default systimestamp not null
);

create index idx_alert_history_user_status
   on alert_history (user_id, status, last_seen_at desc);

create index idx_alert_history_user_fingerprint
   on alert_history (user_id, fingerprint, status);
