create table user_strategy_journey (
   id                varchar2(36) default sys_guid() primary key,
   user_id           varchar2(128) not null,
   strategy          varchar2(32) not null,
   current_step_id   varchar2(64),
   steps_json        clob not null,
   started_at        timestamp default systimestamp not null,
   completed_at      timestamp,
   updated_at        timestamp default systimestamp not null,
   constraint uq_user_strategy_journey unique (user_id, strategy)
);

create index idx_user_strategy_journey_user
   on user_strategy_journey (user_id, strategy);
