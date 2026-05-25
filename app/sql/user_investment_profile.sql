create table user_investment_profile (
   user_id                   varchar2(128) primary key,
   primary_strategy          varchar2(32),
   risk_tolerance            varchar2(16) default 'moderate' not null,
   options_experience        varchar2(16) default 'beginner' not null,
   income_vs_growth          varchar2(16) default 'balanced' not null,
   config_json               clob,
   onboarding_completed_at   timestamp,
   created_at                timestamp default systimestamp not null,
   updated_at                timestamp default systimestamp not null
);

create index idx_user_investment_profile_strategy
   on user_investment_profile (primary_strategy);
