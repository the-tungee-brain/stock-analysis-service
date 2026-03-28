create table app_user (
   id                varchar2(36) default sys_guid() primary key,
   identity_sub      varchar2(255) not null unique,
   identity_provider varchar2(50) default 'google' not null,
   email             varchar2(255) not null unique,
   full_name         varchar2(255),
   avatar_url        varchar2(512),
   created_at        timestamp default systimestamp not null,
   last_login_at     timestamp
);