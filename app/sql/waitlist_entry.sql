create table waitlist_entry (
   id                varchar2(36) default sys_guid() primary key,
   identity_sub      varchar2(255) not null unique,
   identity_provider varchar2(50) default 'google' not null,
   email             varchar2(255) not null,
   full_name         varchar2(255),
   avatar_url        varchar2(512),
   status            varchar2(20) default 'waiting' not null,
   created_at        timestamp default systimestamp not null,
   updated_at        timestamp default systimestamp not null
);

create index idx_waitlist_entry_status_created
    on waitlist_entry (status, created_at);
