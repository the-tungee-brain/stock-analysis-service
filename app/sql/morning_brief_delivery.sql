create table morning_brief_delivery (
   id                varchar2(36) default sys_guid() primary key,
   user_id           varchar2(36) not null,
   delivery_date     date not null,
   email             varchar2(255) not null,
   status            varchar2(20) default 'sent' not null,
   error_message     varchar2(2000),
   created_at        timestamp default systimestamp not null,
   constraint uq_morning_brief_delivery_user_date unique (user_id, delivery_date)
);

create index idx_morning_brief_delivery_date
   on morning_brief_delivery (delivery_date desc);
