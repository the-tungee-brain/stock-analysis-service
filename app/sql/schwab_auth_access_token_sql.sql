create table schwab_auth_access_token (
   id                 number generated always as identity primary key,
   user_id            varchar2(100) not null unique,
   access_token       clob not null,
   refresh_token      clob,
   access_expires_at  timestamp with time zone not null,
   refresh_expires_at timestamp with time zone,
   created_at         timestamp with time zone default systimestamp,
   updated_at         timestamp with time zone default systimestamp
);

create index idx_schwab_token_expires on
   schwab_auth_access_token (
      access_expires_at
   );