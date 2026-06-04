declare
   table_count number;
begin
   select count(*)
     into table_count
     from user_tables
    where table_name = 'WATCHLIST_WORKSPACE';

   if table_count = 0 then
      execute immediate '
         create table watchlist_workspace (
            user_id     varchar2(64) primary key,
            version     number default 0 not null,
            created_at  timestamp with time zone default systimestamp not null,
            updated_at  timestamp with time zone default systimestamp not null
         )
      ';
   end if;
end;
/
