declare
   table_count number;
begin
   select count(*)
     into table_count
     from user_tables
    where table_name = 'PROVIDER_SYMBOL_PROFILE';

   if table_count = 0 then
      execute immediate '
         create table provider_symbol_profile (
            provider        varchar2(32) not null,
            symbol          varchar2(32) not null,
            status          varchar2(24) default ''available'' not null,
            fetched_at      timestamp with time zone not null,
            updated_at      timestamp with time zone default systimestamp not null,
            name            varchar2(512),
            currency        varchar2(16),
            exchange_name   varchar2(128),
            quote_type      varchar2(64),
            asset_type      varchar2(64),
            sector          varchar2(256),
            industry        varchar2(256),
            country         varchar2(128),
            website         varchar2(1024),
            current_price   number,
            previous_close  number,
            market_cap      number,
            total_assets    number,
            volume          number,
            avg_volume      number,
            trailing_pe     number,
            forward_pe      number,
            price_to_book   number,
            dividend_yield  number,
            dividend_yield_pct number,
            raw_dividend_yield number,
            raw_dividend_yield_source varchar2(128),
            dividend_rate   number,
            expense_ratio   number,
            beta            number,
            raw_json        clob,
            constraint provider_symbol_profile_pk primary key (provider, symbol)
         )
      ';
   end if;

   for col_rec in (
      select 'DIVIDEND_YIELD_PCT' as name, 'number' as ddl from dual
      union all
      select 'RAW_DIVIDEND_YIELD' as name, 'number' as ddl from dual
      union all
      select 'RAW_DIVIDEND_YIELD_SOURCE' as name, 'varchar2(128)' as ddl from dual
   ) loop
      select count(*)
        into table_count
       from user_tab_columns
       where table_name = 'PROVIDER_SYMBOL_PROFILE'
         and column_name = col_rec.name;

      if table_count = 0 then
         execute immediate
            'alter table provider_symbol_profile add (' ||
            lower(col_rec.name) || ' ' || col_rec.ddl || ')';
      end if;
   end loop;
end;
/
