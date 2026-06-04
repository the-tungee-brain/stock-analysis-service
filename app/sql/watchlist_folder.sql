create table watchlist_folder (
   id            varchar2(36) primary key,
   user_id       varchar2(64) not null,
   name          varchar2(120) not null,
   icon_name     varchar2(80) default 'folder.fill' not null,
   swatch_id     varchar2(32) default 'slate' not null,
   accent_hex    number,
   is_pinned     number(1) default 0 not null,
   is_collapsed  number(1) default 0 not null,
   sort_order    number default 0 not null,
   created_at    timestamp with time zone default systimestamp not null,
   updated_at    timestamp with time zone default systimestamp not null
);

create index idx_watchlist_folder_user on
   watchlist_folder (
      user_id,
      sort_order
   );

create table watchlist_item (
   id            varchar2(36) primary key,
   user_id       varchar2(64) not null,
   folder_id     varchar2(36) not null,
   symbol        varchar2(16) not null,
   sort_order    number default 0 not null,
   created_at    timestamp with time zone default systimestamp not null,
   updated_at    timestamp with time zone default systimestamp not null,
   constraint fk_watchlist_item_folder foreign key ( folder_id )
      references watchlist_folder ( id ) on delete cascade
);

create index idx_watchlist_item_folder on
   watchlist_item (
      folder_id,
      sort_order
   );

create unique index uq_watchlist_item_folder_symbol on
   watchlist_item (
      folder_id,
      symbol
   );

create index idx_watchlist_item_user on
   watchlist_item (
      user_id,
      symbol
   );

create table watchlist_workspace (
   user_id     varchar2(64) primary key,
   version     number default 0 not null,
   created_at  timestamp with time zone default systimestamp not null,
   updated_at  timestamp with time zone default systimestamp not null
);
