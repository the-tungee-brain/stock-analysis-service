from __future__ import annotations

import asyncio

import oracledb
import pytest
from fastapi import HTTPException

from app.adapters.user.watchlist_adapter import (
    WatchlistAdapter,
    WatchlistVersionConflictError,
)
from app.api.watchlist_routes import sync_watchlist_workspace
from app.models.watchlist_models import (
    WatchlistFolderRecord,
    WatchlistWorkspaceSyncRequest,
)
from app.services.watchlist_service import WatchlistService


def _folder(folder_id: str = "folder-1", name: str = "Tech") -> WatchlistFolderRecord:
    return WatchlistFolderRecord(
        id=folder_id,
        name=name,
        iconName="folder.fill",
        swatchID="slate",
        isPinned=False,
        isCollapsed=False,
        sortOrder=0,
        symbols=[],
    )


class _TickerService:
    def get_by_symbols(self, symbols):
        return {}


class _FinnhubBuilder:
    def get_quote(self, symbol):
        return None


class _VersionedWatchlistAdapter:
    def __init__(self):
        self.folders: list[WatchlistFolderRecord] = []
        self.version = 0

    def list_workspace(self, user_id: str) -> list[WatchlistFolderRecord]:
        return [folder.model_copy(deep=True) for folder in self.folders]

    def get_workspace_version(self, user_id: str) -> int:
        return self.version

    def get_workspace_snapshot(
        self, user_id: str
    ) -> tuple[list[WatchlistFolderRecord], int]:
        return self.list_workspace(user_id), self.version

    def sync_workspace(
        self,
        user_id: str,
        folders: list[WatchlistFolderRecord],
        *,
        base_version: int | None = None,
    ) -> tuple[list[WatchlistFolderRecord], int]:
        if base_version is not None and base_version != self.version:
            raise WatchlistVersionConflictError(
                current_version=self.version,
                base_version=base_version,
            )
        self.folders = [folder.model_copy(deep=True) for folder in folders]
        self.version += 1
        return self.list_workspace(user_id), self.version

    def build_response(self, folders, **kwargs):
        return WatchlistAdapter.build_response(folders, **kwargs)


def _service(adapter: _VersionedWatchlistAdapter) -> WatchlistService:
    return WatchlistService(
        watchlist_adapter=adapter,  # type: ignore[arg-type]
        ticker_service=_TickerService(),  # type: ignore[arg-type]
        finnhub_builder=_FinnhubBuilder(),  # type: ignore[arg-type]
    )


class _SnapshotOnlyWatchlistAdapter(_VersionedWatchlistAdapter):
    def __init__(self):
        super().__init__()
        self.snapshot_calls = 0

    def list_workspace(self, user_id: str) -> list[WatchlistFolderRecord]:
        raise AssertionError("get_workspace must use get_workspace_snapshot")

    def get_workspace_version(self, user_id: str) -> int:
        raise AssertionError("get_workspace must use get_workspace_snapshot")

    def get_workspace_snapshot(
        self, user_id: str
    ) -> tuple[list[WatchlistFolderRecord], int]:
        self.snapshot_calls += 1
        return [folder.model_copy(deep=True) for folder in self.folders], self.version


def test_initial_get_returns_workspace_version():
    adapter = _VersionedWatchlistAdapter()
    service = _service(adapter)

    response = service.get_workspace(user_id="user-1", include_quotes=False)

    assert response.workspace_version == 0
    assert response.model_dump(by_alias=True)["workspaceVersion"] == 0


def test_get_workspace_uses_consistent_snapshot_adapter_path():
    adapter = _SnapshotOnlyWatchlistAdapter()
    adapter.folders = [_folder(name="Snapshot")]
    adapter.version = 7
    service = _service(adapter)

    response = service.get_workspace(user_id="user-1", include_quotes=False)

    assert adapter.snapshot_calls == 1
    assert response.workspace_version == 7
    assert response.folders[0].name == "Snapshot"


def test_first_sync_without_base_version_increments_version():
    adapter = _VersionedWatchlistAdapter()
    service = _service(adapter)

    response = service.sync_workspace(
        user_id="user-1",
        payload=WatchlistWorkspaceSyncRequest(folders=[_folder()]),
    )

    assert response.workspace_version == 1
    assert adapter.version == 1
    assert adapter.folders[0].name == "Tech"


def test_sync_with_matching_base_version_succeeds_and_increments_version():
    adapter = _VersionedWatchlistAdapter()
    adapter.folders = [_folder(name="Original")]
    adapter.version = 1
    service = _service(adapter)

    response = service.sync_workspace(
        user_id="user-1",
        payload=WatchlistWorkspaceSyncRequest(
            folders=[_folder(name="Updated")],
            baseVersion=1,
        ),
    )

    assert response.workspace_version == 2
    assert adapter.version == 2
    assert adapter.folders[0].name == "Updated"


def test_stale_base_version_returns_409():
    adapter = _VersionedWatchlistAdapter()
    adapter.folders = [_folder(name="Original")]
    adapter.version = 2
    service = _service(adapter)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            sync_watchlist_workspace(
                payload=WatchlistWorkspaceSyncRequest(
                    folders=[_folder(name="Stale")],
                    baseVersion=1,
                ),
                user_id="user-1",
                watchlist_service=service,
            )
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "watchlist_version_conflict"
    assert exc.value.detail["currentVersion"] == 2
    assert exc.value.detail["baseVersion"] == 1


def test_stale_sync_does_not_mutate_workspace():
    adapter = _VersionedWatchlistAdapter()
    adapter.folders = [_folder(name="Original")]
    adapter.version = 2
    service = _service(adapter)

    with pytest.raises(WatchlistVersionConflictError):
        service.sync_workspace(
            user_id="user-1",
            payload=WatchlistWorkspaceSyncRequest(
                folders=[_folder(name="Stale")],
                baseVersion=1,
            ),
        )

    assert adapter.version == 2
    assert adapter.folders[0].name == "Original"


class _DuplicateInsertCursor:
    def __init__(self):
        self.select_count = 0
        self.insert_count = 0

    def execute(self, sql, params):
        if "SELECT version" in sql and "FOR UPDATE" in sql:
            self.select_count += 1
            return
        if "INSERT INTO WATCHLIST_WORKSPACE" in sql:
            self.insert_count += 1
            raise oracledb.DatabaseError("ORA-00001: unique constraint violated")
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        if self.select_count == 1:
            return None
        if self.select_count == 2:
            return (3,)
        raise AssertionError("Unexpected fetchone call")


def test_first_workspace_version_duplicate_insert_reselects_for_update():
    adapter = WatchlistAdapter(client=object())  # type: ignore[arg-type]
    cursor = _DuplicateInsertCursor()

    version = adapter._get_or_create_workspace_version_for_update(
        cursor,
        "user-1",
    )

    assert version == 3
    assert cursor.insert_count == 1
    assert cursor.select_count == 2
