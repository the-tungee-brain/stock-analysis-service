import pytest
from datetime import datetime, timezone

from app.adapters.user.watchlist_adapter import WatchlistAdapter
from app.constants.watchlist_swatches import DEFAULT_WATCHLIST_SWATCH_ID
from app.models.watchlist_models import WatchlistFolderRecord, WatchlistSymbolRecord
from app.services.watchlist_service import WatchlistService


def _folder(
    *,
    folder_id: str = "folder-1",
    symbol: str = "AAPL",
    item_id: str = "item-1",
) -> WatchlistFolderRecord:
    return WatchlistFolderRecord(
        id=folder_id,
        name="Tech",
        iconName="chart.line.uptrend.xyaxis",
        swatchID="lavender",
        accentHex=None,
        isPinned=True,
        isCollapsed=False,
        sortOrder=0,
        symbols=[
            WatchlistSymbolRecord(
                id=item_id,
                ticker=symbol,
                sortOrder=0,
            )
        ],
    )


def test_validate_workspace_rejects_duplicate_symbols_in_folder():
    folder = WatchlistFolderRecord(
        id="folder-1",
        name="Tech",
        iconName="folder.fill",
        swatchID="slate",
        accentHex=None,
        isPinned=False,
        isCollapsed=False,
        sortOrder=0,
        symbols=[
            WatchlistSymbolRecord(id="item-1", ticker="AAPL", sortOrder=0),
            WatchlistSymbolRecord(id="item-2", ticker="aapl", sortOrder=1),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate symbol"):
        WatchlistAdapter._validate_workspace([folder])


def test_normalize_folders_sanitizes_swatch_and_symbols():
    service = WatchlistService(
        watchlist_adapter=object(),  # type: ignore[arg-type]
        ticker_service=object(),  # type: ignore[arg-type]
        finnhub_builder=object(),  # type: ignore[arg-type]
    )

    normalized = service._normalize_folders(
        [
            WatchlistFolderRecord(
                id="folder-1",
                name="  Tech  ",
                iconName="",
                swatchID="not-real",
                accentHex=None,
                isPinned=False,
                isCollapsed=False,
                sortOrder=0,
                symbols=[
                    WatchlistSymbolRecord(id="item-1", ticker=" nvda ", sortOrder=0),
                ],
            )
        ]
    )

    assert normalized[0].name == "Tech"
    assert normalized[0].icon_name == "folder.fill"
    assert normalized[0].swatch_id == DEFAULT_WATCHLIST_SWATCH_ID
    assert normalized[0].symbols[0].ticker == "NVDA"


def test_normalize_folders_preserves_premium_swatch_ids():
    service = WatchlistService(
        watchlist_adapter=object(),  # type: ignore[arg-type]
        ticker_service=object(),  # type: ignore[arg-type]
        finnhub_builder=object(),  # type: ignore[arg-type]
    )

    normalized = service._normalize_folders(
        [
            WatchlistFolderRecord(
                id="folder-1",
                name="Premium",
                iconName="star.fill",
                swatchID="orb-iris",
                accentHex=None,
                isPinned=False,
                isCollapsed=False,
                sortOrder=0,
                symbols=[],
            ),
            WatchlistFolderRecord(
                id="folder-2",
                name="Classic",
                iconName="folder.fill",
                swatchID="lavender",
                accentHex=None,
                isPinned=False,
                isCollapsed=False,
                sortOrder=1,
                symbols=[],
            ),
        ]
    )

    assert normalized[0].swatch_id == "orb-iris"
    assert normalized[1].swatch_id == "lavender"


def test_build_response_includes_quote_fields():
    folders = [_folder()]
    created_at = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    symbol_created_at = datetime(2026, 2, 1, 9, 30, tzinfo=timezone.utc)
    folders[0] = folders[0].model_copy(
        update={
            "created_at": created_at,
            "symbols": [
                WatchlistSymbolRecord(
                    id="item-1",
                    ticker="AAPL",
                    sortOrder=0,
                    createdAt=symbol_created_at,
                )
            ],
        }
    )
    response = WatchlistAdapter.build_response(
        folders,
        titles_by_symbol={"AAPL": "Apple Inc."},
        quotes_by_symbol={"AAPL": (190.0, 1.5, 0.8)},
    )

    folder = response.folders[0]
    assert folder.created_at == created_at
    symbol = folder.symbols[0]
    assert symbol.company_name == "Apple Inc."
    assert symbol.price == 190.0
    assert symbol.day_change == 1.5
    assert symbol.day_change_percent == 0.8
    assert symbol.created_at == symbol_created_at
