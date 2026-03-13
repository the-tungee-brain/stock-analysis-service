from app.models.schwab_models import SchwabAccessTokenResponse, SchwabAuthTokenItem


def schwab_token_to_item(
    user_id: str, token: SchwabAccessTokenResponse
) -> SchwabAuthTokenItem:
    return SchwabAuthTokenItem(
        id=None,
        user_id=user_id,
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        access_expires_at=token.access_expires_at,
        refresh_expires_at=token.refresh_expires_at,
        created_at=None,
        updated_at=None,
    )
