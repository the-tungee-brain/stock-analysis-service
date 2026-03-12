from app.adapters.schwab.schwab_auth import SchwabAuth
from app.models.schwab_models import SchwabAccessTokenResponse


class SchwabAuthBuilder:
    def __init__(self):
        self.schwab_auth = SchwabAuth()

    def get_access_token(self, auth_code: str) -> SchwabAccessTokenResponse:
        response = self.schwab_auth.get_access_token(auth_code=auth_code)
        token_data = response.json()
        token = SchwabAccessTokenResponse(**token_data)
        token.set_expiration()
        return token

    def get_refreshed_access_token(
        self, refresh_token: str
    ) -> SchwabAccessTokenResponse:
        response = self.schwab_auth.get_refreshed_access_token(
            refresh_token=refresh_token
        )
        token_data = response.json()
        token = SchwabAccessTokenResponse(**token_data)
        token.set_expiration()
        return token
