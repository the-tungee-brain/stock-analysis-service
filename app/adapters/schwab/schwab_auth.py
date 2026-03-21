import requests
import base64
import os
from dotenv import load_dotenv
from urllib.parse import unquote


class SchwabAuth:

    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        load_dotenv()

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def _get_basic_auth_header(self):
        auth = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        return {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def get_refreshed_access_token(self, refresh_token):
        headers = self._get_basic_auth_header()
        data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        print(data, self.TOKEN_URL, headers)
        response = requests.post(self.TOKEN_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response
        else:
            raise Exception(f"Failed to refresh access token: {response.text}")

    def get_access_token(self, auth_code):
        headers = self._get_basic_auth_header()
        data = {
            "grant_type": "authorization_code",
            "code": unquote(auth_code),
            "redirect_uri": self.redirect_uri,
        }
        response = requests.post(self.TOKEN_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response
        else:
            raise Exception(f"Failed to get access token: {response.text}")
