import os
import validators

from dateutil import parser
from datetime import datetime
from typing import Dict, Optional

from .constants import (
    API_KEY_CONFIG_KEY,
    BACKEND_OVERRIDE_CONFIG_KEY,
    BEARER_TOKEN_CONFIG_KEY,
    BEARER_TOKEN_EXPIRES_AT_CONFIG_KEY,
    MQL_OVERRIDE_CONFIG_KEY,
)
from .exceptions import AuthException, URLException
from .interfaces import MQLInterface, BackendInterface
from .models import User, Organization, ModelKey, UserState
from .config_handler import ConfigHandler


class TransformAuth:  # noqa: D
    def __init__(  # noqa: D
        self,
        api_key: Optional[str] = None,
        mql_server_url: Optional[str] = None,
        override_config: bool = True,
        use_async: bool = False,
        target_org_id: Optional[int] = None,  # no-op for non-admin use
    ) -> None:
        self.__user_state: Optional[UserState] = None
        self.__mql_server_url: Optional[str] = None
        self.__mql_client: Optional[MQLInterface] = None
        self.__auth_header: Optional[Dict[str, str]] = None
        self.__api_key: Optional[str] = None
        self._use_async = use_async
        self.target_org_id = target_org_id
        self.config = ConfigHandler("config.yml", override_config)

        # Overrides
        if api_key is not None:
            self.api_key = api_key
        if mql_server_url is not None:
            self.mql_server_url_config_override = mql_server_url

    @property
    def api_key(self) -> Optional[str]:
        """Pull API key from config file by default, also supporting a fallback environment variable TRANSFORM_API_KEY."""
        if self.__api_key is None:
            config_api_key = self.config.get_config_value(API_KEY_CONFIG_KEY)
            return config_api_key if config_api_key else os.getenv("TRANSFORM_API_KEY")
        return self.__api_key

    @api_key.setter
    def api_key(self, api_key: str) -> None:
        """Set the users' API key and record to the config file."""

        # Validate API Key we expect a format like tfdk-prefix-secret
        # Do some very basic validation that this fits before setting key
        if not api_key.count("-") == 2:
            raise AuthException

        self.__api_key = api_key

        # Remove bearer token auth if exists
        self.config.pop_config_value(BEARER_TOKEN_CONFIG_KEY)
        self.config.pop_config_value(BEARER_TOKEN_EXPIRES_AT_CONFIG_KEY)

        self.config.set_config_value(API_KEY_CONFIG_KEY, api_key)

        # after setting a new API key, reset the user_state to reflect the new auth
        self.reset_user_state()

    @property
    def bearer_token(self) -> Optional[str]:
        """Get bearer token from config, if exists."""
        token = self.config.get_config_value(BEARER_TOKEN_CONFIG_KEY)
        if (
            token is not None
            and self.bearer_token_expires_at is not None
            and parser.parse(self.bearer_token_expires_at) <= datetime.now()
        ):
            raise AuthException("Bearer token has expired")

        return token

    @bearer_token.setter
    def bearer_token(self, bearer_token: str) -> None:  # noqa: D
        self.config.set_config_value(BEARER_TOKEN_CONFIG_KEY, bearer_token)
        self.config.pop_config_value(API_KEY_CONFIG_KEY)

    @property
    def bearer_token_expires_at(self) -> Optional[str]:
        """Time that current bearer token will expire, if exists."""
        return self.config.get_config_value(BEARER_TOKEN_EXPIRES_AT_CONFIG_KEY)

    @bearer_token_expires_at.setter
    def bearer_token_expires_at(self, bearer_token_expires_at: str) -> None:  # noqa: D
        self.config.set_config_value(BEARER_TOKEN_EXPIRES_AT_CONFIG_KEY, bearer_token_expires_at)

    @property
    def auth_header(self) -> Dict[str, str]:
        """Get auth header based on creds in config file. If none, return None."""
        if not self.__auth_header:
            if self.api_key is not None:
                self.__auth_header = {"Authorization": f"X-Api-Key {self.api_key}"}
            elif self.bearer_token is not None:
                self.__auth_header = {"Authorization": f"Bearer {self.bearer_token}"}
        if self.__auth_header is None:
            raise AuthException

        if self.target_org_id is not None:
            self.__auth_header["X-Target-Org-Id"] = str(self.target_org_id)

        return self.__auth_header

    @property
    def backend_client(self) -> BackendInterface:
        """Returns backend client"""
        overrides = {}
        backend_override = self.config.get_config_value(BACKEND_OVERRIDE_CONFIG_KEY)
        if backend_override:
            overrides["rest_api_url"] = backend_override
        return BackendInterface(auth_header=self.auth_header, use_async=self._use_async, **overrides)

    @property
    def user_state(self) -> UserState:  # noqa: D
        if self.__user_state is None:
            self.__user_state = self.backend_client.get_user_state()
        return self.__user_state

    def reset_user_state(self) -> None:
        """Resets the auth_header & pulls new user_state to ensure context is updated.

        Function is necessary to update old context ith new user_state when re-authenticating.
        """
        self.__auth_header = None
        self.__user_state = self.backend_client.get_user_state()

    @property
    def org(self) -> Organization:  # noqa: D
        return self.user_state.organization

    @property
    def user(self) -> User:  # noqa: D
        return self.user_state.user

    @property
    def current_model(self) -> Optional[ModelKey]:  # noqa: D
        return self.user_state.current_model

    @property
    def mql_server_url_status(self) -> str:  # noqa: D
        return f"Your MQL server is {self.mql_server_url} " + (
            f"which is overriden via {self.config.config_file_path} from {self.user.mql_server_url}"
            if self.mql_server_url_config_override
            else ""
        )

    @property
    def mql_server_url(self) -> str:
        """The URL for the org's MQL server.

        Overrideable via passing MQL server url at instantiation process for users who need to redirect using a local proxy.
        """
        if self.mql_server_url_config_override:
            return self.mql_server_url_config_override
        if not self.__mql_server_url:
            self.__mql_server_url = self.user.mql_server_url
        return self.__mql_server_url

    @property
    def mql_server_url_config_override(self) -> Optional[str]:
        """Config override for the users' MQL server URL

        Overrideable via passing MQL server url at instantiation process for users who need to redirect using a local proxy.
        """
        return self.config.get_config_value(MQL_OVERRIDE_CONFIG_KEY) if self.config.override_config else None

    @mql_server_url_config_override.setter
    def mql_server_url_config_override(self, mql_server_url: str) -> None:  # noqa: D
        if not mql_server_url:
            self.config.pop_config_value(MQL_OVERRIDE_CONFIG_KEY)
        elif validators.url(mql_server_url, public=False):
            self.__mql_server_url = mql_server_url
            self.config.set_config_value(MQL_OVERRIDE_CONFIG_KEY, mql_server_url)
        else:
            raise URLException(mql_server_url)

    @property
    def mql_client(self) -> MQLInterface:  # noqa: D
        if not self.__mql_client:
            self.__mql_client = MQLInterface(
                query_server_url=self.mql_server_url, auth_header=self.auth_header, use_async=self._use_async
            )
        return self.__mql_client
