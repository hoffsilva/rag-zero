"""Configuration from environment variables (and api/.env.local in development).

Importing this module validates nothing: each credential is required only where it is used,
so the health check and most tests run without an OpenAI key or a database.
"""

from functools import lru_cache
from typing import Any

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain import Strategy
from app.paths import ENV_FILE


class ConfigurationError(RuntimeError):
    """A required setting is missing."""


class Settings(BaseSettings):
    """Environment variables win over api/.env.local; empty strings count as missing."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    # rerank had the best hit rate and MRR in eval/ (see README), for one extra chat call.
    retrieval_strategy: Strategy = "rerank"
    database_url: SecretStr | None = None

    @field_validator("openai_api_key", "openai_chat_model", "database_url", mode="before")
    @classmethod
    def _empty_is_missing(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("openai_embedding_model", "retrieval_strategy", mode="before")
    @classmethod
    def _empty_is_default(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str) and not value.strip() and info.field_name is not None:
            return cls.model_fields[info.field_name].default
        return value

    def require_openai_api_key(self) -> str:
        if self.openai_api_key is None:
            raise ConfigurationError("OPENAI_API_KEY is not set")
        return self.openai_api_key.get_secret_value()

    def require_chat_model(self) -> str:
        if self.openai_chat_model is None:
            raise ConfigurationError("OPENAI_CHAT_MODEL is not set")
        return self.openai_chat_model

    def require_database_url(self) -> str:
        if self.database_url is None:
            raise ConfigurationError("DATABASE_URL is not set")
        return self.database_url.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
