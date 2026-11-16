import pytest

from app.config import ConfigurationError, Settings


def test_defaults_without_any_configuration(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "RETRIEVAL_STRATEGY",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.retrieval_strategy == "rerank"
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.require_openai_api_key()
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.require_database_url()


def test_empty_values_count_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.retrieval_strategy == "rerank"


def test_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_CHAT_MODEL=from-file\nRETRIEVAL_STRATEGY=keyword\n")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "from-env")
    monkeypatch.delenv("RETRIEVAL_STRATEGY", raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.require_chat_model() == "from-env"
    assert settings.retrieval_strategy == "keyword"


def test_an_unknown_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "bm25")

    with pytest.raises(ValueError, match="retrieval_strategy"):
        Settings(_env_file=None)


def test_secrets_are_not_printed():
    settings = Settings(_env_file=None, openai_api_key="sk-secret", database_url="postgresql://u:pw@h/db")

    assert "sk-secret" not in repr(settings)
    assert "pw@" not in repr(settings)
