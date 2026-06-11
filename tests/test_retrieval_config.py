"""Settings coverage for the retrieval config block."""

from docstra.core.config.settings import RetrievalConfig, UserConfig


def test_retrieval_config_defaults():
    config = RetrievalConfig()
    assert config.rrf_k == 60
    assert config.fts_chunks_top_k == 50
    assert config.fts_symbols_top_k == 25


def test_user_config_exposes_retrieval():
    user_config = UserConfig()
    assert isinstance(user_config.retrieval, RetrievalConfig)
    assert user_config.retrieval.rrf_k == 60
