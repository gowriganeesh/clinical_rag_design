"""Central production configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _path_env(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Config:
    project_root: Path = PROJECT_ROOT

    # Azure OpenAI.
    aoai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    aoai_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    aoai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    aoai_chat_api_version: str = os.getenv(
        "AZURE_OPENAI_CHAT_API_VERSION",
        os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    embed_deployment: str = os.getenv(
        "AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"
    )
    router_deployment: str = os.getenv("AZURE_OPENAI_ROUTER_DEPLOYMENT", "gpt-5.4-nano")
    answer_deployment: str = os.getenv("AZURE_OPENAI_ANSWER_DEPLOYMENT", "gpt-5.4")
    fallback_deployment: str = os.getenv(
        "AZURE_OPENAI_FALLBACK_DEPLOYMENT", "gpt-5.4-nano"
    )
    embed_dimensions: int = _int_env("EMBED_DIMENSIONS", 1024)
    max_output_tokens: int = _int_env("MAX_OUTPUT_TOKENS", 700)
    chat_token_param: str = os.getenv("OPENAI_CHAT_TOKEN_PARAM", "auto")

    # Azure AI Search.
    search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    search_key: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "clinical-kb")
    retrieval_top_k: int = _int_env("RETRIEVAL_TOP_K", 8)
    rerank_top_n: int = _int_env("RERANK_TOP_N", 5)
    reranker_score_threshold: float = _float_env("RERANKER_SCORE_THRESHOLD", 2.0)

    # Cache.
    cache_similarity_threshold: float = _float_env("CACHE_SIMILARITY_THRESHOLD", 0.94)

    # Chunking.
    chunk_size_tokens: int = _int_env("CHUNK_SIZE_TOKENS", 500)
    chunk_overlap_sentences: int = _int_env("CHUNK_OVERLAP_SENTENCES", 1)

    # Data.
    data_dir: Path = _path_env("DATA_DIR", "data")

    def validate_production(self) -> None:
        missing = []
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", self.aoai_endpoint),
            ("AZURE_OPENAI_API_KEY", self.aoai_key),
            ("AZURE_SEARCH_ENDPOINT", self.search_endpoint),
            ("AZURE_SEARCH_API_KEY", self.search_key),
        ]:
            if not value:
                missing.append(name)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Production mode requires Azure settings: {joined}")


config = Config()
