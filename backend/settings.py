import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Coiltech RAG Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    # Override via env var: PUBLIC_BASE_URL=http://51.20.115.179:8000
    # If empty, auto-detected from AWS metadata or local IP at startup.
    PUBLIC_BASE_URL: str = ""
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080,http://localhost:5173"

    DATABASE_URL: str = "postgresql://postgres:test1234@localhost:5432/ragdb"
    SQLITE_DB_PATH: str = "data/coiltech_platform.db"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_GENERATION_MODEL: str = "gemma3"
    OLLAMA_ROUTING_MODEL: str = "gemma3"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 300
    DEFAULT_MODEL: str = "ollama:gemma3"

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_MODELS: str = "deepseek-chat,deepseek-reasoner"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODELS: str = "gemini-2.5-flash,gemini-2.5-pro"

    DEFAULT_ANSWER_LANGUAGE: str = "tr"
    RAG_MULTI_QUERY: bool = False
    RAG_TOP_K: int = 10
    RAG_ALPHA: float = 0.3
    RAG_SCORE_THRESHOLD: float = 0.3
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    RAG_USE_GRAPH: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "test1234"

    EXCEL_SEED_PATH: str = "issue_solution_final.xlsx"
    OPENWEBUI_DEFAULT_COLLECTION: str = "service_maintenance"
    OPENWEBUI_WATCH_ENABLED: bool = False
    OPENWEBUI_UPLOADS_DIR: str = ""
    OPENWEBUI_SCAN_INTERVAL: int = 30

    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.csv,.xlsx,.md"

    AGENTS_STORE_DIR: str = "agents_store"
    PENDING_AGENT_TTL_MINUTES: int = 30

    ENABLE_CAG: bool = True
    CAG_CACHE_TTL_SECONDS: int = 300
    CAG_MAX_ENTRIES: int = 200

    ANSWER_CACHE_ENABLED: bool = True
    ANSWER_CACHE_TTL_SECONDS: int = 86400

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    RAG_COLLECTIONS: str = (
        "products_compact_drives,products_loop_drives,products_blank_lines,"
        "products_destacker,products_press_transfer,products_press_transfer_robots,"
        "products_servo_drives,products_decoilers,products_ctl_lines,"
        "special_projects,installation_commissioning,service_maintenance,"
        "spare_parts,quality_documents,sales_and_quotations"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def rag_collections_list(self) -> List[str]:
        return [c.strip() for c in self.RAG_COLLECTIONS.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip() for e in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def deepseek_models_list(self) -> List[str]:
        if not self.DEEPSEEK_API_KEY:
            return []
        return [m.strip() for m in self.DEEPSEEK_MODELS.split(",") if m.strip()]

    @property
    def gemini_models_list(self) -> List[str]:
        if not self.GEMINI_API_KEY:
            return []
        return [m.strip() for m in self.GEMINI_MODELS.split(",") if m.strip()]

    def ensure_directories(self) -> None:
        for d in [self.UPLOAD_DIR, os.path.dirname(self.SQLITE_DB_PATH)]:
            if d:
                os.makedirs(d, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    s.ensure_directories()
    return s
