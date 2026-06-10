"""
Central configuration module for the Coiltech Industrial Automation Platform.

This module provides application-wide settings using Pydantic Settings
for environment variable management and configuration validation.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    Sensitive values like API keys should always be set via environment.
    """

    # ======================================================================
    # Application Settings
    # ======================================================================
    APP_NAME: str = Field(
        default="Coiltech Industrial Automation Platform",
        description="Application name",
    )
    APP_VERSION: str = Field(default="2.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode flag")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # ======================================================================
    # API Settings
    # ======================================================================
    API_HOST: str = Field(default="0.0.0.0", description="API host address")
    API_PORT: int = Field(default=8000, description="API port number")
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080,http://localhost:5173",
        description="Comma-separated CORS origins",
    )

    # ======================================================================
    # LLM Settings - Ollama (Primary)
    # ======================================================================
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    OLLAMA_ROUTING_MODEL: str = Field(
        default="llama3.2",
        description="Fast model for intent classification/routing",
    )
    OLLAMA_GENERATION_MODEL: str = Field(
        default="llama3.1:8b", description="Powerful model for response generation"
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model (optional, fallback)",
    )
    OLLAMA_TIMEOUT: int = Field(
        default=120, description="Ollama request timeout in seconds"
    )

    # ======================================================================
    # LLM Settings - OpenAI (Fallback)
    # ======================================================================
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL (or compatible endpoint)",
    )
    OPENAI_ROUTING_MODEL: str = Field(
        default="gpt-3.5-turbo", description="OpenAI model for routing"
    )
    OPENAI_GENERATION_MODEL: str = Field(
        default="gpt-4-turbo-preview", description="OpenAI model for generation"
    )

    # ======================================================================
    # LLM Settings - Anthropic (Fallback)
    # ======================================================================
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    ANTHROPIC_ROUTING_MODEL: str = Field(
        default="claude-3-haiku-20240307", description="Anthropic model for routing"
    )
    ANTHROPIC_GENERATION_MODEL: str = Field(
        default="claude-3-sonnet-20240229", description="Anthropic model for generation"
    )

    # LLM Provider Priority
    LLM_PROVIDER_PRIORITY: str = Field(
        default="ollama,openai,anthropic",
        description="Comma-separated LLM provider priority",
    )

    # ======================================================================
    # Database Settings
    # ======================================================================
    SQLITE_DB_PATH: str = Field(
        default="data/coiltech_platform.db", description="SQLite database file path"
    )
    DB_ECHO: bool = Field(default=False, description="Echo SQL statements")

    # ======================================================================
    # ChromaDB / Vector Store Settings
    # ======================================================================
    CHROMA_PERSIST_DIR: str = Field(
        default="data/chroma_db", description="ChromaDB persistence directory"
    )
    CHROMA_COLLECTION_PREFIX: str = Field(
        default="coiltech", description="Prefix for ChromaDB collection names"
    )

    # ======================================================================
    # Embedding Settings
    # ======================================================================
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers embedding model",
    )
    EMBEDDING_DEVICE: str = Field(
        default="cpu", description="Device for embeddings (cpu, cuda, mps)"
    )

    # ======================================================================
    # RAG Settings
    # ======================================================================
    CHUNK_SIZE: int = Field(default=512, description="Document chunk size in tokens")
    CHUNK_OVERLAP: int = Field(default=64, description="Chunk overlap in tokens")
    RAG_TOP_K: int = Field(default=5, description="Number of documents to retrieve")
    RAG_SCORE_THRESHOLD: float = Field(
        default=0.3, description="Minimum similarity score threshold"
    )

    # ======================================================================
    # Domain Collections Configuration
    # ======================================================================
    RAG_COLLECTIONS: str = Field(
        default=(
            "products_compact_drives,products_loop_drives,products_blank_lines,"
            "products_destacker,products_press_transfer,products_press_transfer_robots,"
            "products_servo_drives,products_decoilers,products_ctl_lines,"
            "special_projects,installation_commissioning,service_maintenance,"
            "spare_parts,quality_documents,sales_and_quotations"
        ),
        description="Comma-separated RAG collection names (without prefix)",
    )
    DEFAULT_COLLECTION: str = Field(
        default="general", description="Default RAG collection for unclassified docs"
    )

    # ======================================================================
    # Digital Twin Settings — domain-specific thresholds
    # ======================================================================
    SENSOR_SERVO_LOAD_THRESHOLD: float = Field(
        default=85.0, description="Servo load threshold in percent"
    )
    SENSOR_FEED_DEVIATION_THRESHOLD: float = Field(
        default=0.15, description="Feed length deviation threshold in mm"
    )
    SENSOR_BEARING_TEMP_THRESHOLD: float = Field(
        default=75.0, description="Bearing temperature threshold in Celsius"
    )
    SENSOR_VIBRATION_THRESHOLD: float = Field(
        default=7.5, description="Vibration threshold in mm/s"
    )
    SENSOR_AIR_PRESSURE_MIN: float = Field(
        default=4.5, description="Minimum air pressure in bar"
    )
    SENSOR_AIR_PRESSURE_MAX: float = Field(
        default=7.0, description="Maximum air pressure in bar"
    )

    # ======================================================================
    # Session Settings
    # ======================================================================
    SESSION_TIMEOUT: int = Field(
        default=3600, description="Session timeout in seconds"
    )
    MAX_CONVERSATION_HISTORY: int = Field(
        default=20, description="Maximum messages in conversation history"
    )

    # ======================================================================
    # File Upload Settings
    # ======================================================================
    UPLOAD_DIR: str = Field(
        default="data/uploads", description="Directory for uploaded files"
    )
    MAX_UPLOAD_SIZE: int = Field(
        default=50 * 1024 * 1024, description="Maximum upload file size in bytes"
    )
    ALLOWED_EXTENSIONS: str = Field(
        default=".pdf,.docx,.txt,.csv,.xlsx,.md",
        description="Comma-separated allowed file extensions",
    )

    # ======================================================================
    # CAG (Context-Augmented Generation) Cache Settings
    # ======================================================================
    ENABLE_CAG: bool = Field(
        default=True, description="Enable in-memory context cache for repeated queries"
    )
    CAG_CACHE_TTL_SECONDS: int = Field(
        default=300, description="Time-to-live for cached RAG context entries (seconds)"
    )
    CAG_MAX_ENTRIES: int = Field(
        default=200, description="Maximum number of cached context entries"
    )

    # ======================================================================
    # Open WebUI Integration Settings
    # ======================================================================
    OPENWEBUI_UPLOADS_DIR: str = Field(
        default="",
        description="Open WebUI uploads directory path. Set to watch for new files.",
    )
    OPENWEBUI_WATCH_ENABLED: bool = Field(
        default=False, description="Enable file watching for Open WebUI uploads"
    )
    OPENWEBUI_SCAN_INTERVAL: int = Field(
        default=5, description="Interval in seconds to scan for new files"
    )
    OPENWEBUI_DEFAULT_COLLECTION: str = Field(
        default="special_projects",
        description="Default collection for Open WebUI uploaded files",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """Accept common deployment strings such as 'release' and 'prod'."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug", "development"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return False

    @field_validator("SQLITE_DB_PATH", mode="before")
    @classmethod
    def normalize_legacy_db_path(cls, value):
        if value is None:
            return "data/coiltech_platform.db"
        normalized = str(value).strip()
        if normalized.endswith("industrial_platform.db"):
            return "data/coiltech_platform.db"
        return normalized

    @field_validator("CHROMA_COLLECTION_PREFIX", mode="before")
    @classmethod
    def normalize_legacy_collection_prefix(cls, value):
        if value is None:
            return "coiltech"
        normalized = str(value).strip().lower()
        if normalized in {"industrial", "industrial_ai", "industrial-ai"}:
            return "coiltech"
        return normalized

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def rag_collections_list(self) -> List[str]:
        """Get RAG collections as a list."""
        return [c.strip() for c in self.RAG_COLLECTIONS.split(",")]

    @property
    def llm_providers_list(self) -> List[str]:
        """Get LLM providers as a list in priority order."""
        return [provider.strip() for provider in self.LLM_PROVIDER_PRIORITY.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get allowed file extensions as a list."""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            os.path.dirname(self.SQLITE_DB_PATH),
            self.CHROMA_PERSIST_DIR,
            self.UPLOAD_DIR,
        ]
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# =========================================================================
# Intent categories for routing
# =========================================================================
INTENT_CATEGORIES = [
    "product_lookup",
    "application_engineering",
    "configuration_request",
    "quotation_request",
    "project_status",
    "commissioning_support",
    "maintenance_request",
    "spare_parts_lookup",
    "installed_base_query",
    "anomaly_detection",
    "quality_compliance",
    "executive_summary",
    "general_knowledge",
]

# Intent → Agent mapping
INTENT_AGENT_MAP = {
    "product_lookup": "ProductCatalogAgent",
    "application_engineering": "ApplicationEngineeringAgent",
    "configuration_request": "ConfigurationAgent",
    "quotation_request": "QuotationAgent",
    "project_status": "ProjectExecutionAgent",
    "commissioning_support": "InstallationCommissioningAgent",
    "maintenance_request": "ServiceMaintenanceAgent",
    "spare_parts_lookup": "SparePartsAgent",
    "installed_base_query": "ServiceMaintenanceAgent",
    "anomaly_detection": "DigitalTwinAgent",
    "quality_compliance": "QualityComplianceAgent",
    "executive_summary": "ManagementReportingAgent",
    "general_knowledge": "GeneralAgent",
}

# Intent → primary RAG collections mapping
INTENT_COLLECTION_MAP = {
    "product_lookup": [
        "products_compact_drives", "products_loop_drives", "products_blank_lines",
        "products_destacker", "products_press_transfer", "products_press_transfer_robots",
        "products_servo_drives", "products_decoilers", "products_ctl_lines",
    ],
    "application_engineering": [
        "products_compact_drives", "products_loop_drives", "products_servo_drives",
        "products_decoilers", "products_ctl_lines", "special_projects",
    ],
    "configuration_request": [
        "products_compact_drives", "products_loop_drives", "products_servo_drives",
        "products_decoilers", "products_ctl_lines",
    ],
    "quotation_request": ["sales_and_quotations"],
    "project_status": ["special_projects"],
    "commissioning_support": ["installation_commissioning"],
    "maintenance_request": ["service_maintenance"],
    "spare_parts_lookup": ["spare_parts"],
    "installed_base_query": ["service_maintenance"],
    "anomaly_detection": ["service_maintenance"],
    "quality_compliance": ["quality_documents"],
    "executive_summary": ["sales_and_quotations"],
    "general_knowledge": [],
}


RAG_COLLECTION_REGISTRY = {
    "products_compact_drives": {
        "label": "Compact Drives",
        "domain": "product_catalog",
        "description": "Compact drive brochures, datasheets, application notes, and layouts.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_loop_drives": {
        "label": "Loop Type Drives",
        "domain": "product_catalog",
        "description": "Loop drive product information and application guidance.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_blank_lines": {
        "label": "Blank Lines",
        "domain": "product_catalog",
        "description": "Blank line references, layouts, and commercial brochures.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_destacker": {
        "label": "Destacker Systems",
        "domain": "product_catalog",
        "description": "Destacker systems documentation and technical references.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_press_transfer": {
        "label": "Press Transfer Systems",
        "domain": "product_catalog",
        "description": "Press transfer system documentation and layouts.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_press_transfer_robots": {
        "label": "Press Transfer Robots",
        "domain": "product_catalog",
        "description": "Robot transfer product literature, manuals, and configuration guides.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_servo_drives": {
        "label": "Servo Drives",
        "domain": "product_catalog",
        "description": "Servo drive brochures, sizing notes, and setup references.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_decoilers": {
        "label": "Decoilers",
        "domain": "product_catalog",
        "description": "Mechanical, hydraulic, double mandrel, and pallet type decoiler documents.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "products_ctl_lines": {
        "label": "Cut-to-Length Lines",
        "domain": "product_catalog",
        "description": "Standard and high-performance CTL line references.",
        "document_types": ["manual", "datasheet", "drawing", "training"],
    },
    "special_projects": {
        "label": "Special Projects",
        "domain": "project_execution",
        "description": "Custom project references, line concepts, and engineering notes.",
        "document_types": ["manual", "drawing", "training", "commissioning_checklist"],
    },
    "installation_commissioning": {
        "label": "Installation & Commissioning",
        "domain": "service",
        "description": "Site installation procedures, commissioning checklists, and FAT/SAT notes.",
        "document_types": ["manual", "commissioning_checklist", "training", "sop"],
    },
    "service_maintenance": {
        "label": "Service & Maintenance",
        "domain": "service",
        "description": "Maintenance manuals, troubleshooting guides, and service bulletins.",
        "document_types": ["manual", "service_bulletin", "training", "sop"],
    },
    "spare_parts": {
        "label": "Spare Parts",
        "domain": "service",
        "description": "Spare part catalogs, parts books, and replacement guidance.",
        "document_types": ["manual", "datasheet", "service_bulletin"],
    },
    "quality_documents": {
        "label": "Quality & Compliance",
        "domain": "quality",
        "description": "CE certificates, quality procedures, audits, and compliance records.",
        "document_types": ["certificate", "sop", "manual", "training"],
    },
    "sales_and_quotations": {
        "label": "Sales & Quotations",
        "domain": "sales",
        "description": "Quotation templates, commercial terms, and sales references.",
        "document_types": ["datasheet", "training", "manual"],
    },
}


# =========================================================================
# Prompt templates
# =========================================================================
ROUTING_SYSTEM_PROMPT = """You are an intent classification system for the Coiltech Industrial Automation Platform.
This company manufactures press feeding systems, decoilers, cut-to-length lines, transfer systems, and servo drives
for the metal stamping / metalforming industry.

Classify user queries into one of these categories:

CATEGORIES:
- product_lookup: Questions about product families, models, specifications, capabilities, catalogs
- application_engineering: Technical selection, feasibility, sizing, material/speed/width requirements
- configuration_request: Line configuration, BOM assembly, component selection for a project
- quotation_request: Pricing, RFQ, quote creation/status, commercial terms
- project_status: Project tracking, milestones, delivery schedules, project history
- commissioning_support: Installation procedures, commissioning checklists, startup support
- maintenance_request: Service tickets, preventive maintenance, repair history, PM schedules
- spare_parts_lookup: Parts availability, stock levels, part ordering, part identification
- installed_base_query: Customer machine registry, installed machines, warranty status
- anomaly_detection: Sensor alerts, machine alarms, anomaly detection, real-time monitoring
- quality_compliance: CE marking, quality events, acceptance tests, compliance docs
- executive_summary: Management KPIs, pipeline, revenue, cross-domain summaries
- general_knowledge: General questions, greetings, platform help

Respond ONLY with a JSON object:
{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}"""

NL_TO_SQL_PROMPT = """You are a SQL query generator for a Coiltech industrial automation database.
Convert natural language questions into SQLite queries.

DATABASE SCHEMA:
{schema}

RULES:
1. Generate only SELECT queries (no INSERT, UPDATE, DELETE)
2. Use proper SQLite syntax
3. Include relevant columns for the question
4. Use appropriate JOINs when needed
5. Add ORDER BY and LIMIT when appropriate
6. Use date/time functions for temporal queries
7. Never query sensor_streams without a WHERE clause on timestamp or machine_id

USER QUESTION: {question}

Respond with ONLY the SQL query, no explanations or markdown."""
