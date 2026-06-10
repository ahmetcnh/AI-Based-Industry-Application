"""
Pydantic models for the Coiltech Industrial Automation Platform API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntentCategoryEnum(str, Enum):
    PRODUCT_LOOKUP = "product_lookup"
    APPLICATION_ENGINEERING = "application_engineering"
    CONFIGURATION_REQUEST = "configuration_request"
    QUOTATION_REQUEST = "quotation_request"
    PROJECT_STATUS = "project_status"
    COMMISSIONING_SUPPORT = "commissioning_support"
    MAINTENANCE_REQUEST = "maintenance_request"
    SPARE_PARTS_LOOKUP = "spare_parts_lookup"
    INSTALLED_BASE_QUERY = "installed_base_query"
    ANOMALY_DETECTION = "anomaly_detection"
    QUALITY_COMPLIANCE = "quality_compliance"
    EXECUTIVE_SUMMARY = "executive_summary"
    GENERAL_KNOWLEDGE = "general_knowledge"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    session_id: Optional[str] = None
    stream: bool = False
    collection_hint: Optional[str] = Field(
        default=None,
        description="Optional RAG collection hint such as products_decoilers or service_maintenance.",
    )


class IntentResultModel(BaseModel):
    category: IntentCategoryEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    original_query: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceModel(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection: Optional[str] = None


class ChatResponse(BaseModel):
    status: str = "success"
    content: str
    intent: Optional[IntentResultModel] = None
    agent_used: str
    sources: Optional[List[SourceModel]] = None
    data: Optional[Dict[str, Any]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OpenAIFileAttachment(BaseModel):
    type: str = "file"
    name: Optional[str] = None
    data: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    model: str = "coiltech-platform"
    messages: List[Dict[str, Any]]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=1, le=8192)
    stream: bool = False
    files: Optional[List[OpenAIFileAttachment]] = None
    collection_name: Optional[str] = None


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class UploadRequest(BaseModel):
    collection_name: Optional[str] = None
    title: Optional[str] = None
    doc_type: Optional[str] = None
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    status: str
    file_name: str
    collection_name: str
    chunks_created: int
    chunks_stored: int
    uploaded_document_id: Optional[int] = None
    message: str


class DocumentAgentInfo(BaseModel):
    id: int
    name: str
    slug: str
    model_id: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    rag_collection: str
    source_document_id: Optional[int] = None
    original_filename: Optional[str] = None
    status: str = "ready"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class DocumentAgentCreateResponse(BaseModel):
    status: str
    agent: DocumentAgentInfo
    chunks_created: int
    chunks_stored: int
    message: str


class DocumentAgentsResponse(BaseModel):
    status: str
    agents: List[DocumentAgentInfo]
    count: int


class CollectionInfo(BaseModel):
    name: str
    collection_key: Optional[str] = None
    document_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollectionsResponse(BaseModel):
    status: str
    collections: List[CollectionInfo]


class AgentInfo(BaseModel):
    name: str
    department: str
    description: str


class AgentsResponse(BaseModel):
    status: str
    agents: List[AgentInfo]


class StatusResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DatabaseQueryResponse(BaseModel):
    status: str
    question: str
    generated_sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    error: Optional[str] = None


class ExcelAnalysisResponse(BaseModel):
    status: str
    file_name: str
    operation: str
    sheet_names: List[str]
    selected_sheet: Optional[str] = None
    question: Optional[str] = None
    result: Dict[str, Any]
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    preferences: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    status: str
    session: SessionInfo


class ProductFamilyRecord(BaseModel):
    id: int
    code: str
    name: str
    category: str
    description: Optional[str] = None
    is_active: int


class ProductFamilyListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class ProjectListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class ProjectDetailResponse(BaseModel):
    status: str
    item: Dict[str, Any]


class QuotationListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class InstalledBaseListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class ServiceTicketListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class SparePartListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int


class AlertListResponse(BaseModel):
    status: str
    items: List[Dict[str, Any]]
    count: int
