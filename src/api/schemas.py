"""Pydantic request/response models for Deer-Scholar API."""

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ---- 统一响应格式 ----

class APIResponse(BaseModel, Generic[T]):
    """统一 API 响应格式。"""
    code: int = Field(200, description="HTTP 状态码")
    message: str = Field("success", description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")


# ---- 对话相关 ----

class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="会话标题，为空则自动生成")


class SessionResponse(BaseModel):
    id: str = Field(..., description="会话 ID (UUID)")
    title: str = Field(..., description="会话标题")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="最后更新时间")


class MessageResponse(BaseModel):
    id: str = Field(..., description="消息 ID (UUID)")
    session_id: str = Field(..., description="所属会话 ID")
    role: str = Field(..., description="角色: user / assistant")
    content: str = Field(..., description="消息内容")
    sources: Optional[List[str]] = Field(None, description="引用的 arXiv ID 列表")
    created_at: datetime = Field(..., description="创建时间")


class SessionDetailResponse(BaseModel):
    session: SessionResponse = Field(..., description="会话信息")
    messages: List[MessageResponse] = Field(default_factory=list, description="消息列表")


class ChatStreamRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话 ID，为空则自动创建")
    query: str = Field(..., description="用户提问内容")
    mode: str = Field("scholar", description="对话模式: default / scholar")
    locale: str = Field("zh-CN", description="语言区域")
    max_plan_iterations: int = Field(1, description="最大计划迭代次数")
    max_step_num: int = Field(3, description="计划最大步骤数")
    max_search_results: int = Field(3, description="最大搜索结果数")
    auto_accepted_plan: bool = Field(False, description="是否自动接受计划")
    enable_background_investigation: bool = Field(True, description="是否启用背景调查")


# ---- 知识库相关 ----

class IngestRequest(BaseModel):
    source_type: str = Field(
        "arxiv_dataset",
        description="数据源类型: arxiv_dataset / pdf_upload",
    )
    dataset_name: str = Field(
        "jamescalam/ai-arxiv-chunked",
        description="HuggingFace 数据集名称",
    )
    limit: int = Field(2000, description="导入行数上限", ge=1, le=100000)
    batch_size: int = Field(64, description="批量处理大小", ge=1, le=512)
    recreate: bool = Field(False, description="是否重建 collection")


class IngestTaskResponse(BaseModel):
    id: str = Field(..., description="任务 ID (UUID)")
    status: str = Field(..., description="任务状态: pending / processing / completed / failed")
    source_type: str = Field(..., description="数据源类型")
    total_chunks: int = Field(0, description="总 chunk 数")
    processed_chunks: int = Field(0, description="已处理 chunk 数")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class KnowledgeStatsResponse(BaseModel):
    collection_name: str = Field(..., description="Qdrant collection 名称")
    total_points: int = Field(0, description="总向量数（chunk 数）")
    unique_papers: int = Field(0, description="去重后论文数（估算）")
    vector_dimension: int = Field(0, description="向量维度")
    distance_metric: str = Field("Cosine", description="距离度量方式")


# ---- 检索相关 ----

class ScholarSearchRequest(BaseModel):
    query: str = Field(..., description="检索查询文本")
    top_k: int = Field(5, description="返回论文数量", ge=1, le=50)
    category: Optional[str] = Field(None, description="arXiv 分类过滤 (如 cs.AI)")
    year_from: Optional[int] = Field(None, description="起始年份过滤")
    score_threshold: Optional[float] = Field(None, description="相似度阈值", ge=0, le=1)


class PaperResult(BaseModel):
    score: float = Field(..., description="相似度分数")
    title: str = Field(..., description="论文标题")
    arxiv_id: str = Field(..., description="arXiv ID")
    url: Optional[str] = Field(None, description="arXiv 链接")
    snippet: str = Field("", description="证据片段")
    categories: Optional[Any] = Field(None, description="论文分类")
    published: Optional[str] = Field(None, description="发布日期")


class SimilarSearchRequest(BaseModel):
    arxiv_id: str = Field(..., description="目标论文 arXiv ID")
    top_k: int = Field(5, description="返回数量", ge=1, le=50)


# ---- 系统相关 ----

class ServiceStatus(BaseModel):
    status: str = Field(..., description="服务状态: healthy / unhealthy")
    error: Optional[str] = Field(None, description="错误详情")


class HealthResponse(BaseModel):
    status: str = Field(..., description="整体状态: healthy / degraded / unhealthy")
    checks: dict[str, ServiceStatus] = Field(..., description="各服务检查结果")


class MetricsResponse(BaseModel):
    uptime_seconds: float = Field(..., description="服务运行时间（秒）")
    total_requests: int = Field(0, description="总请求数")
    total_searches: int = Field(0, description="总检索数")
    avg_search_latency_ms: float = Field(0, description="平均检索延迟（毫秒）")
