"""ORM 模型：ChatSession, ChatMessage, IngestTask, SearchLog。"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    """对话会话。"""
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    title = Column(String(200), nullable=False, default="新对话")
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """对话消息。"""
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON 序列化的 arXiv ID 列表
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")


class IngestTask(Base):
    """入库任务。"""
    __tablename__ = "ingest_tasks"

    id = Column(String(36), primary_key=True, default=_uuid)
    status = Column(String(20), nullable=False, default="pending")
    source_type = Column(String(50), nullable=False, default="arxiv_dataset")
    total_chunks = Column(Integer, nullable=False, default=0)
    processed_chunks = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)


class SearchLog(Base):
    """检索日志（用于分析优化）。"""
    __tablename__ = "search_logs"

    id = Column(String(36), primary_key=True, default=_uuid)
    query = Column(Text, nullable=False)
    top_k = Column(Integer, nullable=False, default=5)
    threshold = Column(Float, nullable=True)
    result_count = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
