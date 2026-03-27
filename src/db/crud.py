"""CRUD 操作封装。"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import ChatMessage, ChatSession, IngestTask, SearchLog


# ---- ChatSession ----

def create_session(db: Session, title: str = "新对话") -> ChatSession:
    session = ChatSession(title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str) -> Optional[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def list_sessions(db: Session, skip: int = 0, limit: int = 50) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def delete_session(db: Session, session_id: str) -> bool:
    session = get_session(db, session_id)
    if not session:
        return False
    db.delete(session)
    db.commit()
    return True


def update_session_time(db: Session, session_id: str) -> None:
    session = get_session(db, session_id)
    if session:
        session.updated_at = datetime.now(timezone.utc)
        db.commit()


# ---- ChatMessage ----

def create_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[list[str]] = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=json.dumps(sources) if sources else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages(db: Session, session_id: str) -> list[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )


# ---- IngestTask ----

def create_ingest_task(db: Session, source_type: str = "arxiv_dataset") -> IngestTask:
    task = IngestTask(source_type=source_type)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_ingest_task(db: Session, task_id: str) -> Optional[IngestTask]:
    return db.query(IngestTask).filter(IngestTask.id == task_id).first()


def list_ingest_tasks(
    db: Session, skip: int = 0, limit: int = 20
) -> list[IngestTask]:
    return (
        db.query(IngestTask)
        .order_by(IngestTask.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_ingest_task(
    db: Session,
    task_id: str,
    *,
    status: Optional[str] = None,
    total_chunks: Optional[int] = None,
    processed_chunks: Optional[int] = None,
    error_message: Optional[str] = None,
    completed_at: Optional[datetime] = None,
) -> Optional[IngestTask]:
    task = get_ingest_task(db, task_id)
    if not task:
        return None
    if status is not None:
        task.status = status
    if total_chunks is not None:
        task.total_chunks = total_chunks
    if processed_chunks is not None:
        task.processed_chunks = processed_chunks
    if error_message is not None:
        task.error_message = error_message
    if completed_at is not None:
        task.completed_at = completed_at
    db.commit()
    db.refresh(task)
    return task


# ---- SearchLog ----

def create_search_log(
    db: Session,
    query: str,
    top_k: int,
    threshold: Optional[float],
    result_count: int,
    latency_ms: float,
) -> SearchLog:
    log = SearchLog(
        query=query,
        top_k=top_k,
        threshold=threshold,
        result_count=result_count,
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
