"""对话管理 API：SSE 流式问答、会话 CRUD。"""

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.schemas import (
    APIResponse,
    ChatStreamRequest,
    CreateSessionRequest,
    MessageResponse,
    SessionDetailResponse,
    SessionResponse,
)
from src.db.database import get_db
from src.db import crud
from src.graph.builder import build_graph_with_memory

logger = logging.getLogger("deer_scholar.api.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["对话管理"])

# 复用 DeerFlow 原有的 graph
_graph = build_graph_with_memory()


def _session_to_response(s) -> SessionResponse:
    return SessionResponse(
        id=s.id, title=s.title, created_at=s.created_at, updated_at=s.updated_at
    )


def _message_to_response(m) -> MessageResponse:
    sources = None
    if m.sources:
        try:
            sources = json.loads(m.sources)
        except (json.JSONDecodeError, TypeError):
            sources = None
    return MessageResponse(
        id=m.id,
        session_id=m.session_id,
        role=m.role,
        content=m.content,
        sources=sources,
        created_at=m.created_at,
    )


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(request: ChatStreamRequest, db: Session = Depends(get_db)):
    """流式问答接口，基于 DeerFlow LangGraph 工作流。"""
    session_id = request.session_id

    # 自动创建或获取会话
    if not session_id:
        session = crud.create_session(
            db, title=request.query[:50] if request.query else "新对话"
        )
        session_id = session.id
    else:
        session = crud.get_session(db, session_id)
        if not session:
            session = crud.create_session(
                db, title=request.query[:50] if request.query else "新对话"
            )
            session_id = session.id

    # 记录用户消息
    crud.create_message(db, session_id=session_id, role="user", content=request.query)
    crud.update_session_time(db, session_id)

    # 构造 DeerFlow 原始消息格式
    messages = [{"role": "user", "content": request.query}]
    thread_id = session_id

    async def event_generator():
        collected_content = []
        try:
            async for event in _graph.astream(
                {
                    "messages": messages,
                    "locale": request.locale,
                    "mode": request.mode,
                    "max_plan_iterations": request.max_plan_iterations,
                    "max_step_num": request.max_step_num,
                    "max_search_results": request.max_search_results,
                    "auto_accepted_plan": request.auto_accepted_plan,
                    "enable_background_investigation": request.enable_background_investigation,
                },
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 30,
                },
                stream_mode="messages",
            ):
                if isinstance(event, tuple) and len(event) == 2:
                    msg_chunk, metadata = event
                    content = getattr(msg_chunk, "content", "")
                    if content and isinstance(content, str):
                        collected_content.append(content)
                        data = json.dumps(
                            {
                                "thread_id": thread_id,
                                "content": content,
                                "agent": metadata.get("langgraph_node", ""),
                            },
                            ensure_ascii=False,
                        )
                        yield f"data: {data}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # 记录助手消息到数据库
        full_content = "".join(collected_content)
        if full_content:
            from src.db.database import SessionLocal

            with SessionLocal() as db_inner:
                crud.create_message(
                    db_inner,
                    session_id=session_id,
                    role="assistant",
                    content=full_content,
                )
                crud.update_session_time(db_inner, session_id)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/sessions", summary="创建新对话")
async def create_session_endpoint(
    request: CreateSessionRequest = None, db: Session = Depends(get_db)
):
    """创建一个新的对话会话。"""
    title = (request.title if request and request.title else "新对话")
    session = crud.create_session(db, title=title)
    return APIResponse(data=_session_to_response(session))


@router.get("/sessions", summary="获取对话列表")
async def list_sessions(db: Session = Depends(get_db)):
    """获取所有对话会话列表，按更新时间倒序。"""
    sessions = crud.list_sessions(db)
    return APIResponse(data=[_session_to_response(s) for s in sessions])


@router.get("/sessions/{session_id}", summary="获取单个对话历史")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """获取指定对话的详细信息和消息历史。"""
    session = crud.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    msgs = crud.get_messages(db, session_id)
    return APIResponse(
        data=SessionDetailResponse(
            session=_session_to_response(session),
            messages=[_message_to_response(m) for m in msgs],
        )
    )


@router.delete("/sessions/{session_id}", summary="删除对话")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """删除指定对话及其所有消息。"""
    if not crud.delete_session(db, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return APIResponse(message="对话已删除")
