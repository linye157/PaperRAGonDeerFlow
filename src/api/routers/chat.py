"""对话管理 API：SSE 流式问答、会话 CRUD。"""

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.store.memory import InMemoryStore

from src.api.schemas import (
    APIResponse,
    ChatStreamRequest,
    CreateSessionRequest,
    MessageResponse,
    SessionDetailResponse,
    SessionResponse,
)
from src.graph.builder import build_graph_with_memory

logger = logging.getLogger("deer_scholar.api.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["对话管理"])

# 复用 DeerFlow 原有的 graph
_in_memory_store = InMemoryStore()
_graph = build_graph_with_memory()

# 内存中的会话存储（任务2会迁移到 SQLite）
_sessions: dict[str, dict] = {}
_messages: dict[str, list[dict]] = {}


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(request: ChatStreamRequest):
    """流式问答接口，基于 DeerFlow LangGraph 工作流。"""
    from datetime import datetime, timezone

    session_id = request.session_id or str(uuid4())

    # 自动创建会话
    if session_id not in _sessions:
        now = datetime.now(timezone.utc)
        _sessions[session_id] = {
            "id": session_id,
            "title": request.query[:50] if request.query else "新对话",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        _messages[session_id] = []

    # 记录用户消息
    user_msg = {
        "id": str(uuid4()),
        "session_id": session_id,
        "role": "user",
        "content": request.query,
        "sources": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _messages[session_id].append(user_msg)

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
                # event 是 (message_chunk, metadata) 元组
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

        # 记录助手消息
        full_content = "".join(collected_content)
        if full_content:
            assistant_msg = {
                "id": str(uuid4()),
                "session_id": session_id,
                "role": "assistant",
                "content": full_content,
                "sources": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _messages[session_id].append(assistant_msg)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/sessions", summary="创建新对话")
async def create_session(request: CreateSessionRequest = None):
    """创建一个新的对话会话。"""
    from datetime import datetime, timezone

    session_id = str(uuid4())
    now = datetime.now(timezone.utc)
    session = {
        "id": session_id,
        "title": (request.title if request and request.title else "新对话"),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    _sessions[session_id] = session
    _messages[session_id] = []

    return APIResponse(data=SessionResponse(**session))


@router.get("/sessions", summary="获取对话列表")
async def list_sessions():
    """获取所有对话会话列表，按更新时间倒序。"""
    sessions = sorted(
        _sessions.values(),
        key=lambda s: s["updated_at"],
        reverse=True,
    )
    return APIResponse(data=[SessionResponse(**s) for s in sessions])


@router.get("/sessions/{session_id}", summary="获取单个对话历史")
async def get_session(session_id: str):
    """获取指定对话的详细信息和消息历史。"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    session = SessionResponse(**_sessions[session_id])
    msgs = [MessageResponse(**m) for m in _messages.get(session_id, [])]

    return APIResponse(data=SessionDetailResponse(session=session, messages=msgs))


@router.delete("/sessions/{session_id}", summary="删除对话")
async def delete_session(session_id: str):
    """删除指定对话及其所有消息。"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    del _sessions[session_id]
    _messages.pop(session_id, None)

    return APIResponse(message="对话已删除")
