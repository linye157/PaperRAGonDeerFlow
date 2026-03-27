"""数据库引擎初始化与 Session 工厂。"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 默认 SQLite 路径：项目根目录 data/deer_scholar.db
_default_db_path = Path(__file__).resolve().parent.parent.parent / "data" / "deer_scholar.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_default_db_path}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 需要
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """创建所有表（如果不存在）。"""
    from src.db.models import Base
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI 依赖注入用的 DB Session 生成器。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
