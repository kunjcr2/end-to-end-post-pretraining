# Database Configuration and CRUD Operations
#
# This module handles the database connection using SQLAlchemy.
# It uses:
# - SQLAlchemy ORM for database interactions (Post & User models)
# - Pydantic V2 for data validation and serialization
# - Context managers (with Session(...) as session) for thread-safe session management
# - Automatic table creation on startup for all registered tables (posts, users)
# - A duplicate-user guard to prevent creating users with existing emails

from sqlalchemy import create_engine, URL, Integer, Boolean, String, TIMESTAMP, func, text
from sqlalchemy.orm import mapped_column, DeclarativeBase

from dotenv import load_dotenv
import os
import pathlib

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
TABLES = ["posts", "users"]  # All tables to auto-create on startup

class _Base(DeclarativeBase):
    pass

class Post(_Base):
    __tablename__ = "posts"

    # Column or mapped_column — both work; mapped_column is the modern SQLAlchemy 2.0 style
    id = mapped_column(Integer, primary_key=True, nullable=False)
    title = mapped_column(String, nullable=False)
    content = mapped_column(String, nullable=False)
    published = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # server_default=func.now() makes PostgreSQL set the timestamp automatically
    created_at = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self):
        return f"Post(id={self.id}, title='{self.title}', content='{self.content}', published={self.published}, created_at={self.created_at})"

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "published": self.published,
            "created_at": self.created_at
        }

class User(_Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    email = mapped_column(String, nullable=False, unique=True)
    password = mapped_column(String, nullable=False)
    created_at = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self):
        return f"User(id={self.id}, email='{self.email}', created_at={self.created_at})"

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "password": self.password,
            "created_at": self.created_at
        }

_engine = create_engine(URL.create(
    "postgresql+psycopg2",
    username=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    database=os.getenv("DATABASE")
))