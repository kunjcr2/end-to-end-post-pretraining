# Database Configuration and CRUD Operations
#
# This module handles the database connection using SQLAlchemy.
# It uses:
# - SQLAlchemy ORM for database interactions (Post & User models)
# - Pydantic V2 for data validation and serialization
# - Context managers (with Session(...) as session) for thread-safe session management
# - Automatic table creation on startup for all registered tables (posts, users)
# - A duplicate-user guard to prevent creating users with existing emails

from sqlalchemy import create_engine, URL, Integer, Boolean, String, TIMESTAMP, func, select
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session, mapped_column, DeclarativeBase
from backend_demo.database.schema import Posts, UserCreate
from backend_demo.utils.hash import hash

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

class User(_Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    email = mapped_column(String, nullable=False, unique=True)
    password = mapped_column(String, nullable=False)
    created_at = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self):
        return f"User(id={self.id}, email='{self.email}', created_at={self.created_at})"

_engine = create_engine(URL.create(
    "postgresql+psycopg2",
    username=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    database=os.getenv("DATABASE")
))

def make_table():
    """Iterate over TABLES and create any that don't exist yet."""
    print("-" * 50)
    inspector = inspect(_engine)  # Introspects the DB to check existing tables
    for TABLE in TABLES:
        if inspector.has_table(TABLE):
            print(f"Table {TABLE} exists")
        else:
            print(f"Table {TABLE} does not exist")
            _Base.metadata.create_all(bind=_engine)

def check_connection():
    try:
        with _engine.connect() as conn:
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def get_all_posts():
    with Session(_engine, expire_on_commit=False) as session:
        stmt = select(Post)
        posts = session.execute(stmt).scalars().all()
        return posts

def get_post_by_id(id):
    with Session(_engine, expire_on_commit=False) as session:
        post = session.get(Post, id)
        if not post:
            return None
        return post

def create_post(post: Posts):
    with Session(_engine, expire_on_commit=False) as session:
        new_post = Post(
            title=post.title,
            content=post.content,
            published=post.published
        )
        session.add(new_post)
        session.commit()
        session.refresh(new_post)
        return new_post

def update_post(id: int, post: Posts):
    with Session(_engine, expire_on_commit=False) as session:
        old_post = session.get(Post, id)
        if not old_post:
            return None
        
        if post.title is not None:
            old_post.title = post.title
        
        if post.content is not None:
            old_post.content = post.content
        
        if post.published is not None:
            old_post.published = post.published
        
        session.commit()
        session.refresh(old_post)
        return old_post

def delete_post(id: int):
    with Session(_engine, expire_on_commit=False) as session:
        post = session.get(Post, id)
        if not post:
            return None
        
        # Create a dictionary of the post data to return
        deleted_post_data = {
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "published": post.published,
            "created_at": post.created_at
        }
        
        session.delete(post)
        session.commit()

        return deleted_post_data

def create_user(user: UserCreate):
    """Create a new user. Returns None if the email already exists (unique constraint)."""

    with Session(_engine, expire_on_commit=False) as session:
        new_user = User(
            email=user.email,
            password=hash(user.password)
        )

        try:
            session.add(new_user)
            session.commit()
        except Exception:
            # Unique-constraint violation (duplicate email) → signal failure to caller
            return None
        session.refresh(new_user)

        return new_user
        