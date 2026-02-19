# Database Configuration and CRUD Operations
#
# This module handles the database connection using SQLAlchemy.
# It uses:
# - SQLAlchemy ORM for database interactions
# - Pydantic V2 for data validation and serialization
# - Context managers (with Session(...) as session) for thread-safe session management
# - Automatic table creation on connection check

from sqlalchemy import create_engine, URL, Integer, Boolean, String, TIMESTAMP, func, select
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session, mapped_column, DeclarativeBase
from backend_demo.database.schema import Posts
from dotenv import load_dotenv
import os
import pathlib

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
TABLE_NAME="posts"

class _Base(DeclarativeBase):
    pass

class Post(_Base):
    __tablename__ = TABLE_NAME # Table name in remote

    # Column or mapped_collumn, both can be used here.
    id = mapped_column(Integer, primary_key=True, nullable=False)
    title = mapped_column(String, nullable=False)
    content = mapped_column(String, nullable=False)
    published = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # IDK BELOW
    created_at = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self):
        return f"Post(id={self.id}, title='{self.title}', content='{self.content}', published={self.published}, created_at={self.created_at})"

_engine = create_engine(URL.create(
    "postgresql+psycopg2",
    username=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    database=os.getenv("DATABASE")
))

def make_table():
    print("-"*50)
    # Checking if the table exists
    inspector = inspect(_engine) # Creates an inspection table
    if inspector.has_table(TABLE_NAME):
        print("Table exists")
    else:
        print("Table does not exist")
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
