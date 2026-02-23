from ..database.database import _engine, _Base, TABLES, Post, User

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from ..database.schema import Posts, UserCreate
from ..utils.hash import hash_password

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
    """Create a new user with a bcrypt-hashed password.
    Uses hash_password() from utils/hash.py — never Python's built-in hash().
    Returns None if the email already exists (unique constraint violation).
    """

    with Session(_engine, expire_on_commit=False) as session:
        new_user = User(
            email=user.email,
            password=hash_password(user.password)
        )

        try:
            session.add(new_user)
            session.commit()
        except Exception:
            # Unique-constraint violation (duplicate email) → signal failure to caller
            return None
        session.refresh(new_user)

        return new_user
        
def get_user_by_id(id: int):
    """Look up a user by email using select().where() — not session.get(),
    which only works for primary-key lookups.
    Returns the User ORM object or None if not found.`
    """
    with Session(_engine, expire_on_commit=False) as session:
        stmt = select(User).where(User.id == id)
        user = session.execute(stmt).scalars().first()
        if user is None:
            return None

        return user
    
def get_user_by_email(creds: UserCreate):
    with Session(_engine, expire_on_commit=False) as session:
        stmt = select(User).where(User.email == creds.email)
        user = session.execute(stmt).scalars().first()
        if user is None:
            return None

        return user