# REVIEW AND CONTINUE FROM HERE
# https://docs.sqlalchemy.org/en/20/tutorial/orm_data_manipulation.html#inserting-rows-using-the-orm-unit-of-work-pattern

from sqlalchemy import create_engine, text # Creates engine
from sqlalchemy.engine import URL # Helps in creating the URL
from sqlalchemy.orm import Session # Now we make a connection using an engine

from dotenv import load_dotenv
from pathlib import Path
import os

# Resolve .env relative to THIS file's directory, not the cwd
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

url_obj = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    database=os.getenv("DATABASE")
)
engine = create_engine(url_obj) # Creates an engine (Doesnt connect yet)

# ### GETTING ALL/SOME posts, UPDATING POSTS
# # Makes a connection with a session and exectures the statement
# stmt = text("SELECT * FROM posts WHERE id>2") # This shows a textual SQL query
# with Session(engine) as s:
#     res = s.execute(stmt)
#     print(res.all())
#     print("-"*50)

#     res = s.execute(
#         text(
#             "UPDATE posts SET published=:y WHERE id=:x"
#         ),
#         {"x": 3, "y": False}
#     )
#     s.commit() # Commit to make changes to the actual database
#     print("Updated!")
#     print("-"*50)

#     s.close()

from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Integer, String, Boolean, TIMESTAMP, func, select

# Need to make a Base class
class Base(DeclarativeBase):
    pass

# And we extend this Base class to our "Tables"
class Post(Base):
    __tablename__ = "posts" # Table name in remote

    # Column or mapped_collumn, both can be used here.
    id = mapped_column(Integer, primary_key=True, nullable=False)
    title = mapped_column(String, nullable=False)
    content = mapped_column(String, nullable=False)
    published = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # IDK BELOW
    created_at = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self):
        return f"Post(id={self.id}, title='{self.title}', content='{self.content}', published={self.published}, created_at={self.created_at})"

# Create the tables in the database
Base.metadata.create_all(bind=engine)

session = Session(engine)
posts = [
    Post(title="Post 1", content="Content 1", published=True),
    Post(title="Post 2", content="Content 2", published=True),
    Post(title="Post 3", content="Content 3", published=True)
]
# Adding
for post in posts:
    session.add(post)
    session.flush()

# Pushing to the remote server
session.commit()

# Verify insertion
print("Posts in database:")
res = session.execute(select(Post)).scalars().all()
for row in res:
    print(row)

session.close()