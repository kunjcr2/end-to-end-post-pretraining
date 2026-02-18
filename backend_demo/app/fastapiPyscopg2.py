# FastAPI CRUD Demo — REST API for posts backed by PostgreSQL
# Part of the FastAPI + PostgreSQL + CI/CD course
# Connected to PostgreSQL via psycopg2 with raw SQL (no ORM)
# Endpoints: health check, get/create/update/delete posts
#
# This file uses raw parameterized SQL queries through psycopg2.
# For the SQLAlchemy ORM version, see fastapiSqlalchemy.py
#
# This file is a standalone learning exercise separate from the inference API
# (see api/app.py for the actual vLLM-powered model serving endpoint).
# The patterns here — Pydantic models, route structure, status codes — carry
# over directly to the inference server.

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import random

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import time 
import os

# loading the .env file
load_dotenv()

# building the connection to the db
while True:
    try:
        
        conn = psycopg2.connect(
            host=os.getenv("HOST"),
            database=os.getenv("DATABASE"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            cursor_factory=RealDictCursor # this makes it return python dictionaries instead of tuples
        )
        cur = conn.cursor()
        print("Database connection successful!")
        break

    except Exception as er:
        print(f"Database connection failed. Error: {er}")
        print("Trying connection in 3 seconds...")
        time.sleep(3)

# --- Pydantic schema for a Post ------------------------------------------------
# Same idea as api/schema.py — define request/response shapes with BaseModel.
# When switching to PostgreSQL, this model will mirror the SQLAlchemy ORM model.
class Post(BaseModel):
    id: int = None        # auto-assigned on creation
    title: str
    content: str
    published: bool = True


# --- App instance ----------------------------------------------------------------
app = FastAPI()

# --- Routes ----------------------------------------------------------------------

# Health check — same pattern used in api/app.py for the inference server
@app.get("/health")
def check_health():
    return {"message": "healthy"}


# GET all posts — later: paginate with query params (limit, offset)
@app.get("/posts", status_code=status.HTTP_200_OK)
def get_posts():
    cur.execute("""SELECT * FROM posts""")
    data = cur.fetchall()
    return {"message": data}


# GET single post by ID — uses path parameter, returns [] if missing
@app.get("/posts/{id}", status_code=status.HTTP_200_OK)
def get_post(id: int):

    # Safer way to do so, instead of f string
    cur.execute("""SELECT * FROM posts WHERE id=%s""", (id,)) # needs to add comma to make it a tuple
    data = cur.fetchone() # If we dont find anything, it will return simply just an empty list

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return {"message": data}


# CREATE a new post — PostgreSQL has auto-increment for ID
@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: Post):

    # Can do below thing, but it will make it open to SQL Injetion attacks
    # cur.execute(f"""INSERT INTO posts (title, content) VALUES ('{post.title}', '{post.content}')""")

    # And hence we do this, also, returning returns the created item
    cur.execute("""INSERT INTO posts (title, content, published) VALUES (%s, %s, %s) RETURNING *""", 
                (post.title, post.content, post.published))
    data = cur.fetchone()
    conn.commit() # commits it to the database server

    return {"message": data}


# UPDATE an existing post — full replacement (PUT semantics)
@app.put("/posts/{id}", status_code=status.HTTP_201_CREATED)
def update_post(id: int, post: Post):

    cur.execute("""
        UPDATE posts SET title=%s, content=%s, published=%s WHERE id=%s RETURNING *
    """, (post.title, post.content, post.published, id))
    data = cur.fetchone()
    conn.commit()

    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")

    return {"message": data}


# DELETE a post — removes from list; will become a SQL DELETE later
@app.delete("/posts/{id}", status_code=status.HTTP_200_OK)
def delete_post(id: int):

    cur.execute("""
        DELETE FROM posts WHERE id = %s RETURNING *
    """, (id,))
    data = cur.fetchone()
    conn.commit() # We are commiting right away !

    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")

    return {"message": data}