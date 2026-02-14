# FastAPI CRUD Demo — basic REST API for posts
# Part of the FastAPI + PostgreSQL + CI/CD course
# Currently using in-memory storage; will migrate to PostgreSQL w/ SQLAlchemy
# Endpoints: health check, get/create/update/delete posts
#
# This file is a standalone learning exercise separate from the inference API
# (see api/app.py for the actual vLLM-powered model serving endpoint).
# The patterns here — Pydantic models, route structure, status codes — carry
# over directly to the inference server.
#
# Next steps / things to add:
#   - Replace in-memory list with PostgreSQL via SQLAlchemy ORM
#   - Add Alembic for database migrations
#   - Add request validation & error handling middleware
#   - Add authentication (OAuth2 / JWT) to protect routes
#   - Wire up CI/CD pipeline (GitHub Actions → Docker → deploy)
#   - Eventually merge this CRUD pattern into the inference API
#     so model queries can be stored/retrieved from a database

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import random


# --- Pydantic schema for a Post ------------------------------------------------
# Same idea as api/schema.py — define request/response shapes with BaseModel.
# When switching to PostgreSQL, this model will mirror the SQLAlchemy ORM model.
class Post(BaseModel):
    id: int = None        # auto-assigned on creation
    title: str
    content: str


# --- In-memory "database" -------------------------------------------------------
# Temporary storage — will be replaced by a real PostgreSQL table.
# Each dict here represents a row; keys map to future column names.
posts = [
    {
        "id": 18419,
        "title": "post 1",
        "content": "content 1"
    },
    {
        "id": 138253,
        "title": "post 2",
        "content": "content 2"
    },
    {
        "id": 14141,
        "title": "post 3",
        "content": "content 3"
    }
]


# --- App instance ----------------------------------------------------------------
app = FastAPI()


# --- Helper ----------------------------------------------------------------------
# Linear search over the list — will become a SQL query once we have a real DB.
def get_index_from_id(id: int):
    for i, post in enumerate(posts):
        if post["id"] == id:
            return i
    return None


# --- Routes ----------------------------------------------------------------------

# Health check — same pattern used in api/app.py for the inference server
@app.get("/health")
def check_health():
    return {"message": "healthy"}


# GET all posts — later: paginate with query params (limit, offset)
@app.get("/posts")
def get_posts():
    return {"message": posts}


# GET single post by ID — uses path parameter, returns 404 if missing
@app.get("/posts/{id}")
def get_post(id: int):
    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    return {"message": posts[index]}


# CREATE a new post — random ID for now; PostgreSQL will auto-increment
@app.post("/posts")
def create_post(post: Post):
    rand_id = random.randint(1, 100000)
    post.id = rand_id
    posts.append(post)
    return {"message": post}


# UPDATE an existing post — full replacement (PUT semantics)
@app.put("/posts/{id}")
def update_post(id: int, post: Post):

    # getting the index of the post
    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    # converting it to a dictionary and saving and returning
    post.id = id
    posts[index] = post.dict()
    return {"message": post}


# DELETE a post — removes from list; will become a SQL DELETE later
@app.delete("/posts/{id}")
def delete_post(id: int):

    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    post = posts.pop(index)
    return {"message": post}