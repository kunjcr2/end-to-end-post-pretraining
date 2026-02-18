# FastAPI CRUD Demo — REST API for posts backed by PostgreSQL
# Part of the FastAPI + PostgreSQL + CI/CD course
# Connected to PostgreSQL via SQLAlchemy ORM
# Endpoints: health check, get/create/update/delete posts
#
# This file uses SQLAlchemy ORM with Pydantic V2 models.
# Database interactions are handled via context managers in database.py
#
# This file is a standalone learning exercise separate from the inference API
# (see api/app.py for the actual vLLM-powered model serving endpoint).
# The patterns here — Pydantic models, route structure, status codes — carry
# over directly to the inference server.

from fastapi import FastAPI, HTTPException, status

from backend_demo.database.schema import Posts
import backend_demo.database.database as db
import time

# building the connection to the db
while True:
    try:
        db.make_table()
        if db.check_connection():
            print("Database connection successful!")
            break

    except Exception as er:
        print(f"Database connection failed. Error: {er}")
        print("Trying connection in 3 seconds...")
        time.sleep(3)


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
    
    res = db.get_all_posts()
    return res


# GET single post by ID — uses path parameter, returns [] if missing
@app.get("/posts/{id}", status_code=status.HTTP_200_OK)
def get_post(id: int):

    res = db.get_post_by_id(id)

    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return res


# CREATE a new post — PostgreSQL has auto-increment for ID
@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: Posts):

    res = db.create_post(post)
    return res


# UPDATE an existing post — full replacement (PUT semantics)
@app.put("/posts/{id}", status_code=status.HTTP_201_CREATED)
def update_post(id: int, post: Posts):

    pass


# DELETE a post — removes from list; will become a SQL DELETE later
@app.delete("/posts/{id}", status_code=status.HTTP_200_OK)
def delete_post(id: int):

    res = db.delete_post(id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return res