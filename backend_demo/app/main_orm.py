# FastAPI CRUD Demo — App entry point
# Connected to PostgreSQL via SQLAlchemy ORM
#
# This file creates the FastAPI app instance, establishes the DB connection,
# and mounts route modules via include_router():
#   - routes/posts.py  → /posts CRUD endpoints
#   - routes/users.py  → /users registration & lookup
#
# The /health endpoint is kept here as a top-level check.
#
# Database interactions are handled in utils/query_db.py using context-managed
# sessions from database/database.py (SQLAlchemy ORM + Pydantic V2).
#
# This file is a standalone learning exercise separate from the inference API
# (see api/app.py for the actual vLLM-powered model serving endpoint).
# The patterns here — Pydantic models, route structure, status codes — carry
# over directly to the inference server.

from fastapi import FastAPI, status
import time

import backend_demo.routes.posts as posts
import backend_demo.routes.users as users
import backend_demo.utils.query_db as db

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
@app.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    return {"message": "healthy"}

app.include_router(posts.router)
app.include_router(users.router)