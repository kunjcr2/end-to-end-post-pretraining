# Posts Router — CRUD endpoints for the posts resource
# Mounted in main_orm.py via app.include_router()
# DB queries are delegated to utils/query_db.py

from fastapi import HTTPException, status, APIRouter
from typing import List

import backend_demo.utils.query_db as db
from backend_demo.database.schema import Posts

router = APIRouter(prefix="/posts", tags=["Posts"])

# GET all posts — later: paginate with query params (limit, offset)
@router.get("/", status_code=status.HTTP_200_OK, response_model=List[Posts])
def get_posts():
    
    res = db.get_all_posts()
    return res


# GET single post by ID — uses path parameter, returns [] if missing
@router.get("/{id}", status_code=status.HTTP_200_OK, response_model=Posts)
def get_post(id: int):

    res = db.get_post_by_id(id)

    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return res


# CREATE a new post — PostgreSQL has auto-increment for ID
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=Posts)
def create_post(post: Posts):

    res = db.create_post(post)
    return res


# UPDATE an existing post — full replacement (PUT semantics)
@router.put("/{id}", status_code=status.HTTP_201_CREATED, response_model=Posts)
def update_post(id: int, post: Posts):

    res = db.update_post(id, post)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return res

# DELETE a post — removes from DB via SQLAlchemy session.delete()
@router.delete("/{id}", status_code=status.HTTP_200_OK, response_model=Posts)
def delete_post(id: int):

    res = db.delete_post(id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id: {id} was not found"
        )
    return res