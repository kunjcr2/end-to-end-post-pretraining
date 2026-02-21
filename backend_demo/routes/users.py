# Users Router — registration & lookup endpoints
# Mounted in main_orm.py via app.include_router()
# DB queries are delegated to utils/query_db.py

from fastapi import status, APIRouter, HTTPException
from backend_demo.database.schema import UserCreate, UserSent
import backend_demo.utils.query_db as db

router = APIRouter(prefix="/users", tags=["Users"])

# Register a new user — returns 400 if the email is already taken
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=UserSent)
def create_user(user: UserCreate):
    res = db.create_user(user)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email: {user.email} already exists"
        )
    return {
        "id": res.id,
        "email": res.email,
        "created_at": res.created_at
    }

@router.get("/{id}", status_code=status.HTTP_200_OK, response_model=UserSent)
def get_user(id: int):
    res = db.get_user_by_id(id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id: {id} was not found"
        )
    return res