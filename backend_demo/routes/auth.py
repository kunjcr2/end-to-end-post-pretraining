# Auth Router — login endpoint for user authentication
# Mounted in main_orm.py via app.include_router()
#
# Login flow:
#   1. Look up user by email via query_db.get_user_by_email()
#   2. Verify the plaintext password against the stored bcrypt hash
#      using passlib's pwd_context.verify() (see utils/hash.py)
#   3. Return a token on success, or raise 404 / 401 on failure
#
# NOTE: Passwords are hashed with bcrypt (passlib) — never use Python's
# built-in hash(), which returns a non-cryptographic integer hash.

from fastapi import APIRouter, status, HTTPException

import backend_demo.utils.query_db as db
from backend_demo.database.schema import UserSent, UserCreate, Token
from backend_demo.utils.hash import verify_password
from backend_demo.utils.OAuth2 import create_access_token

router = APIRouter(tags=["Authentication"])

# Authenticate a user — looks up by email, verifies bcrypt-hashed password
@router.post("/login", status_code=status.HTTP_200_OK, response_model=Token)
def login(creds: UserCreate):
    
    user = db.get_user_by_email(creds)
    if user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User not found")
    
    if verify_password(creds.password, user.password) is False:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid credentials")
    
    encoded_jwt = create_access_token(user)
    
    return Token(
        token=encoded_jwt,
        token_type="Bearer"
    )