# OAuth2 Bearer Token Authentication
#
# Full auth flow:
#   1. User logs in via POST /login → gets back a JWT token
#   2. Frontend stores the token and sends it on every request as:
#         Authorization: Bearer eyJ...
#   3. On protected endpoints, Depends(get_current_user) triggers the chain:
#         oauth2_scheme (extracts token) → decode_jwt (verifies token) → endpoint runs
#
# Three key pieces:
#   oauth2_scheme   — EXTRACTS the token string from the Authorization header
#   decode_jwt      — VERIFIES the token (signature, expiration, payload)
#   get_current_user — Ties them together as a single dependency for endpoints

import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os

from backend_demo.database.schema import UserSent, TokenData, Token

load_dotenv()

# OAuth2PasswordBearer — this does NOT verify anything.
# It simply reads the "Authorization: Bearer <token>" header from
# the incoming HTTP request, strips the "Bearer " prefix, and returns
# the raw token string. If the header is missing, it auto-returns 401.
# tokenUrl="login" tells Swagger UI (/docs) where the login endpoint is,
# so the "Authorize" button knows where to send credentials.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET = os.getenv("SECRET")
ALGORITHM = os.getenv("ALGORITHM")
EXPIRE_MIN = os.getenv("EXPIRE_MIN")

def create_access_token(payload: UserSent):
    """Called during login — encodes user data into a signed JWT token.
    The token contains the user's id, email, and an expiration time.
    This token is what the frontend stores and sends back on every request.
    """
    payload_2 = payload.to_dict()
    # Convert created_at to string so it's JSON-serializable for jwt.encode()
    if payload_2.get("created_at") is not None:
        payload_2["created_at"] = payload_2["created_at"].isoformat()
    # Expiration must be set in the payload BEFORE encoding
    payload_2["exp"] = datetime.now() + timedelta(minutes=int(EXPIRE_MIN))
    encoded_jwt = jwt.encode(payload_2, SECRET, algorithm=ALGORITHM)
    
    return encoded_jwt

def decode_jwt(token: str):
    """VERIFIES and decodes a JWT token. This is where the actual
    security check happens:
      - Decodes the token using the secret key (verifies signature)
      - jwt library auto-checks the "exp" field (raises ExpiredSignatureError)
      - Extracts the user id from the decoded payload
      - Returns TokenData(id=...) on success, raises 401 on any failure
    """
    try:
        decoded_jwt = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        id: str = decoded_jwt.get("id")
        if id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        token_data = TokenData(id=id)

    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

    return token_data

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency for protected endpoints. Add to any route like:
        @router.get("/posts")
        def get_posts(current_user: TokenData = Depends(get_current_user)):

    When a request comes in:
      1. Depends(oauth2_scheme) EXTRACTS the token from the Authorization header
      2. This function passes it to decode_jwt() which VERIFIES it
      3. Returns TokenData with the user's id — or raises 401
    """
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    return decode_jwt(token)
