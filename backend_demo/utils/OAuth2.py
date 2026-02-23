import jwt
from datetime import datetime, timedelta

from fastapi import HTTPException
from backend_demo.database.schema import UserSent, TokenData

SECRET = "cqiuwriwoqnfurgvnfoirvnioqwngqiwrjgvbqwohrngfckwrungkgb"
ALGORITHM = "HS256"
EXPIRE_MIN = 30

def create_access_token(payload: UserSent):
    payload_2 = payload.to_dict()
    # Convert created_at to string so it's JSON-serializable for jwt.encode()
    if payload_2.get("created_at") is not None:
        payload_2["created_at"] = payload_2["created_at"].isoformat()
    # Expiration must be set in the payload BEFORE encoding
    payload_2["exp"] = datetime.now() + timedelta(minutes=EXPIRE_MIN)
    encoded_jwt = jwt.encode(payload_2, SECRET, algorithm=ALGORITHM)
    
    return encoded_jwt

def decode_jwt(token: str):

    decoded_jwt = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    token_data = TokenData(**decoded_jwt)

    if token_data.id is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return token_data

def get_current_user(token: str):
    
    token_data = decode_jwt(token)
    user = UserSent.get(token_data.id)
    return user