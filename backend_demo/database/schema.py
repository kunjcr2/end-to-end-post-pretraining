# Pydantic V2 schemas for request validation and response serialization
#
# Models:
#   Posts      — CRUD schema for blog posts
#   UserCreate — incoming payload when registering (uses EmailStr for validation)
#   UserSent   — safe response model (omits password, includes created_at)

from pydantic import BaseModel, ConfigDict, EmailStr
import datetime

class Posts(BaseModel):
    id: int = None
    title: str
    content: str
    published: bool = True

    # Pydantic V2: from_attributes=True lets FastAPI serialize SQLAlchemy ORM objects
    model_config = ConfigDict(from_attributes=True)

# Incoming user registration payload — email is validated via pydantic[email]
class UserCreate(BaseModel):
    id: int = None
    email: EmailStr
    password: str

    model_config = ConfigDict(from_attributes=True)

# Response model returned after user creation — excludes the password field
class UserSent(BaseModel):
    id: int = None
    email: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)