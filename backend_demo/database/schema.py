from pydantic import BaseModel, ConfigDict

class Posts(BaseModel):
    id: int = None
    title: str
    content: str
    published: bool = True

    # Pydantic V2: from_attributes=True lets FastAPI serialize SQLAlchemy ORM objects
    model_config = ConfigDict(from_attributes=True)