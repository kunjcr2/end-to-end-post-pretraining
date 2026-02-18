from pydantic import BaseModel

class Posts(BaseModel):
    id: int = None
    title: str
    content: str
    published: bool = True

    # Lets the orm model use this non mapped pydantic model as a mapped orm model
    class Config:
        orm_mode = True