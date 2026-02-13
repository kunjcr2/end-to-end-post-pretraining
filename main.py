# CRUD API

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import random

class Post(BaseModel):
    id: int = None
    title: str
    content: str

posts = [
    {
        "id": 18419,
        "title": "post 1",
        "content": "content 1"
    },
    {
        "id": 138253,
        "title": "post 2",
        "content": "content 2"
    },
    {
        "id": 14141,
        "title": "post 3",
        "content": "content 3"
    }
]

app = FastAPI()

def get_index_from_id(id: int):
    for i, post in enumerate(posts):
        if post["id"] == id:
            return i
    return None

# Health check
@app.get("/health")
def check_health():
    return {"message": "healthy"}

# Get all posts
@app.get("/posts")
def get_posts():
    return {"message": posts}

# Get a post with ID
@app.get("/posts/{id}")
def get_post(id: int):
    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    return {"message": posts[index]}

# Create a new post
@app.post("/posts")
def create_post(post: Post):
    rand_id = random.randint(1, 100000)
    post.id = rand_id
    posts.append(post)
    return {"message": post}

# Update a post
@app.put("/posts/{id}")
def update_post(id: int, post: Post):

    # getting the index of the post
    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    # converting it to a dictionary and saving and returning
    post.id = id
    posts[index] = post.dict()
    return {"message": post}

# Delete a post
@app.delete("/posts/{id}")
def delete_post(id: int):

    index = get_index_from_id(id)
    if index is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    post = posts.pop(index)
    return {"message": post}