# Backend Demo — FastAPI + PostgreSQL CRUD

A standalone REST API for **posts** and **users**, built with FastAPI, SQLAlchemy ORM, and Pydantic V2.

---

## Quick Start

```bash
# Install dependencies
pip install -r backend_demo/requirements.txt

# Run the server (from the repo root)
uvicorn backend_demo.app.main_orm:app --reload
```

> The app requires a `.env` file in `backend_demo/` with your PostgreSQL credentials
> (`USER`, `PASSWORD`, `HOST`, `PORT`, `DATABASE`).

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/posts` | List all posts |
| `GET` | `/posts/{id}` | Get a single post |
| `POST` | `/posts` | Create a post |
| `PUT` | `/posts/{id}` | Update a post |
| `DELETE` | `/posts/{id}` | Delete a post |
| `POST` | `/users` | Register a new user |

### POST /users

- **Request body** — validated by `UserCreate` (uses `EmailStr` from `pydantic[email]`):
  ```json
  { "email": "user@example.com", "password": "secret" }
  ```
- **Response** — `UserSent` (password is excluded):
  ```json
  { "id": 1, "email": "user@example.com", "created_at": "2026-02-18T..." }
  ```
- Returns **400** if the email is already registered.

---

## Project Structure

```
backend_demo/
├── app/
│   ├── main_orm.py            # FastAPI routes (SQLAlchemy ORM)
│   └── main_psycopg2.py       # FastAPI routes (raw psycopg2)
├── database/
│   ├── database.py            # SQLAlchemy engine, ORM models, CRUD functions
│   └── schema.py              # Pydantic V2 request/response schemas
├── utils/
│   └── hash.py                # Password hashing utility
├── .env                       # PostgreSQL credentials (not committed)
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Two implementation approaches** — `main_orm.py` uses SQLAlchemy ORM (recommended), while `main_psycopg2.py` shows the same API built with raw SQL via psycopg2.
- **Multi-table auto-creation** — `make_table()` iterates a `TABLES` list and creates any missing tables on startup.
- **Context-manager sessions** — every CRUD function opens its own `Session(...)` via a `with` block, ensuring connections are released immediately.
- **Duplicate-email guard** — `create_user()` catches the unique-constraint violation and returns `None`, which the route converts into a 400 response.
- **EmailStr validation** — the `UserCreate` schema uses `pydantic[email]` to reject malformed email addresses at the request level.
- **Safe response model** — `UserSent` omits the `password` field so it is never leaked in API responses.
- **Password hashing** — passwords are hashed via `utils/hash.py` before being stored.

---

## Dependencies

See [`requirements.txt`](requirements.txt):

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `pydantic` | Data validation |
| `pydantic[email]` | `EmailStr` support |
| `psycopg2` | PostgreSQL driver |
| `uvicorn` | ASGI server |
| `python-dotenv` | `.env` loading |
| `SQLAlchemy` | ORM & database engine |
