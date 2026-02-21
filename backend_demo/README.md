# Backend Demo — FastAPI + PostgreSQL CRUD

A standalone REST API for **posts**, **users**, and **authentication**, built with FastAPI, SQLAlchemy ORM, and Pydantic V2.
Routes are organized into dedicated modules under `routes/` and mounted via `include_router()`.

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
| `GET` | `/users/{id}` | Get a user by ID |
| `POST` | `/login` | Authenticate & get token |

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

### POST /login

- **Request body** — same `UserCreate` schema:
  ```json
  { "email": "user@example.com", "password": "secret" }
  ```
- Looks up the user by email, then verifies the plaintext password against the stored **bcrypt** hash using `passlib`'s `pwd_context.verify()`.
- Returns **404** if the email is not found, **401** if the password doesn't match.

---

## Project Structure

```
backend_demo/
├── app/
│   └── main_orm.py            # App entry point — creates FastAPI instance, mounts routers
├── routes/
│   ├── posts.py               # /posts CRUD endpoints (APIRouter)
│   ├── users.py               # /users registration & lookup (APIRouter)
│   └── auth.py                # /login authentication (APIRouter)
├── database/
│   ├── database.py            # SQLAlchemy engine, ORM models (Post, User)
│   └── schema.py              # Pydantic V2 request/response schemas
├── utils/
│   ├── query_db.py            # All DB queries (context-managed sessions)
│   └── hash.py                # Password hashing utility
├── .env                       # PostgreSQL credentials (not committed)
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Router-based architecture** — routes are split into `routes/posts.py` and `routes/users.py` using FastAPI's `APIRouter`, then mounted in `main_orm.py` via `app.include_router()`. This keeps the entry point lean and each resource self-contained.
- **Multi-table auto-creation** — `make_table()` iterates a `TABLES` list and creates any missing tables on startup.
- **Context-manager sessions** — every CRUD function in `utils/query_db.py` opens its own `Session(...)` via a `with` block, ensuring connections are released immediately.
- **Duplicate-email guard** — `create_user()` catches the unique-constraint violation and returns `None`, which the route converts into a 400 response.
- **EmailStr validation** — the `UserCreate` schema uses `pydantic[email]` to reject malformed email addresses at the request level.
- **Safe response model** — `UserSent` omits the `password` field so it is never leaked in API responses.
- **Password hashing** — passwords are hashed with **bcrypt** via `utils/hash.py` (passlib) before being stored. Always use `hash_password()` — never Python's built-in `hash()`, which returns a non-cryptographic integer.
- **Password verification** — `verify_password()` uses `pwd_context.verify()` to compare, which correctly extracts the bcrypt salt. Direct string comparison does not work because bcrypt produces a different hash each time.

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
| `passlib[bcrypt]` | Password hashing (bcrypt) |
