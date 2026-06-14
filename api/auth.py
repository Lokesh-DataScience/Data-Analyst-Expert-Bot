"""
api/auth.py

Full authentication module for DataAnalystBot (FastAPI).

Provides:
- User signup / login with hashed passwords (bcrypt)
- JWT access tokens
- SQLite-backed persistent user storage (no external DB required)
- Dependencies to protect routes (required + optional auth)
- /auth/signup, /auth/login, /auth/me, /auth/logout (token-based, stateless)

Usage in api/main.py:

    from api.auth import router as auth_router, get_current_user, get_current_user_optional
    app.include_router(auth_router)

    @app.post("/multi-upload")
    def multi_upload_endpoint(request: ..., current_user: dict = Depends(get_current_user)):
        ...
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field


# ============================================================
# CONFIG
# ============================================================
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-prod-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))  # 7 days

DB_PATH = os.getenv("AUTH_DB_PATH", "users.db")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# auto_error=False so we can build an "optional auth" dependency too
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================
# DATABASE (SQLite — no extra services required)
# ============================================================
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


# Initialize on import
init_db()


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        return cur.fetchone()


def create_user(name: str, email: str, hashed_password: str) -> sqlite3.Row:
    user_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, name, email, hashed_password, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, email, hashed_password, created_at),
        )
    return get_user_by_email(email)


# ============================================================
# SCHEMAS
# ============================================================
class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserOut


# ============================================================
# PASSWORD / TOKEN HELPERS
# ============================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================
# DEPENDENCIES
# ============================================================
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Use this for routes that REQUIRE authentication.
    Raises 401 if no/invalid token.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def get_current_user_optional(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """
    Use this for routes that work with or without auth
    (useful while migrating existing endpoints).
    Returns None if no/invalid token instead of raising.
    """
    if not token:
        return None
    try:
        return get_current_user(token)
    except HTTPException:
        return None


# ============================================================
# ROUTES
# ============================================================
@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest):
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(req.password)
    user = create_user(name=req.name, email=req.email, hashed_password=hashed)

    token = create_access_token({"sub": user["email"]})
    return TokenResponse(
        token=token,
        user=UserOut(id=user["id"], name=user["name"], email=user["email"]),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user["email"]})
    return TokenResponse(
        token=token,
        user=UserOut(id=user["id"], name=user["name"], email=user["email"]),
    )


@router.get("/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    return UserOut(**current_user)


@router.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    """
    JWTs are stateless, so 'logout' is handled client-side by
    discarding the token. This endpoint exists for API symmetry
    and can be extended with a token-blocklist if needed.
    """
    return {"message": "Logged out successfully"}