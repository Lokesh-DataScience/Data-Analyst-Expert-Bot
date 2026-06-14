"""
api/auth.py

Full authentication module for DataAnalystBot (FastAPI).

Provides:
- User signup / login with bcrypt hashed passwords
- JWT access tokens
- SQLite-backed persistent user storage
- Password reset via email token (fastapi-mail)
- User profile update (name, email, password)
- Rate limiting helpers (used by main.py via slowapi)
- /auth/signup, /auth/login, /auth/me, /auth/logout
- /auth/forgot-password, /auth/reset-password
- /auth/update-profile

Install:
    pip install "passlib[bcrypt]" "python-jose[cryptography]" python-multipart
    pip install "bcrypt==4.0.1" fastapi-mail slowapi
"""

import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# ============================================================
# CONFIG
# ============================================================
SECRET_KEY              = os.getenv("JWT_SECRET_KEY", "change-this-in-prod-please")
ALGORITHM               = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))  # 7 days
RESET_TOKEN_EXPIRE_MINUTES  = 30  # password-reset link valid for 30 minutes
DB_PATH                 = os.getenv("AUTH_DB_PATH", "users.db")
FRONTEND_URL            = os.getenv("FRONTEND_URL", "http://localhost:5500")

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

auth_router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================
# DATABASE
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               TEXT PRIMARY KEY,
                name             TEXT NOT NULL,
                email            TEXT UNIQUE NOT NULL,
                hashed_password  TEXT NOT NULL,
                created_at       TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token       TEXT PRIMARY KEY,
                email       TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                used        INTEGER DEFAULT 0
            )
        """)


init_db()


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()


def get_user_by_id(user_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def create_user(name: str, email: str, hashed_password: str) -> sqlite3.Row:
    user_id    = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, name, email, hashed_password, created_at) VALUES (?,?,?,?,?)",
            (user_id, name, email, hashed_password, created_at),
        )
    return get_user_by_email(email)


def update_user(user_id: str, name: str, email: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET name = ?, email = ? WHERE id = ?",
            (name, email, user_id),
        )
    return get_user_by_id(user_id)


def update_user_password(user_id: str, hashed_password: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET hashed_password = ? WHERE id = ?",
            (hashed_password, user_id),
        )


def store_reset_token(email: str, token: str) -> None:
    expires_at = (datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)).isoformat()
    with get_db() as conn:
        # Invalidate any existing unused tokens for this email
        conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE email = ? AND used = 0",
            (email,),
        )
        conn.execute(
            "INSERT INTO password_reset_tokens (token, email, expires_at, used) VALUES (?,?,?,0)",
            (token, email, expires_at),
        )


def get_reset_token_row(token: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM password_reset_tokens WHERE token = ? AND used = 0",
            (token,),
        ).fetchone()


def mark_reset_token_used(token: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE token = ?",
            (token,),
        )


# ============================================================
# SCHEMAS
# ============================================================
class SignupRequest(BaseModel):
    name:     str      = Field(..., min_length=1, max_length=100)
    email:    EmailStr
    password: str      = Field(..., max_length=128)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str = Field(..., min_length=8, max_length=128)


class UpdateProfileRequest(BaseModel):
    name:             str            = Field(..., min_length=1, max_length=100)
    email:            EmailStr
    current_password: str
    new_password:     Optional[str]  = Field(None, max_length=128)


class UserOut(BaseModel):
    id:    str
    name:  str
    email: str


class TokenResponse(BaseModel):
    token:      str
    token_type: str = "bearer"
    user:       UserOut


# ============================================================
# PASSWORD / TOKEN HELPERS
# ============================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
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
    """Required auth — raises 401 if missing/invalid."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    email   = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def get_current_user_optional(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Optional auth — returns None for demo/unauthenticated requests."""
    if not token or token == "demo-token":
        return None
    try:
        return get_current_user(token)
    except HTTPException:
        return None


def get_rate_limit_key(request: Request) -> str:
    """
    Key for slowapi rate limiting.
    Uses authenticated user email when available, falls back to IP.
    """
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token and token != "demo-token":
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email   = payload.get("sub")
            if email:
                return f"user:{email}"
        except JWTError:
            pass
    # Fallback to IP
    forwarded = request.headers.get("X-Forwarded-For")
    return f"ip:{forwarded.split(',')[0].strip() if forwarded else request.client.host}"


# ============================================================
# OPTIONAL EMAIL HELPER
# ============================================================
async def send_reset_email(email: str, token: str) -> bool:
    """
    Sends a password-reset email.
    Requires MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_SERVER in .env.
    Returns True on success, False if env vars are missing (dev fallback).
    """
    required = ["MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "MAIL_SERVER"]
    if not all(os.getenv(k) for k in required):
        # Dev mode: print the link to the terminal instead
        reset_link = f"{FRONTEND_URL}/?reset_token={token}"
        print(f"\n[DEV] Password reset link for {email}:\n{reset_link}\n")
        return False

    try:
        from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

        conf = ConnectionConfig(
            MAIL_USERNAME   = os.getenv("MAIL_USERNAME"),
            MAIL_PASSWORD   = os.getenv("MAIL_PASSWORD"),
            MAIL_FROM       = os.getenv("MAIL_FROM"),
            MAIL_PORT       = int(os.getenv("MAIL_PORT", 587)),
            MAIL_SERVER     = os.getenv("MAIL_SERVER"),
            MAIL_STARTTLS   = True,
            MAIL_SSL_TLS    = False,
            USE_CREDENTIALS = True,
        )
        reset_link = f"{FRONTEND_URL}/?reset_token={token}"
        body = f"""
        <h2>DataAnalystBot — Password Reset</h2>
        <p>You requested a password reset. Click the link below to set a new password.
        This link expires in {RESET_TOKEN_EXPIRE_MINUTES} minutes.</p>
        <p><a href="{reset_link}">{reset_link}</a></p>
        <p>If you didn't request this, you can safely ignore this email.</p>
        """
        message = MessageSchema(
            subject    = "Reset your DataAnalystBot password",
            recipients = [email],
            body       = body,
            subtype    = MessageType.html,
        )
        fm = FastMail(conf)
        await fm.send_message(message)
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


# ============================================================
# ROUTES
# ============================================================
@auth_router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered.")

    user  = create_user(
        name            = req.name,
        email           = req.email,
        hashed_password = hash_password(req.password),
    )
    token = create_access_token({"sub": user["email"]})
    return TokenResponse(
        token = token,
        user  = UserOut(id=user["id"], name=user["name"], email=user["email"]),
    )


@auth_router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token({"sub": user["email"]})
    return TokenResponse(
        token = token,
        user  = UserOut(id=user["id"], name=user["name"], email=user["email"]),
    )


@auth_router.get("/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    return UserOut(**current_user)


@auth_router.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    """Stateless logout — client discards the token."""
    return {"message": "Logged out successfully."}


@auth_router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """
    Generates a reset token and emails it.
    Always returns 200 regardless of whether the email exists
    (to prevent user enumeration).
    """
    user = get_user_by_email(req.email)
    if user:
        token = secrets.token_urlsafe(32)
        store_reset_token(req.email, token)
        await send_reset_email(req.email, token)
    return {"message": "If that email is registered, a reset link has been sent."}


@auth_router.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    row = get_reset_token_row(req.token)
    if not row:
        raise HTTPException(status_code=400, detail="Invalid or already-used reset token.")

    # Check expiry
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.utcnow() > expires_at:
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")

    user = get_user_by_email(row["email"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    update_user_password(user["id"], hash_password(req.new_password))
    mark_reset_token_used(req.token)
    return {"message": "Password updated successfully. You can now log in."}


@auth_router.put("/update-profile", response_model=TokenResponse)
def update_profile(
    req:          UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Update name, email, and optionally password.
    Requires current_password to verify identity before any change.
    """
    user = get_user_by_email(current_user["email"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Verify current password
    if not verify_password(req.current_password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    # If email is changing, make sure it's not taken by another user
    if req.email != current_user["email"]:
        existing = get_user_by_email(req.email)
        if existing and existing["id"] != current_user["id"]:
            raise HTTPException(status_code=400, detail="That email is already in use.")

    # Update name + email
    updated = update_user(current_user["id"], req.name, req.email)

    # Optionally update password
    if req.new_password:
        if len(req.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
        update_user_password(current_user["id"], hash_password(req.new_password))

    # Re-issue token (email may have changed)
    token = create_access_token({"sub": updated["email"]})
    return TokenResponse(
        token = token,
        user  = UserOut(id=updated["id"], name=updated["name"], email=updated["email"]),
    )