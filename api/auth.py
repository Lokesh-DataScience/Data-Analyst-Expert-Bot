"""
api/auth.py

Full authentication module for DataAnalystBot (FastAPI).

Changes in this version:
- User storage moved from raw sqlite3 to SQLAlchemy, which works
  against both Postgres (production) and SQLite (local dev fallback)
  via a single DATABASE_URL setting — see api/config.py.
- JWT can now also be set as an HttpOnly, Secure, SameSite cookie
  (in addition to being returned in the JSON body), controlled by
  USE_SECURE_COOKIES. This reduces XSS exposure for any client that
  reads the cookie instead of storing the token in localStorage.
- Replaced print() debugging with structured logging.

Install:
    pip install "passlib[bcrypt]" "python-jose[cryptography]" python-multipart
    pip install "bcrypt==4.0.1" fastapi-mail slowapi
    pip install sqlalchemy psycopg2-binary
"""

import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, String, DateTime, Integer, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from api.config import settings
from api.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================
# CONFIG (pulled from centralized settings)
# ============================================================
SECRET_KEY                  = settings.JWT_SECRET_KEY
ALGORITHM                   = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
RESET_TOKEN_EXPIRE_MINUTES  = settings.RESET_TOKEN_EXPIRE_MINUTES
FRONTEND_URL                = settings.FRONTEND_URL

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

auth_router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================
# DATABASE — SQLAlchemy (Postgres in prod, SQLite fallback in dev)
# ============================================================
def _normalize_db_url(url: str) -> str:
    """
    Some hosts (e.g. Heroku-style providers) hand out URLs starting
    with postgres:// — SQLAlchemy 1.4+/2.0 requires postgresql://.
    """
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


DATABASE_URL = _normalize_db_url(settings.DATABASE_URL)

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Sensible pool defaults for Postgres under concurrent load
    engine_kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)

engine       = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()

logger.info(
    "Database engine initialized",
    extra={"dialect": engine.dialect.name, "app_env": settings.APP_ENV},
)


class User(Base):
    __tablename__ = "users"

    id              = Column(String(36), primary_key=True)
    name            = Column(String(100), nullable=False)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    token      = Column(String(128), primary_key=True)
    email      = Column(String(255), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used       = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _db_session() -> Session:
    """Non-dependency-injected session for use in plain helper functions."""
    return SessionLocal()


# ============================================================
# DB HELPERS
# ============================================================
def get_user_by_email(email: str) -> Optional[User]:
    db = _db_session()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def get_user_by_id(user_id: str) -> Optional[User]:
    db = _db_session()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


def create_user(name: str, email: str, hashed_password: str) -> User:
    db = _db_session()
    try:
        user = User(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("User created", extra={"email": email, "user_id": user.id})
        return user
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_user(user_id: str, name: str, email: str) -> Optional[User]:
    db = _db_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        user.name  = name
        user.email = email
        db.commit()
        db.refresh(user)
        return user
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_user_password(user_id: str, hashed_password: str) -> None:
    db = _db_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.hashed_password = hashed_password
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def store_reset_token(email: str, token: str) -> None:
    db = _db_session()
    try:
        db.query(PasswordResetToken).filter(
            PasswordResetToken.email == email,
            PasswordResetToken.used == 0,
        ).update({"used": 1})

        expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
        db.add(PasswordResetToken(token=token, email=email, expires_at=expires_at, used=0))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_reset_token_row(token: str) -> Optional[PasswordResetToken]:
    db = _db_session()
    try:
        return db.query(PasswordResetToken).filter(
            PasswordResetToken.token == token,
            PasswordResetToken.used == 0,
        ).first()
    finally:
        db.close()


def mark_reset_token_used(token: str) -> None:
    db = _db_session()
    try:
        db.query(PasswordResetToken).filter(PasswordResetToken.token == token).update({"used": 1})
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


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
# SECURE COOKIE HELPERS
# ============================================================
def set_auth_cookie(response: Response, token: str) -> None:
    """
    Sets the JWT as an HttpOnly, Secure cookie when USE_SECURE_COOKIES
    is enabled (defaults to on in production). The token is still also
    returned in the JSON body for clients using localStorage/Bearer auth —
    this is additive, not a replacement, so existing frontend code keeps
    working unchanged while gaining the option to rely on the cookie.
    """
    if not settings.USE_SECURE_COOKIES:
        return
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.is_production,   # only require HTTPS in prod; allows local http testing
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    if not settings.USE_SECURE_COOKIES:
        return
    response.delete_cookie(key=settings.COOKIE_NAME, domain=settings.COOKIE_DOMAIN, path="/")


def _extract_token(request: Request, bearer_token: Optional[str]) -> Optional[str]:
    """Prefers the Authorization header; falls back to the secure cookie."""
    if bearer_token:
        return bearer_token
    if settings.USE_SECURE_COOKIES:
        return request.cookies.get(settings.COOKIE_NAME)
    return None


# ============================================================
# DEPENDENCIES
# ============================================================
def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> dict:
    """Required auth — raises 401 if missing/invalid. Checks header, then cookie."""
    token = _extract_token(request, token)
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
    return {"id": user.id, "name": user.name, "email": user.email}


def get_current_user_optional(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> Optional[dict]:
    """Optional auth — returns None for demo/unauthenticated requests."""
    token = _extract_token(request, token)
    if not token or token == "demo-token":
        return None
    try:
        payload = decode_token(token)
        email   = payload.get("sub")
        if not email:
            return None
        user = get_user_by_email(email)
        if not user:
            return None
        return {"id": user.id, "name": user.name, "email": user.email}
    except HTTPException:
        return None


def get_rate_limit_key(request: Request) -> str:
    """Key for slowapi rate limiting — authenticated user email, else IP."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token and settings.USE_SECURE_COOKIES:
        token = request.cookies.get(settings.COOKIE_NAME, "")
    if token and token != "demo-token":
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email   = payload.get("sub")
            if email:
                return f"user:{email}"
        except JWTError:
            pass
    forwarded = request.headers.get("X-Forwarded-For")
    return f"ip:{forwarded.split(',')[0].strip() if forwarded else request.client.host}"


# ============================================================
# EMAIL HELPER
# ============================================================
async def send_reset_email(email: str, token: str) -> bool:
    required = [settings.MAIL_USERNAME, settings.MAIL_PASSWORD, settings.MAIL_FROM, settings.MAIL_SERVER]
    if not all(required):
        reset_link = f"{FRONTEND_URL}/?reset_token={token}"
        logger.warning(
            "Email not configured — printing reset link instead",
            extra={"email": email, "reset_link": reset_link},
        )
        return False

    try:
        from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

        conf = ConnectionConfig(
            MAIL_USERNAME   = settings.MAIL_USERNAME,
            MAIL_PASSWORD   = settings.MAIL_PASSWORD,
            MAIL_FROM       = settings.MAIL_FROM,
            MAIL_PORT       = settings.MAIL_PORT,
            MAIL_SERVER     = settings.MAIL_SERVER,
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
        logger.info("Password reset email sent", extra={"email": email})
        return True
    except Exception as e:
        logger.error("Failed to send reset email", exc_info=True, extra={"email": email})
        return False


# ============================================================
# ROUTES
# ============================================================
@auth_router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, response: Response):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered.")

    user  = create_user(
        name            = req.name,
        email           = req.email,
        hashed_password = hash_password(req.password),
    )
    token = create_access_token({"sub": user.email})
    set_auth_cookie(response, token)

    return TokenResponse(
        token = token,
        user  = UserOut(id=user.id, name=user.name, email=user.email),
    )


@auth_router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, response: Response):
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user.hashed_password):
        logger.warning("Failed login attempt", extra={"email": req.email})
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token({"sub": user.email})
    set_auth_cookie(response, token)
    logger.info("User logged in", extra={"email": user.email})

    return TokenResponse(
        token = token,
        user  = UserOut(id=user.id, name=user.name, email=user.email),
    )


@auth_router.get("/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    return UserOut(**current_user)


@auth_router.post("/logout")
def logout(response: Response, current_user: dict = Depends(get_current_user)):
    clear_auth_cookie(response)
    logger.info("User logged out", extra={"email": current_user.get("email")})
    return {"message": "Logged out successfully."}


@auth_router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
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

    if datetime.utcnow() > row.expires_at:
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")

    user = get_user_by_email(row.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    update_user_password(user.id, hash_password(req.new_password))
    mark_reset_token_used(req.token)
    logger.info("Password reset completed", extra={"email": user.email})
    return {"message": "Password updated successfully. You can now log in."}


@auth_router.put("/update-profile", response_model=TokenResponse)
def update_profile(
    req:          UpdateProfileRequest,
    response:     Response,
    current_user: dict = Depends(get_current_user),
):
    user = get_user_by_email(current_user["email"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not verify_password(req.current_password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    if req.email != current_user["email"]:
        existing = get_user_by_email(req.email)
        if existing and existing.id != current_user["id"]:
            raise HTTPException(status_code=400, detail="That email is already in use.")

    updated = update_user(current_user["id"], req.name, req.email)

    if req.new_password:
        if len(req.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
        update_user_password(current_user["id"], hash_password(req.new_password))

    token = create_access_token({"sub": updated.email})
    set_auth_cookie(response, token)
    logger.info("Profile updated", extra={"email": updated.email})

    return TokenResponse(
        token = token,
        user  = UserOut(id=updated.id, name=updated.name, email=updated.email),
    )