"""
api/config.py

Centralized, environment-aware configuration.

Loads .env.development or .env.production based on the APP_ENV
variable (defaults to "development"), so you never accidentally
point a local dev run at production credentials or vice versa.

Usage:
    from api.config import settings
    settings.DATABASE_URL
    settings.JWT_SECRET_KEY
    settings.is_production

Set APP_ENV=production in your actual production environment
(e.g. as a real OS env var on the server — not in a committed file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "development").lower()
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE_MAP = {
    "development": BASE_DIR / ".env.development",
    "production":  BASE_DIR / ".env.production",
    "test":        BASE_DIR / ".env.test",
}

env_file = ENV_FILE_MAP.get(APP_ENV, ENV_FILE_MAP["development"])
print(f"Loading environment: {APP_ENV}")
print(f"Using env file: {env_file}")

# Fall back to a plain .env if the environment-specific file doesn't exist
# (keeps things working for anyone who hasn't split their env files yet).
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    fallback = BASE_DIR / ".env"
    if fallback.exists():
        load_dotenv(fallback, override=True)


def _require(key: str, default=None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable '{key}'. "
            f"Set it in {env_file.name if env_file.exists() else '.env'}."
        )
    return value


def _bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # ── Environment ──
    APP_ENV: str = APP_ENV
    is_production: bool = APP_ENV == "production"
    is_development: bool = APP_ENV == "development"

    # ── Database ──
    # Postgres connection string, e.g.:
    #   postgresql://user:password@localhost:5432/dataanalystbot
    # Falls back to a local SQLite file only in development, so the
    # app still runs out-of-the-box without standing up Postgres first.
    DATABASE_URL: str = _require(
        "DATABASE_URL",
        default=(
            None if APP_ENV == "production"
            else "sqlite:///./users.db"
        ),
        required=(APP_ENV == "production"),
    )

    # ── Auth / JWT ──
    JWT_SECRET_KEY: str = _require(
        "JWT_SECRET_KEY",
        default=("dev-only-insecure-secret-do-not-use-in-prod" if APP_ENV != "production" else None),
        required=(APP_ENV == "production"),
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))
    RESET_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", 30))

    # ── Cookie-based auth (see api/auth.py for usage) ──
    # When True, the JWT is also set as an HttpOnly secure cookie instead
    # of relying solely on the frontend storing it in localStorage.
    USE_SECURE_COOKIES: bool = _bool("USE_SECURE_COOKIES", default=(APP_ENV == "production"))
    COOKIE_NAME: str = os.getenv("COOKIE_NAME", "dab_session")
    COOKIE_DOMAIN: str = os.getenv("COOKIE_DOMAIN", None)
    COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "lax")  # "lax" | "strict" | "none"

    # ── HTTPS enforcement ──
    FORCE_HTTPS: bool = _bool("FORCE_HTTPS", default=(APP_ENV == "production"))

    # ── CORS ──
    CORS_ALLOWED_ORIGINS: list = [
        o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()
    ]

    # ── Frontend URL (used in password reset emails, etc.) ──
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5500")

    # ── Email (password reset) ──
    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD")
    MAIL_FROM:     str = os.getenv("MAIL_FROM")
    MAIL_SERVER:   str = os.getenv("MAIL_SERVER")
    MAIL_PORT:     int = int(os.getenv("MAIL_PORT", 587))

    # ── Groq / LLM ──
    GROQ_API_KEY: str = _require("GROQ_API_KEY", required=(APP_ENV == "production"))

    # ── Logging ──
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if APP_ENV != "production" else "INFO")
    LOG_JSON:  bool = _bool("LOG_JSON", default=(APP_ENV == "production"))

    # ── Cache / vector store paths ──
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./.cache")
    AUTH_DB_PATH: str = os.getenv("AUTH_DB_PATH", "users.db")  # legacy SQLite fallback path

    def summary(self) -> dict:
        """Returns a redacted snapshot of active config, safe to log on startup."""
        def redact(v):
            return "***" if v else None

        return {
            "APP_ENV": self.APP_ENV,
            "env_file": str(env_file) if env_file.exists() else "(none — using OS env / defaults)",
            "DATABASE_URL": self.DATABASE_URL.split("@")[-1] if self.DATABASE_URL and "@" in self.DATABASE_URL else self.DATABASE_URL,
            "JWT_SECRET_KEY": redact(self.JWT_SECRET_KEY),
            "USE_SECURE_COOKIES": self.USE_SECURE_COOKIES,
            "FORCE_HTTPS": self.FORCE_HTTPS,
            "CORS_ALLOWED_ORIGINS": self.CORS_ALLOWED_ORIGINS,
            "GROQ_API_KEY": redact(self.GROQ_API_KEY),
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_JSON": self.LOG_JSON,
        }


settings = Settings()