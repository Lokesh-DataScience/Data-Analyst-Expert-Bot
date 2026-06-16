from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.chains.rag_chain import build_chain, build_contextual_chain, add_documents_to_user_store
from typing import List, Dict, Optional
from api.memory.session_memory import get_memory
from langchain_core.messages import HumanMessage, AIMessage
from groq import Groq
import base64
import tempfile
import os
from api.loaders.docs_loader import DocumentLoader
from diskcache import Cache
import hashlib
import pandas as pd
from api.utils.data_analyzer import DataAnalyzer
from api.utils.data_augmentor import DataAugmentor
from api.auth import (
    auth_router,
    get_current_user,
    get_current_user_optional,
    get_rate_limit_key,
)

# ============================================================
# RATE LIMITING (slowapi)
# pip install slowapi
# ============================================================
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_rate_limit_key, default_limits=["200/minute"])

# ============================================================
# CACHE & CHAT STORE
# ============================================================
cache      = Cache(directory="./.cache")
chat_store = cache.get("chat_store", {})


def hash_data(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def user_session_key(email: str, session_id: str) -> str:
    """Scope every chat session to the authenticated user."""
    return f"{email}:{session_id}"


# ============================================================
# APP
# ============================================================
app = FastAPI(title="DataAnalystBot API", version="2.0.0")

# Rate limiter must be set on app state before adding exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

LOADER = DocumentLoader()

# Per-user retrieval chains are built lazily and cached in-process,
# since loading a FAISS index has nontrivial cost. Falls back to the
# shared base chain for anonymous/demo users.
_user_chain_cache: dict = {}


def get_chain_for_user(user_email: str):
    key = user_email or "anonymous"
    if key not in _user_chain_cache:
        _user_chain_cache[key] = build_chain(user_email=key)
    return _user_chain_cache[key]


def invalidate_user_chain(user_email: str):
    """Call after ingesting new documents so the next request re-loads the updated index."""
    _user_chain_cache.pop(user_email, None)

# ============================================================
# INPUT SCHEMAS
# ============================================================
class QueryRequest(BaseModel):
    question:    str
    chat_history: Optional[List[Dict]] = None
    session_id:  Optional[str]         = None


class DataAnalysisRequest(BaseModel):
    csv_base64:  str
    csv_filename: str
    session_id:  Optional[str] = None


class MultiUploadQueryRequest(BaseModel):
    question:     str
    session_id:   Optional[str] = None
    chat_history: Optional[List[Dict]] = None
    image_base64: Optional[str] = None
    image_type:   Optional[str] = None
    csv_base64:   Optional[str] = None
    csv_filename: Optional[str] = None
    pdf_base64:   Optional[str] = None
    pdf_filename: Optional[str] = None


class SQLGenerationRequest(BaseModel):
    description: str
    db_schema:   Optional[str] = None
    db_type:     Optional[str] = "PostgreSQL"
    query_type:  Optional[str] = None
    session_id:  Optional[str] = None


class DataAugmentationRequest(BaseModel):
    csv_base64:              str
    csv_filename:            str
    session_id:              Optional[str] = None
    apply_imputation:        bool = True
    apply_outlier_treatment: bool = True
    apply_synthetic_rows:    bool = False
    apply_deduplication:     bool = True
    apply_transformations:   bool = False


# ============================================================
# MEMORY / HISTORY HELPERS
# ============================================================
def update_memory_and_history(memory, chat_history, session_key: str) -> str:
    memory.messages.clear()
    existing_history = chat_store.get(session_key, [])
    updated_history  = []

    for msg in chat_history or []:
        if msg["type"] == "human":
            memory.add_message(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            memory.add_message(AIMessage(content=msg["content"]))
        entry = {"type": msg["type"], "content": msg["content"]}
        if "file" in msg:
            entry["file"] = msg["file"]
        updated_history.append(entry)

    chat_store[session_key] = existing_history + updated_history
    cache["chat_store"]     = chat_store

    return "\n".join([f"{m.type}: {m.content}" for m in memory.messages])


# ============================================================
# CONTEXT HELPERS
# ============================================================
def get_image_context(image_base64: str, image_type: str) -> str:
    key = hash_data(image_base64)
    if key in cache:
        return cache[key]
    client = Groq()
    resp   = client.chat.completions.create(
        model    = "meta-llama/llama-4-maverick-17b-128e-instruct",
        messages = [{
            "role":    "user",
            "content": [
                {"type": "text", "text": "Describe this image for data analysis:"},
                {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_base64}"}},
            ],
        }],
    )
    result       = resp.choices[0].message.content
    cache[key]   = result
    return result


def get_csv_context(csv_base64: str, question: str = "", user_email: str = None) -> str:
    """
    Returns a compact pandas summary instead of a raw row-dump,
    keeping token usage low while giving the LLM full context.
    Also ingests the CSV rows into the user's personal vector store
    so future chat questions can retrieve from it semantically.
    """
    key = hash_data(csv_base64 + question)
    if key in cache:
        return cache[key]

    csv_bytes = base64.b64decode(csv_base64)
    df        = pd.read_csv(__import__("io").BytesIO(csv_bytes))

    context = (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns: {list(df.columns)}\n"
        f"Data types:\n{df.dtypes.to_string()}\n\n"
        f"First 10 rows:\n{df.head(10).to_string()}\n\n"
        f"Summary statistics:\n{df.describe(include='all').to_string()}"
    )
    cache[key] = context

    # Ingest into the user's personal knowledge base
    if user_email and user_email != "anonymous":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
                tmp.write(csv_bytes)
                tmp_path = tmp.name
            try:
                docs = LOADER.load_csv(tmp_path)
                add_documents_to_user_store(user_email, docs)
                invalidate_user_chain(user_email)
            finally:
                os.unlink(tmp_path)
        except Exception as ingest_err:
            print(f"[vectorstore] CSV ingestion skipped: {ingest_err}")

    return context


def get_pdf_context(pdf_base64: str, question: str = "", user_email: str = None) -> str:
    """
    Extracts text from the uploaded PDF for immediate context, and
    ingests the page-level documents into the user's personal vector
    store so the PDF becomes part of their permanent knowledge base.
    """
    key = hash_data(pdf_base64 + question)
    if key in cache:
        return cache[key]
    pdf_bytes = base64.b64decode(pdf_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        docs     = LOADER.load_pdf(tmp_path)
        result   = "\n".join([d.page_content for d in docs])
        cache[key] = result

        if user_email and user_email != "anonymous":
            try:
                add_documents_to_user_store(user_email, docs)
                invalidate_user_chain(user_email)
            except Exception as ingest_err:
                print(f"[vectorstore] PDF ingestion skipped: {ingest_err}")

        return result
    finally:
        os.unlink(tmp_path)


# ============================================================
# SERIALIZATION HELPERS
# ============================================================
PLOTLY_BOOL_PROPS = {
    "showarrow","automargin","showlegend","matches","visible","autosize",
    "showticklabels","showgrid","zeroline","showline","mirror","ticks",
    "showspikes","showaxeslabels","fixedrange","constraintoward",
    "connectgaps","fill","showscale","reversescale","autocolorscale",
    "showcolorbar","transpose","zauto","ncontours","autocontour",
    "autobinx","autobiny","standoff","clicktoshow","captureevents",
    "autorange","outlinewidth","borderwidth","thickness","len",
    "fillcolor","opacity",
}


def fix_plotly_bools(obj, parent_key=None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in PLOTLY_BOOL_PROPS and isinstance(v, int):
                out[k] = bool(v)
            elif k in PLOTLY_BOOL_PROPS and isinstance(v, str):
                out[k] = v.lower() in ("true", "1")
            else:
                out[k] = fix_plotly_bools(v, k)
        return out
    if isinstance(obj, list):
        return [fix_plotly_bools(i, parent_key) for i in obj]
    if parent_key in PLOTLY_BOOL_PROPS and isinstance(obj, int):
        return bool(obj)
    return obj


def clean_dict_for_json(obj):
    if isinstance(obj, dict):
        return {
            k: (bool(cv) if k in PLOTLY_BOOL_PROPS and isinstance(cv := clean_dict_for_json(v), int) else clean_dict_for_json(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [clean_dict_for_json(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return clean_dict_for_json(obj.tolist())
    if isinstance(obj, (int, float)):
        try:
            return None if (pd.isna(obj) or not np.isfinite(obj)) else obj
        except Exception:
            return obj
    return obj


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        try:
            return None if (pd.isna(obj) or not np.isfinite(obj)) else float(obj)
        except Exception:
            return float(obj)
    if isinstance(obj, (np.ndarray, pd.Series)):
        return convert_to_serializable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict("records"))
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return obj


def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(
        lambda col: col.fillna(col.median()) if col.dtype in [np.float64, np.int64] else col
    )
    return df


# ============================================================
# CHAT
# ============================================================
@app.post("/chat")
@limiter.limit("30/minute")
def chat_endpoint(
    request:      Request,
    body:         QueryRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    email       = (current_user or {}).get("email", "anonymous")
    session_key = user_session_key(email, body.session_id or "default")
    memory      = get_memory(session_key)
    history_str = update_memory_and_history(memory, body.chat_history, session_key)

    chain    = get_chain_for_user(email)
    response = chain.invoke({"input": body.question, "chat_history": history_str})
    answer   = response.get("answer", "No response")

    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({"type": "ai", "content": answer})
    cache["chat_store"] = chat_store
    return {"response": answer}


# ============================================================
# MULTI-UPLOAD  (rate-limited: 20/minute per user)
# ============================================================
@app.post("/multi-upload")
@limiter.limit("20/minute")
def multi_upload_endpoint(
    request:      Request,
    body:         MultiUploadQueryRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    email       = (current_user or {}).get("email", "anonymous")
    session_key = user_session_key(email, body.session_id or "default")
    memory      = get_memory(session_key)
    history_str = update_memory_and_history(memory, body.chat_history, session_key)

    contexts = []
    if body.image_base64 and body.image_type:
        contexts.append(f"Image context: {get_image_context(body.image_base64, body.image_type)}")
    if body.csv_base64 and body.csv_filename:
        contexts.append(f"CSV context: {get_csv_context(body.csv_base64, body.question or '', user_email=email)}")
    if body.pdf_base64 and body.pdf_filename:
        contexts.append(f"PDF context: {get_pdf_context(body.pdf_base64, body.question or '', user_email=email)}")

    combined = "\n\n".join(contexts) if contexts else None

    if combined:
        chain    = build_contextual_chain()
        response = chain.invoke({"input": body.question, "chat_history": history_str, "context": combined})
        answer   = response.content if hasattr(response, "content") else str(response)
    else:
        chain    = get_chain_for_user(email)
        response = chain.invoke({"input": body.question, "chat_history": history_str})
        answer   = response.get("answer", "No response")

    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({"type": "ai", "content": answer})
    cache["chat_store"] = chat_store
    return {"response": answer}


# ============================================================
# RECENT CHATS  (scoped per user)
# ============================================================
@app.get("/recent-chats/{session_id}")
def get_recent_chats(
    session_id:   str,
    current_user: dict = Depends(get_current_user_optional),
):
    email = (current_user or {}).get("email", "anonymous")
    key   = user_session_key(email, session_id)
    return {"chat_history": chat_store.get(key, [])}


@app.get("/recent-chat-titles")
def get_recent_chat_titles(
    current_user: dict = Depends(get_current_user_optional),
):
    email  = (current_user or {}).get("email", "anonymous")
    prefix = f"{email}:"
    titles = []
    for key, history in chat_store.items():
        if not key.startswith(prefix):
            continue
        session_id = key[len(prefix):]
        for msg in history:
            if msg["type"] == "human":
                titles.append({"session_id": session_id, "title": msg["content"]})
                break
    return {"sessions": titles}


@app.post("/save-chat")
def save_chat_endpoint(
    data:         dict,
    current_user: dict = Depends(get_current_user_optional),
):
    email      = (current_user or {}).get("email", "anonymous")
    session_id = data.get("session_id")
    history    = data.get("chat_history", [])
    if session_id and history:
        key                  = user_session_key(email, session_id)
        chat_store[key]      = history
        cache["chat_store"]  = chat_store
        return {"success": True}
    return {"success": False, "error": "Missing session_id or chat_history"}


# ============================================================
# ANALYZE DATA  (rate-limited: 20/minute per user)
# ============================================================
@app.post("/analyze-data")
@limiter.limit("20/minute")
def analyze_data_endpoint(
    request: Request,
    body:    DataAnalysisRequest,
    _:       dict = Depends(get_current_user_optional),
):
    try:
        csv_bytes = base64.b64decode(body.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        try:
            df       = pd.read_csv(tmp_path)
            analyzer = DataAnalyzer()

            cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
            cleaned_df               = clean_for_json(cleaned_df)
            insights                 = analyzer.generate_insights(cleaned_df, cleaning_log)
            statistical_summary      = analyzer.statistical_analysis(cleaned_df)

            if isinstance(statistical_summary, dict):
                statistical_summary = {
                    k: (v if isinstance(v, (int, float)) and pd.notna(v) and np.isfinite(v) else None)
                    for k, v in statistical_summary.items()
                }

            plots      = analyzer.create_visualizations(cleaned_df)
            plots_json = {}
            for name, fig in plots.items():
                try:
                    plots_json[name] = fix_plotly_bools(clean_dict_for_json(fig.to_dict()))
                except Exception as pe:
                    plots_json[name] = {"error": str(pe)}

            return JSONResponse(content={
                "success":              True,
                "original_shape":       df.shape,
                "cleaned_shape":        cleaned_df.shape,
                "cleaning_log":         convert_to_serializable(cleaning_log),
                "insights":             convert_to_serializable(insights),
                "statistical_summary":  convert_to_serializable(statistical_summary),
                "visualizations":       plots_json,
                "column_info":          convert_to_serializable({
                    "original_columns": list(df.columns),
                    "cleaned_columns":  list(cleaned_df.columns),
                    "data_types":       cleaned_df.dtypes.astype(str).to_dict(),
                    "missing_values":   cleaned_df.isnull().sum().to_dict(),
                    "unique_counts":    cleaned_df.nunique().to_dict(),
                }),
                "sample_data": {
                    "original": convert_to_serializable(df.head()),
                    "cleaned":  convert_to_serializable(cleaned_df.head()),
                },
            })
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return JSONResponse(status_code=429, content={"success": False, "error": "API quota exceeded", "message": msg})
        if "csv" in msg.lower() or "pandas" in msg.lower():
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid CSV", "message": msg})
        return JSONResponse(status_code=500, content={"success": False, "error": "Analysis failed", "message": msg})


# ============================================================
# CLEAN DATA
# ============================================================
@app.post("/clean-data")
@limiter.limit("20/minute")
def clean_data_endpoint(
    request: Request,
    body:    DataAnalysisRequest,
    _:       dict = Depends(get_current_user_optional),
):
    try:
        csv_bytes = base64.b64decode(body.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        try:
            df       = pd.read_csv(tmp_path)
            analyzer = DataAnalyzer()

            cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
            statistical_summary      = analyzer.statistical_analysis(cleaned_df)
            plots                    = analyzer.create_visualizations(cleaned_df)
            plots_json               = {}
            for name, fig in plots.items():
                try:
                    plots_json[name] = fix_plotly_bools(fig.to_dict())
                except Exception as pe:
                    plots_json[name] = {"error": str(pe)}

            return JSONResponse(content={
                "success":             True,
                "original_shape":      df.shape,
                "cleaned_shape":       cleaned_df.shape,
                "cleaning_log":        cleaning_log,
                "statistical_summary": statistical_summary,
                "visualizations":      plots_json,
                "column_info": {
                    "original_columns": list(df.columns),
                    "cleaned_columns":  list(cleaned_df.columns),
                    "data_types":       cleaned_df.dtypes.astype(str).to_dict(),
                    "missing_values":   cleaned_df.isnull().sum().to_dict(),
                    "unique_counts":    cleaned_df.nunique().to_dict(),
                },
                "sample_data": {
                    "original": df.head().to_dict("records"),
                    "cleaned":  cleaned_df.head().to_dict("records"),
                },
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": "Cleaning failed", "message": str(e)})


# ============================================================
# SQL GENERATOR  (rate-limited: 20/minute per user)
# ============================================================
@app.post("/generate-sql")
@limiter.limit("20/minute")
def generate_sql_endpoint(
    request: Request,
    body:    SQLGenerationRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    try:
        schema_section     = f"\n\nDatabase Schema:\n{body.db_schema}" if body.db_schema else ""
        query_type_section = f"\nQuery Type: {body.query_type}" if body.query_type else ""

        prompt = f"""You are an expert SQL developer. Generate a SQL query based on the following:

Database Type: {body.db_type}{query_type_section}{schema_section}

User Request: {body.description}

Respond ONLY in the following JSON format (no markdown, no extra text):
{{
  "sql_query": "<the SQL query>",
  "explanation": "<plain English explanation of what the query does and why>",
  "suggestions": "<performance tips, index recommendations, or alternative approaches>"
}}"""

        client = Groq()
        resp   = client.chat.completions.create(
            model       = "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature = 0.1,
            messages    = [
                {"role": "system", "content": "You are an expert SQL developer. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"sql_query": raw, "explanation": "Raw output returned.", "suggestions": ""}

        # Save to user session history
        if body.session_id:
            email       = (current_user or {}).get("email", "anonymous")
            session_key = user_session_key(email, body.session_id)
            chat_store.setdefault(session_key, [])
            chat_store[session_key].append({"type": "human", "content": f"[SQL] {body.description}"})
            chat_store[session_key].append({"type": "ai",    "content": result.get("sql_query", "")})
            cache["chat_store"] = chat_store

        return JSONResponse(content={
            "success":     True,
            "sql_query":   result.get("sql_query", ""),
            "explanation": result.get("explanation", ""),
            "suggestions": result.get("suggestions", ""),
            "db_type":     body.db_type,
            "query_type":  body.query_type,
        })

    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return JSONResponse(status_code=429, content={"success": False, "error": "API quota exceeded", "message": msg})
        return JSONResponse(status_code=500, content={"success": False, "error": "SQL generation failed", "message": msg})


# ============================================================
# DIAGNOSE DATA
# ============================================================
@app.post("/diagnose-data")
@limiter.limit("20/minute")
def diagnose_data_endpoint(
    request: Request,
    body:    DataAugmentationRequest,
    _:       dict = Depends(get_current_user_optional),
):
    try:
        csv_bytes = base64.b64decode(body.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        try:
            df        = pd.read_csv(tmp_path)
            augmentor = DataAugmentor()
            diagnosis = augmentor.diagnose(df)
            return JSONResponse(content={"success": True, "diagnosis": convert_to_serializable(diagnosis)})
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": "Diagnosis failed", "message": str(e)})


# ============================================================
# AUGMENT DATA
# ============================================================
@app.post("/augment-data")
@limiter.limit("20/minute")
def augment_data_endpoint(
    request: Request,
    body:    DataAugmentationRequest,
    _:       dict = Depends(get_current_user_optional),
):
    try:
        csv_bytes = base64.b64decode(body.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        try:
            df        = pd.read_csv(tmp_path)
            augmentor = DataAugmentor()
            options   = {
                "apply_imputation":        body.apply_imputation,
                "apply_outlier_treatment": body.apply_outlier_treatment,
                "apply_synthetic_rows":    body.apply_synthetic_rows,
                "apply_deduplication":     body.apply_deduplication,
                "apply_transformations":   body.apply_transformations,
            }
            augmented_df, change_log = augmentor.augment(df, options)
            augmented_b64 = base64.b64encode(
                augmented_df.to_csv(index=False).encode("utf-8")
            ).decode("utf-8")

            return JSONResponse(content={
                "success":               True,
                "original_shape":        list(df.shape),
                "augmented_shape":       list(augmented_df.shape),
                "change_log":            convert_to_serializable(change_log),
                "augmented_csv_base64":  augmented_b64,
                "augmented_filename":    f"augmented_{body.csv_filename}",
                "sample_original":       convert_to_serializable(df.head()),
                "sample_augmented":      convert_to_serializable(augmented_df.head()),
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": "Augmentation failed", "message": str(e)})