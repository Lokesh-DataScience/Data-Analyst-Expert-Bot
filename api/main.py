from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.chains.rag_chain import build_chain, build_contextual_chain
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

# Set up cache directory
cache = Cache(directory="./.cache")
# Store chat histories per session
chat_store = cache.get("chat_store", {})

# Util: Create stable hash key
def hash_data(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOADER = DocumentLoader()

# Input schemas
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict]] = None
    session_id: Optional[str] = None

class DataAnalysisRequest(BaseModel):
    csv_base64: str
    csv_filename: str
    session_id: Optional[str] = None

class MultiUploadQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict]] = None
    image_base64: Optional[str] = None
    image_type: Optional[str] = None
    csv_base64: Optional[str] = None
    csv_filename: Optional[str] = None
    pdf_base64: Optional[str] = None
    pdf_filename: Optional[str] = None

class SQLGenerationRequest(BaseModel):
    description: str
    db_schema: Optional[str] = None
    db_type: Optional[str] = "PostgreSQL"
    query_type: Optional[str] = None
    session_id: Optional[str] = None

class DataAugmentationRequest(BaseModel):
    csv_base64: str
    csv_filename: str
    session_id: Optional[str] = None
    apply_imputation: bool = True
    apply_outlier_treatment: bool = True
    apply_synthetic_rows: bool = False  # Off by default — user must opt in
    apply_deduplication: bool = True
    apply_transformations: bool = False  # Log/Box-Cox — opt in

def update_memory_and_history(memory, chat_history, session_id: str):
    session_key = session_id or "default"
    memory.messages.clear()

    # Initialize or fetch session history
    existing_history = chat_store.get(session_key, [])
    updated_history = []

    for msg in chat_history or []:
        # Update memory (langchain) messages
        if msg["type"] == "human":
            memory.add_message(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            memory.add_message(AIMessage(content=msg["content"]))

        # Track file info if provided
        entry = {
            "type": msg["type"],
            "content": msg["content"],
        }
        if "file" in msg:
            entry["file"] = msg["file"]
        updated_history.append(entry)

    # Persist updated chat history
    chat_store[session_key] = existing_history + updated_history
    cache["chat_store"] = chat_store

    # Langchain-style formatted string
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in memory.messages])
    return chat_history_str

# Initialize retrieval chain once
rag_chain = build_chain()

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    memory = get_memory(request.session_id or "default")
    session_key = request.session_id or "default"
    
    # Update memory + store human message
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)
    
    # Invoke model
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str
    })
    
    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": response.get("answer", "No response")
    })
    cache["chat_store"] = chat_store
    return {"response": response.get("answer", "No response")}

def get_image_context(image_base64: str, image_type: str) -> str:
    """Get image context with caching"""
    image_key = hash_data(image_base64)
    if image_key in cache:
        return cache[image_key]
    
    client = Groq()
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image for data analysis:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_type};base64,{image_base64}",
                    },
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    image_context = chat_completion.choices[0].message.content
    cache[image_key] = image_context
    return image_context

def get_csv_context(csv_base64: str, question: str = "") -> str:
    """Get CSV context with caching"""
    csv_key = hash_data(csv_base64 + question)
    if csv_key in cache:
        return cache[csv_key]
    
    csv_bytes = base64.b64decode(csv_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
        tmp_csv.write(csv_bytes)
        tmp_csv_path = tmp_csv.name
    
    try:
        csv_docs = LOADER.load_csv(tmp_csv_path)
        csv_context = "\n".join([doc.page_content for doc in csv_docs])
        cache[csv_key] = csv_context
        return csv_context
    finally:
        os.unlink(tmp_csv_path)

def get_pdf_context(pdf_base64: str, question: str = "") -> str:
    """Get PDF context with caching"""
    pdf_key = hash_data(pdf_base64 + question)
    if pdf_key in cache:
        return cache[pdf_key]
    
    pdf_bytes = base64.b64decode(pdf_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        tmp_pdf_path = tmp_pdf.name
    
    try:
        pdf_docs = LOADER.load_pdf(tmp_pdf_path)
        pdf_context = "\n".join([doc.page_content for doc in pdf_docs])
        cache[pdf_key] = pdf_context
        return pdf_context
    finally:
        os.unlink(tmp_pdf_path)

@app.post("/multi-upload")
def multi_upload_endpoint(request: MultiUploadQueryRequest):
    """
    Accepts any combination of image, CSV, and PDF, and provides a context-aware answer.
    """
    session_key = request.session_id or "default"
    memory = get_memory(session_key)
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)

    # Gather contexts
    contexts = []

    # Image context
    if request.image_base64 and request.image_type:
        image_context = get_image_context(request.image_base64, request.image_type)
        contexts.append(f"Image context: {image_context}")

    # CSV context
    if request.csv_base64 and request.csv_filename:
        csv_context = get_csv_context(request.csv_base64, request.question or "")
        contexts.append(f"CSV context: {csv_context}")

    # PDF context
    if request.pdf_base64 and request.pdf_filename:
        pdf_context = get_pdf_context(request.pdf_base64, request.question or "")
        contexts.append(f"PDF context: {pdf_context}")

    # Combine all contexts
    combined_context = "\n\n".join(contexts) if contexts else None

    # Use the contextual chain if any context is present, else fallback to rag_chain
    if combined_context:
        contextual_chain = build_contextual_chain()
        response = contextual_chain.invoke({
            "input": request.question,
            "chat_history": chat_history_str,
            "context": combined_context
        })
        answer = response.content if hasattr(response, "content") else str(response)
    else:
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": chat_history_str
        })
        answer = response.get("answer", "No response")

    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": answer
    })
    cache["chat_store"] = chat_store

    return {"response": answer}

@app.get("/recent-chats/{session_id}")
def get_recent_chats(session_id: str):
    return {"chat_history": chat_store.get(session_id, [])}

@app.get("/recent-chat-titles")
def get_recent_chat_titles():
    titles = []
    for session_id, history in chat_store.items():
        for msg in history:
            if msg["type"] == "human":
                titles.append({
                    "session_id": session_id,
                    "title": msg["content"]
                })
                break  # Only take the first human message
    return {"sessions": titles}

@app.post("/save-chat")
def save_chat_endpoint(data: dict):
    session_id = data.get("session_id")
    chat_history = data.get("chat_history", [])
    if session_id and chat_history:
        chat_store[session_id] = chat_history
        cache["chat_store"] = chat_store
        return {"success": True}
    return {"success": False, "error": "Missing session_id or chat_history"}

# Plotly boolean properties for fixing serialization issues
PLOTLY_BOOL_PROPS = {
    "showarrow", "automargin", "showlegend", "matches", "visible", "autosize",
    "showticklabels", "showgrid", "zeroline", "showline", "mirror", "ticks",
    "showspikes", "showaxeslabels", "fixedrange", "constraintoward",
    "connectgaps", "fill", "showscale", "reversescale", "autocolorscale",
    "showcolorbar", "transpose", "zauto", "ncontours", "autocontour",
    "autobinx", "autobiny", "standoff", "clicktoshow", "captureevents",
    "autorange", "outlinewidth", "borderwidth", "thickness", "len",
    "fillcolor", "opacity"
}

def fix_plotly_bools(obj, parent_key=None):
    """Recursively fix boolean properties in Plotly figure dictionaries"""
    if isinstance(obj, dict):
        fixed_dict = {}
        for key, value in obj.items():
            if key in PLOTLY_BOOL_PROPS:
                if isinstance(value, int):
                    fixed_dict[key] = bool(value)
                elif isinstance(value, str):
                    if value.lower() in ['true', '1']:
                        fixed_dict[key] = True
                    elif value.lower() in ['false', '0']:
                        fixed_dict[key] = False
                    else:
                        fixed_dict[key] = value
                else:
                    fixed_dict[key] = value
            else:
                fixed_dict[key] = fix_plotly_bools(value, key)
        return fixed_dict
    elif isinstance(obj, list):
        return [fix_plotly_bools(item, parent_key) for item in obj]
    else:
        if parent_key in PLOTLY_BOOL_PROPS and isinstance(obj, int):
            return bool(obj)
        return obj

def clean_dict_for_json(obj):
    """Clean dictionary for JSON serialization with enhanced boolean handling"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            cleaned_v = clean_dict_for_json(v)
            if k in PLOTLY_BOOL_PROPS and isinstance(cleaned_v, int):
                cleaned[k] = bool(cleaned_v)
            else:
                cleaned[k] = cleaned_v
        return cleaned
    elif isinstance(obj, list):
        return [clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return clean_dict_for_json(obj.tolist())
    elif isinstance(obj, (int, float)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return obj
    else:
        return obj

def convert_to_serializable(obj):
    """Recursively convert object to JSON-serializable types"""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict('records'))
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    else:
        return obj

def clean_for_json(dataframe):
    """Clean DataFrame to ensure JSON serialization compatibility"""
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(
        lambda col: col.fillna(col.median()) if col.dtype in [np.float64, np.int64] else col
    )
    return dataframe

@app.post("/analyze-data")
def analyze_data_endpoint(request: DataAnalysisRequest):
    """
    Comprehensive data analysis endpoint that performs:
    - Data cleaning and preprocessing
    - Statistical analysis
    - AI-powered insights generation
    - Interactive visualizations
    """
    try:
        # Decode the CSV data
        csv_bytes = base64.b64decode(request.csv_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(csv_bytes)
            tmp_csv_path = tmp_csv.name
        
        try:
            # Load the CSV into a pandas DataFrame
            df = pd.read_csv(tmp_csv_path)
            
            # Initialize the DataAnalyzer
            analyzer = DataAnalyzer()
            
            # Perform deep data cleaning
            cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
            cleaned_df = clean_for_json(cleaned_df)
            
            # Generate AI-powered insights
            insights = analyzer.generate_insights(cleaned_df, cleaning_log)
            
            # Perform statistical analysis
            statistical_summary = analyzer.statistical_analysis(cleaned_df)
            
            # Clean statistical summary
            if isinstance(statistical_summary, dict):
                statistical_summary = {
                    k: (v if isinstance(v, (int, float)) and pd.notna(v) and np.isfinite(v) else None) 
                    for k, v in statistical_summary.items()
                }
            
            # Create visualizations
            plots = analyzer.create_visualizations(cleaned_df)
            
            # Convert plots to JSON for frontend
            plots_json = {}
            for plot_name, fig in plots.items():
                try:
                    fig_dict = fig.to_dict()
                    cleaned_fig_dict = clean_dict_for_json(fig_dict)
                    fixed_fig_dict = fix_plotly_bools(cleaned_fig_dict)
                    plots_json[plot_name] = fixed_fig_dict
                except Exception as plot_error:
                    plots_json[plot_name] = {"error": f"Could not generate plot: {str(plot_error)}"}

            # Prepare response data
            response_data = {
                "success": True,
                "original_shape": df.shape,
                "cleaned_shape": cleaned_df.shape,
                "cleaning_log": convert_to_serializable(cleaning_log),
                "insights": convert_to_serializable(insights),
                "statistical_summary": convert_to_serializable(statistical_summary),
                "visualizations": plots_json,
                "column_info": convert_to_serializable({
                    "original_columns": list(df.columns),
                    "cleaned_columns": list(cleaned_df.columns),
                    "data_types": cleaned_df.dtypes.astype(str).to_dict(),
                    "missing_values": cleaned_df.isnull().sum().to_dict(),
                    "unique_counts": cleaned_df.nunique().to_dict()
                }),
                "sample_data": {
                    "original": convert_to_serializable(df.head()),
                    "cleaned": convert_to_serializable(cleaned_df.head())
                }
            }
            
            return JSONResponse(content=response_data)
            
        finally:
            os.unlink(tmp_csv_path)
            
    except Exception as e:
        error_message = str(e)
        
        # Handle specific error types
        if "API quota" in error_message or "429" in error_message:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "API quota exceeded",
                    "message": "The AI insights feature has reached its usage limit. Please try again later or check your API quota.",
                    "error_type": "quota_exceeded"
                }
            )
        elif "pandas" in error_message.lower() or "csv" in error_message.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid CSV file",
                    "message": "The uploaded file could not be processed as a valid CSV. Please check the file format.",
                    "error_type": "invalid_csv"
                }
            )
        elif "JSON compliant" in error_message or "out of range" in error_message.lower():
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Data serialization error",
                    "message": "The data contains values that cannot be converted to JSON. This usually indicates infinite or extremely large numbers in your dataset.",
                    "error_type": "serialization_error"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Analysis failed",
                    "message": f"An error occurred during data analysis: {error_message}",
                    "error_type": "analysis_error"
                }
            )

@app.post("/clean-data")
def clean_data_endpoint(request: DataAnalysisRequest):
    """
    Simplified data cleaning endpoint that only performs data preprocessing
    without AI insights (useful when API quota is exceeded)
    """
    try:
        csv_bytes = base64.b64decode(request.csv_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(csv_bytes)
            tmp_csv_path = tmp_csv.name
        
        try:
            df = pd.read_csv(tmp_csv_path)
            analyzer = DataAnalyzer()
            
            # Perform deep data cleaning
            cleaned_df, cleaning_log = analyzer.deep_clean_data(df)
            
            # Perform statistical analysis
            statistical_summary = analyzer.statistical_analysis(cleaned_df)
            
            # Create basic visualizations
            plots = analyzer.create_visualizations(cleaned_df)
            
            # Convert plots to JSON for frontend
            plots_json = {}
            for plot_name, fig in plots.items():
                try:
                    fig_dict = fig.to_dict()
                    cleaned_fig_dict = fix_plotly_bools(fig_dict)
                    plots_json[plot_name] = cleaned_fig_dict
                except Exception as plot_error:
                    plots_json[plot_name] = {"error": f"Could not generate plot: {str(plot_error)}"}            
            
            response_data = {
                "success": True,
                "original_shape": df.shape,
                "cleaned_shape": cleaned_df.shape,
                "cleaning_log": cleaning_log,
                "statistical_summary": statistical_summary,
                "visualizations": plots_json,
                "column_info": {
                    "original_columns": list(df.columns),
                    "cleaned_columns": list(cleaned_df.columns),
                    "data_types": cleaned_df.dtypes.astype(str).to_dict(),
                    "missing_values": cleaned_df.isnull().sum().to_dict(),
                    "unique_counts": cleaned_df.nunique().to_dict()
                },
                "sample_data": {
                    "original": df.head().to_dict('records'),
                    "cleaned": cleaned_df.head().to_dict('records')
                }
            }
            
            return JSONResponse(content=response_data)
            
        finally:
            os.unlink(tmp_csv_path)
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Data cleaning failed",
                "message": f"An error occurred during data cleaning: {str(e)}",
                "error_type": "cleaning_error"
            }
        )

@app.post("/generate-sql")
def generate_sql_endpoint(request: SQLGenerationRequest):
    """
    SQL query generation endpoint that:
    - Accepts a natural language description
    - Optionally uses provided schema or auto-detects from context
    - Returns a generated SQL query, explanation, and optimization suggestions
    """
    try:
        # Build a structured prompt for the SQL generation
        schema_section = f"\n\nDatabase Schema:\n{request.db_schema}" if request.db_schema else ""
        query_type_section = f"\nQuery Type: {request.query_type}" if request.query_type else ""

        prompt = f"""You are an expert SQL developer. Generate a SQL query based on the following:

Database Type: {request.db_type}{query_type_section}{schema_section}

User Request: {request.description}

Respond ONLY in the following JSON format (no markdown, no extra text):
{{
  "sql_query": "<the SQL query>",
  "explanation": "<plain English explanation of what the query does and why>",
  "suggestions": "<performance tips, index recommendations, or alternative approaches>"
}}"""

        client = Groq()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert SQL developer. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1  # Low temperature for deterministic SQL output
        )

        raw_response = chat_completion.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw_response.startswith("```"):
            raw_response = raw_response.split("```")[1]
            if raw_response.startswith("json"):
                raw_response = raw_response[4:]
            raw_response = raw_response.strip()

        # Parse the JSON response
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback: return raw output as the query if JSON parsing fails
            result = {
                "sql_query": raw_response,
                "explanation": "Could not parse structured response. Raw output returned.",
                "suggestions": ""
            }

        # Optionally store in session history
        if request.session_id:
            session_key = request.session_id
            chat_store.setdefault(session_key, [])
            chat_store[session_key].append({
                "type": "human",
                "content": f"[SQL Generator] {request.description}"
            })
            chat_store[session_key].append({
                "type": "ai",
                "content": result.get("sql_query", "")
            })
            cache["chat_store"] = chat_store

        return JSONResponse(content={
            "success": True,
            "sql_query": result.get("sql_query", ""),
            "explanation": result.get("explanation", ""),
            "suggestions": result.get("suggestions", ""),
            "db_type": request.db_type,
            "query_type": request.query_type
        })

    except Exception as e:
        error_message = str(e)

        if "API quota" in error_message or "429" in error_message:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "API quota exceeded",
                    "message": "The SQL generation feature has reached its usage limit. Please try again later.",
                    "error_type": "quota_exceeded"
                }
            )
        elif "json" in error_message.lower():
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Response parsing error",
                    "message": "The AI returned an unexpected format. Please try rephrasing your request.",
                    "error_type": "parse_error"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "SQL generation failed",
                    "message": f"An error occurred during SQL generation: {error_message}",
                    "error_type": "generation_error"
                }
            )

@app.post("/diagnose-data")
def diagnose_data_endpoint(request: DataAugmentationRequest):
    """
    Stage 1: Scan the CSV and return an augmentation plan without modifying data.
    """
    try:
        csv_bytes = base64.b64decode(request.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name

        try:
            df = pd.read_csv(tmp_path)
            augmentor = DataAugmentor()
            diagnosis = augmentor.diagnose(df)
            return JSONResponse(content={"success": True, "diagnosis": convert_to_serializable(diagnosis)})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Diagnosis failed",
            "message": str(e)
        })


@app.post("/augment-data")
def augment_data_endpoint(request: DataAugmentationRequest):
    """
    Stage 2 & 3: Apply augmentation based on user-selected options and return
    augmented CSV + change log.
    """
    try:
        csv_bytes = base64.b64decode(request.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name

        try:
            df = pd.read_csv(tmp_path)
            augmentor = DataAugmentor()

            options = {
                "apply_imputation": request.apply_imputation,
                "apply_outlier_treatment": request.apply_outlier_treatment,
                "apply_synthetic_rows": request.apply_synthetic_rows,
                "apply_deduplication": request.apply_deduplication,
                "apply_transformations": request.apply_transformations,
            }

            augmented_df, change_log = augmentor.augment(df, options)

            # Encode augmented CSV back to base64
            csv_buffer = augmented_df.to_csv(index=False)
            augmented_b64 = base64.b64encode(csv_buffer.encode("utf-8")).decode("utf-8")

            return JSONResponse(content={
                "success": True,
                "original_shape": list(df.shape),
                "augmented_shape": list(augmented_df.shape),
                "change_log": convert_to_serializable(change_log),
                "augmented_csv_base64": augmented_b64,
                "augmented_filename": f"augmented_{request.csv_filename}",
                "sample_original": convert_to_serializable(df.head()),
                "sample_augmented": convert_to_serializable(augmented_df.head())
            })
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Augmentation failed",
            "message": str(e),
            "error_type": "augmentation_error"
        })