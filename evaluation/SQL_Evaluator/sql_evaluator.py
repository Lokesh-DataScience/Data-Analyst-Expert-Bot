import sqlite3
import json
import requests
from difflib import SequenceMatcher

API_URL = "http://localhost:8000/generate-sql"


# -------------------------
# DB EXECUTION
# -------------------------
def execute_sql(db_path, query):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows], None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


# -------------------------
# BOT CALL
# -------------------------
def generate_sql(question, schema):
    payload = {
        "description": question,        
        "db_type": "PostgreSQL",        
        "db_schema": schema,            
        "query_type": "SELECT",        
        "session_id": "eval"            
    }

    try:
        res = requests.post(API_URL, json=payload)

        print("\nSTATUS:", res.status_code)
        print("RAW RESPONSE:", res.text)

        data = res.json()

        if not data.get("success"):
            print("❌ API ERROR:", data)
            return ""

        return data.get("sql_query", "")

    except Exception as e:
        print("❌ REQUEST FAILED:", e)
        return ""


# -------------------------
# METRICS
# -------------------------
def exact_match(pred, gt):
    return 1 if pred.strip().lower() == gt.strip().lower() else 0


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def result_match(pred, gt):
    return 1 if pred == gt else 0


# -------------------------
# EVALUATION LOOP
# -------------------------
def evaluate(dataset_path, db_path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    metrics = {
        "total": 0,
        "sql_validity": 0,
        "execution_accuracy": 0,
        "exact_match": 0,
        "avg_similarity": 0
    }

    for sample in dataset:
        question = sample["question"]
        schema = sample["schema"]
        gt_sql = sample["ground_truth_sql"]
        expected_result = sample["expected_result"]

        pred_sql = generate_sql(question, schema)

        print("\nQ:", question)
        print("Generated SQL:", pred_sql)

        # SQL validity
        pred_result, error = execute_sql(db_path, pred_sql)

        if error is None:
            metrics["sql_validity"] += 1

        # Execution accuracy
        gt_result, _ = execute_sql(db_path, gt_sql)

        if pred_result == gt_result:
            metrics["execution_accuracy"] += 1

        # Exact match
        metrics["exact_match"] += exact_match(pred_sql, gt_sql)

        # Similarity
        metrics["avg_similarity"] += similarity(pred_sql, gt_sql)

        metrics["total"] += 1

    # Normalize
    total = metrics["total"]

    metrics["sql_validity"] /= total
    metrics["execution_accuracy"] /= total
    metrics["exact_match"] /= total
    metrics["avg_similarity"] /= total

    return metrics

results = evaluate("sql_eval_dataset.json", "test.db")

print("\n📊 SQL Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.2f}")