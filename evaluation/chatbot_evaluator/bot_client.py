import requests

API_URL = "http://localhost:8000/multi-upload"

def run_bot_chat(question: str):
    try:
        payload = {
            "question": question,     # ✅ correct field
            "session_id": "eval",     # optional but useful
            "chat_history": [],       # required for your logic
            "image_base64": None,
            "image_type": None,
            "csv_base64": None,
            "csv_filename": None,
            "pdf_base64": None,
            "pdf_filename": None
        }

        response = requests.post(API_URL, json=payload)

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        data = response.json()

        return data.get("response", "")

    except Exception as e:
        print("ERROR:", e)
        return ""