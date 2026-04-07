from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import json
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"   # free + fast

def judge_chatbot(question, response, reference=None):
    prompt = f"""
You are an expert evaluator for a data analyst chatbot.

Evaluate the response on:
1. Relevance
2. Correctness
3. Insightfulness
4. Clarity
5. Groundedness

Question: {question}
Response: {response}
Reference: {reference}

Return ONLY JSON:
{{
    "relevance": int,
    "correctness": int,
    "insightfulness": int,
    "clarity": int,
    "groundedness": int
}}
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = completion.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        # fallback (very important in production)
        return {
            "relevance": 0,
            "correctness": 0,
            "insightfulness": 0,
            "clarity": 0,
            "groundedness": 0
        }