from fastapi import APIRouter, HTTPException
from chains.rag_chain import build_chain
from dotenv import load_dotenv
load_dotenv()
router = APIRouter()
chain = build_chain()

@router.post("/chat")
async def chat_with_bot(payload: dict):
    try:
        query = payload.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' field")

        result = chain.invoke({"input": query})
        return {
            "query": query,
            "response": result.get("answer", "No response generated.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
