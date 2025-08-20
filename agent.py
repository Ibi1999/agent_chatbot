# agent2.py — Router using GitHub Models (no Ollama)
import os
from typing import List, Dict

from calculator import calculator_fn
from rag_model import get_rag_response

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# === GitHub Models (OpenAI-compatible) ===
ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set. Define it in your environment.")

# Chat models
ROUTER_MODEL = os.getenv("GHM_ROUTER_MODEL", "gpt-4o-mini")   # for classification
GENERAL_MODEL = os.getenv("GHM_MODEL", "gpt-4o-mini")         # for general answers


def _llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Construct a ChatOpenAI client pointed at GitHub Models."""
    return ChatOpenAI(
        model=model,
        api_key=GITHUB_TOKEN,
        base_url=ENDPOINT,
        temperature=temperature,
    )


def _to_lc_messages(conversation: List[Dict[str, str]]):
    """Convert [{'role': 'user'|'assistant'|'system', 'content': str}, ...]
    into LangChain message objects."""
    lc_msgs = []
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs


def _route_decision(user_query: str, chat_history=None) -> str:
    """Return 'RAG', 'Calculator', or 'General' using a small routing prompt."""
    system_prompt = (
        "You are an expert assistant that decides how to answer user queries. "
        "If the question is about Ibrahim, his work, his projects, his background, or his hobbies, "
        "respond with 'RAG'. "
        "If the question is general, unrelated to Ibrahim, respond with 'General'. "
        "If the question is a mathematical calculation (like arithmetic, multiplication, division, "
        "percentages, NOT POPULATION etc.), respond with 'Calculator'. "
        "Do not explain your choice. Respond with only 'RAG', 'General', or 'Calculator'."
    )

    convo: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if chat_history:
        convo.extend(chat_history)
    convo.append({"role": "user", "content": user_query})

    messages = _to_lc_messages(convo)
    router_llm = _llm(ROUTER_MODEL, temperature=0.0)
    decision = router_llm.invoke(messages).content.strip().upper()
    # be lenient with prefixes like "RAG\n" or "RAG - use retrieval"
    if decision.startswith("RAG"):
        return "RAG"
    if decision.startswith("CALCULATOR"):
        return "Calculator"
    return "General"


def agent_router(user_query: str, chat_history=None):
    decision = _route_decision(user_query, chat_history)

    if decision == "RAG":
        answer, sources = get_rag_response(user_query, chat_history)
        return {
            "answer": answer,
            "sources": sources,
            "type": "RAG",
        }

    if decision == "Calculator":
        answer = calculator_fn(user_query)
        return {
            "answer": answer,
            "sources": [],
            "type": "Calculator",
        }

    # General answer via GitHub Models
    base_system = (
        "You are a concise, helpful assistant. "
        "Answer clearly and directly. If the user asks for code, provide runnable examples."
    )
    convo: List[Dict[str, str]] = [{"role": "system", "content": base_system}]
    if chat_history:
        convo.extend(chat_history)
    convo.append({"role": "user", "content": user_query})

    messages = _to_lc_messages(convo)
    general_llm = _llm(GENERAL_MODEL, temperature=0.2)
    answer = general_llm.invoke(messages).content

    return {
        "answer": answer,
        "sources": [],
        "type": "General",
    }
