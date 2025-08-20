# calculator.py — LLMMathChain using GitHub Models (OpenAI-compatible), no Ollama
import os

from langchain.chains.llm_math.base import LLMMathChain
from langchain_openai import ChatOpenAI

ENDPOINT = "https://models.inference.ai.azure.com"

def _get_secret(key: str, default: str | None = None):
    """Prefer Streamlit secrets; fall back to env."""
    try:
        import streamlit as st  # only available on Streamlit Cloud / local runs with streamlit
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

GITHUB_TOKEN = _get_secret("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set (Streamlit secrets or env).")

# You can override this per deployment if you want a different model for math prompts
MATH_MODEL = _get_secret("GHM_MATH_MODEL", "gpt-4o-mini")

# Use a low temperature for deterministic math
llm = ChatOpenAI(
    model=MATH_MODEL,
    api_key=GITHUB_TOKEN,
    base_url=ENDPOINT,
    temperature=0.0,
)

# Build the math chain
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

def calculator_fn(query: str) -> str:
    try:
        result = llm_math_chain.run(query)
        return result.strip()
    except Exception as e:
        return f"❌ Failed to evaluate expression: {query}\nError: {str(e)}"
