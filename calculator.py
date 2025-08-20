from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import Ollama

# === Base LLaMA ===
llm = Ollama(
    model="gemma3",
    temperature=0.7,
    num_predict=200
)

# === Math Tool ===
llm_math_chain = LLMMathChain.from_llm(llm=llm)

def calculator_fn(query: str) -> str:
    try:
        result = llm_math_chain.run(query)
        return result
    except Exception as e:
        return f"❌ Failed to evaluate expression: {query}\nError: {str(e)}"