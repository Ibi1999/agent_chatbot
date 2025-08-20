# rag_github.py  — RAG using GitHub Models via OpenAI-compatible endpoint
import os
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === GitHub Models (OpenAI-compatible) ===
ENDPOINT = "https://models.inference.ai.azure.com"   # GitHub Models endpoint
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set. Define it in your environment.")

CHAT_MODEL = os.getenv("GHM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("GHM_EMBED", "text-embedding-3-small")

# === Paths ===
DOCS_DIR = "documents"
PERSIST_DIR = "rag_chroma_db"  # delete & rebuild if switching from any other embedding family

# === Clean up raw text from PDFs and CSVs ===
def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenated line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Merge lines within paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize multiple blank lines
    text = re.sub(r'(?m)^[ \t]*\d+[ \t]*$', '', text)  # Remove page numbers
    return text.strip()

# === Load & clean all PDFs and CSVs ===
def load_and_clean_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.page_content = clean_text(doc.page_content)
                documents.append(doc)

        elif filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
            text = df.to_csv(index=False)
            doc = Document(page_content=clean_text(text), metadata={"source": file_path})
            documents.append(doc)

    return documents

# === Split into chunks ===
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# === Build and save vectorstore using Chroma (GitHub Models embeddings) ===
def create_vector_store():
    docs = load_and_clean_documents(DOCS_DIR)
    chunks = split_documents(docs)

    # Use OpenAIEmbeddings but point it at GitHub Models via base_url + api_key
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=GITHUB_TOKEN,
        base_url=ENDPOINT,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"[✅] Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print(f"[💾] Chroma DB saved to: {PERSIST_DIR}")

# === RAG Retrieval and QA ===
def get_rag_response(query: str, chat_history=None):
    if not hasattr(get_rag_response, "vectordb"):
        # Cache the vector DB / retriever / LLM on first call
        get_rag_response.embedding_model = OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=ENDPOINT,
        )
        get_rag_response.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_rag_response.embedding_model
        )
        get_rag_response.retriever = get_rag_response.vectordb.as_retriever()

        # Chat LLM via GitHub Models (OpenAI-compatible)
        get_rag_response.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.2,
            api_key=GITHUB_TOKEN,
            base_url=ENDPOINT,
            # timeout=60, max_retries=2,  # optional hardening
        )

    retriever = get_rag_response.retriever
    llm = get_rag_response.llm

    # Format chat history if provided
    history_str = ""
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    custom_prompt = (
        "You are a helpful assistant trained on Ibrahim's work and project documents and anything related to him.\n"
        "If the user asks a vague question, kindly guide them to ask about him or his work work in data Analysis, based on his projects.\n\n"
        f"User query: {query}"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": custom_prompt})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    if not source_docs or "you didn't ask a question" in answer.lower():
        fallback = (
            "Hi there! I'm here to help answer questions based on Ibrahim's work and project documents."
            "Try asking me about his football clustering models, prediction systems, or any other data science work!"
        )
        return fallback, []

    sources = []
    for doc in source_docs:
        meta = doc.metadata
        filename = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page", None)
        if page is not None:
            sources.append(f"{filename}, page {page + 1}")
        else:
            sources.append(filename)

    return answer, list(set(sources))

# === Run this once to build the DB ===
if __name__ == "__main__":
    # IMPORTANT: if you previously built with Ollama or OpenAI embeddings,
    # delete the 'rag_chroma_db' folder first to avoid mixing embeddings.
    # Windows PowerShell:  Remove-Item -Recurse -Force .\rag_chroma_db
    create_vector_store()
