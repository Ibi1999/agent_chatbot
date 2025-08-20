# rag_faiss.py — RAG using GitHub Models + FAISS (no sqlite)
import os
import re
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === GitHub Models (OpenAI-compatible) ===
ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set (env or Streamlit secrets).")

CHAT_MODEL = os.getenv("GHM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("GHM_EMBED", "text-embedding-3-small")

# === Paths ===
DOCS_DIR = "documents"
FAISS_DIR = "faiss_index"  # folder where FAISS index is stored

def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?m)^[ \t]*\d+[ \t]*$', '', text)
    return text.strip()

def load_and_clean_documents(folder_path: str):
    documents = []
    if not os.path.isdir(folder_path):
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            for doc in loader.load():
                doc.page_content = clean_text(doc.page_content)
                documents.append(doc)

        elif filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
            text = df.to_csv(index=False)
            documents.append(Document(page_content=clean_text(text), metadata={"source": file_path}))
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# === Build & save FAISS index ===
def create_vector_store():
    docs = load_and_clean_documents(DOCS_DIR)
    chunks = split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=GITHUB_TOKEN, base_url=ENDPOINT)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"[✅] Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print(f"[💾] FAISS index saved to: {FAISS_DIR}")

# === RAG Retrieval and QA ===
def get_rag_response(query: str, chat_history=None):
    # lazy-load FAISS index
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=GITHUB_TOKEN, base_url=ENDPOINT)
    if not os.path.isdir(FAISS_DIR):
        # if index not present (first deploy), build it on the fly
        create_vector_store()

    vectorstore = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=GITHUB_TOKEN, base_url=ENDPOINT)

    # (optional) include past turns as context text
    history_str = ""
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    custom_prompt = (
        "You are a helpful assistant trained on Ibrahim's work and project documents and anything related to him.\n"
        "If the user asks a vague question, kindly guide them to ask about him or his work in data analysis, based on his projects.\n\n"
        f"{history_str}"
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
            "Hi there! I'm here to help answer questions based on Ibrahim's work and project documents. "
            "Try asking about his football clustering models, prediction systems, or other data science work!"
        )
        return fallback, []

    sources = []
    for doc in source_docs:
        meta = doc.metadata
        filename = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page")
        sources.append(f"{filename}, page {page + 1}" if page is not None else filename)

    return answer, sorted(set(sources))

if __name__ == "__main__":
    create_vector_store()
