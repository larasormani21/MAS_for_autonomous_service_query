import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default: str):
    val = os.getenv(key, "").strip()
    return val if val else default

EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "nomic-embed-text")
SERVICE_FOLDER = get_env("SERVICE_FOLDER", "services_descriptions")
INDEX_PATH = get_env("INDEX_PATH", "faiss_index")
RETRIEVER_MODEL = get_env("RETRIEVER_MODEL", "llama3")
CONVERTER_MODEL = get_env("CONVERTER_MODEL", "mistral")
EXECUTOR_MODEL = get_env("EXECUTOR_MODEL", "mistral")
FEEDBACK_MODEL = get_env("FEEDBACK_MODEL", "llama3")
EXTRACTOR_MODEL = get_env("EXTRACTOR_MODEL", "llama3")