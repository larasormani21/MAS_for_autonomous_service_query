from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import SERVICE_FOLDER, INDEX_PATH, EMBEDDING_MODEL
from data_ingestor import DataIngestor
from state import State


class Indexer:
    """
    LangGraph node that builds/updates the FAISS index when requested.
    """

    def __init__(self, services_dir=SERVICE_FOLDER, index_path=INDEX_PATH, embedding_model=EMBEDDING_MODEL):
        self.services_dir = Path(services_dir)
        self.index_path = Path(index_path)
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = None

    def _create_index(self):
        """Build a new FAISS index from the documents in the service directory."""
        ingestor = DataIngestor()
        chunks = []

        for file in self.services_dir.iterdir():
            if not file.is_file():
                continue
            try:
                ext = file.suffix.lower()
                if ext == ".html":
                    docs = ingestor.load_html(str(file))
                elif ext in [".yaml", ".yml"]:
                    docs = ingestor.load_openapi_yaml(str(file))
                elif ext == ".json":
                    docs = ingestor.load_openapi_json(str(file))
                else:
                    continue
                chunks.extend(docs)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")

        if not chunks:
            print("No documents ingested during indexing. Index will not be created.")
            return None

        self.vectorstore = FAISS.from_documents(chunks, self.embedding)
        self.vectorstore.save_local(str(self.index_path))
        print(f"FAISS index created at {self.index_path}, total vectors: {self.vectorstore.index.ntotal}")
        return self.vectorstore

    def run(self, state: State) -> State:
        """
        LangGraph node: builds FAISS index and resets needs_reindex flag.
        """
        print("Running Indexer...\nrebuilding FAISS index")
        vectorstore = self._create_index()
        if not vectorstore:
            return {**state, "done": True, "error": "Indexing failed: no documents found"}

        return {
            **state,
            #"vectorstore": vectorstore,
            "needs_reindex": False,
            #"retrieved": False,
            #"done": False,
        }
