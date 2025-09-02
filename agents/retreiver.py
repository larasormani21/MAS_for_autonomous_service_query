import os
from pathlib import Path
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import SERVICE_FOLDER, INDEX_PATH, EMBEDDING_MODEL, RETRIEVER_MODEL
from state import State

class RetrieverAgent:
    """
    RetrieverAgent queries the FAISS index and selects relevant documents for a given user query.
    Also checks if reindexing is needed.
    """

    def __init__(self, services_dir=SERVICE_FOLDER, index_path=INDEX_PATH, embedding_model=EMBEDDING_MODEL, llm_model=RETRIEVER_MODEL):
        self.services_dir = Path(services_dir)
        self.index_path = Path(index_path)
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=llm_model)

    def _get_current_files_set(self):
        return {
            f.name
            for f in self.services_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".html", ".json", ".yaml", ".yml"]
        }

    def _get_index_files_set(self):
        if not self.index_path.exists():
            return set()
        try:
            vs = FAISS.load_local(str(self.index_path), self.embedding, allow_dangerous_deserialization=True)
            all_docs = list(vs.docstore._dict.values())
            return {os.path.basename(doc.metadata.get("source", "")) for doc in all_docs}
        except Exception:
            return set()

    def _needs_reindex(self) -> bool:
        current_files = self._get_current_files_set()
        index_files = self._get_index_files_set()
        return current_files != index_files

    def _load_vectorstore(self):
        if not self.index_path.exists():
            raise FileNotFoundError("Index path not found. Run Indexer first.")
        return FAISS.load_local(
            str(self.index_path),
            self.embedding,
            allow_dangerous_deserialization=True
        )

    def get_relevant_files(self, query: str, retriever, llm) -> list[str]:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You're an assistant that selects the most relevant documents for a user query.
        You have access to these documents (context):

        {context}

        User query:
        {question}

        Prioritize OpenAPI files (.json, .yaml, .yml) over .html files. 
        Discard irrelevant documents.
        """,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
        )

        result = qa_chain.invoke({"query": query})
        retrieved_docs = result.get("source_documents", [])

        if not retrieved_docs:
            raise ValueError("No relevant API found in FAISS index")

        files_by_source = {}
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "")
            if source and source not in files_by_source:
                files_by_source[source] = source
        return list(files_by_source.keys())

    def run(self, state: State) -> State:
        try:
            print("Running RetrieverAgent...")

            if self._needs_reindex():
                print("Retriever detected index out-of-date. Triggering Indexer...")
                return {**state, "needs_reindex": True, "retrieved": False, "done": False}

            vectorstore = self._load_vectorstore()
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.9, "fetch_k": 20},
            )
            files = self.get_relevant_files(state["user_query"], retriever, self.llm)

            print("Relevant files retrieved:")
            for f in files:
                print(f" - {f}")

            return {
                **state,
                #"retriever": retriever,
                "candidate_files": files,
                "current_index": 0,
                "retrieved": True,
                "needs_reindex": False,
            }

        except Exception as e:
            print(f"RetrieverAgent failed: {e}")
            return {**state, "done": True, "error": str(e), "retrieved": True, "needs_reindex": False}
