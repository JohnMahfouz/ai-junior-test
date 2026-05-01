from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

FAISS_INDEX_PATH = Path(__file__).parent.parent / "faiss_index"

_vectorstore: FAISS | None = None


def _load_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not FAISS_INDEX_PATH.exists():
            raise RuntimeError(
                "FAISS index not found. Run `python -m rag.ingest` first to build the index."
            )
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    return _load_vectorstore().as_retriever(search_kwargs={"k": k})
