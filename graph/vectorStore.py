from sentence_transformers import SentenceTransformer, CrossEncoder
from faiss import IndexFlatL2

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain.vectorstores import VectorStore
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

from typing import List
import numpy as np
import sqlite3


class SQLiteVectorStore(VectorStore):
  def __init__(self, embedding_model: SentenceTransformer, faiss_db: IndexFlatL2, sql_db: str):
    self.embedding_model = embedding_model
    self.faiss_db = faiss_db
    self.sql_db = sql_db
  
  @classmethod
  def from_texts(cls, texts: List[str], embedding_model: SentenceTransformer, faiss_db: IndexFlatL2, sql_db: str, **kwargs) -> "SQLiteVectorStore":
    """Creates a new SQLiteVectorStore instance from raw texts."""
    instance = cls(embedding_model, faiss_db, sql_db)
    instance.add_texts(texts)
    return instance
    
  def add_texts(self, data: list[str]):
    conn = sqlite3.connect(self.sql_db)
    cursor = conn.cursor()
    embeddings = self.embedding_model.encode(data)
    self.faiss_db.add(np.array(embeddings))
    
    for text in data:
      cursor.execute("INSERT INTO data (text) VALUES (?)", (text,))
    conn.commit()
  
  def similarity_search(self, query: str, k=100) -> list[Document]:
    """Searches FAISS for similar vectors and retrieves corresponding text from SQLite."""
    conn = sqlite3.connect(self.sql_db)
    cursor = conn.cursor()
    embedding = self.embedding_model.encode(query)
    _, idxs = self.faiss_db.search(np.array([embedding]), k=k)

    if not idxs.size:
      return []

    placeholders = ','.join(['?'] * len(idxs.flatten()))
    cursor.execute(f"SELECT text FROM data WHERE rowid IN ({placeholders})", idxs.flatten().tolist())
    results = cursor.fetchall()

    return [Document(page_content=row[0]) for row in results]


  def as_retriever(self, search_kwargs=None) -> VectorStoreRetriever:
    """Returns a VectorStoreRetriever for use in LangChain."""
    search_kwargs = search_kwargs or {"k": 200}
    return VectorStoreRetriever(vectorstore=self, search_kwargs=search_kwargs)


class CustomCrossEncoder(BaseCrossEncoder):
  def __init__(self, name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
    """Initializes the reranker with a cross-encoder model."""
    super().__init__()  # Initialize BaseCrossEncoder
    self.model = CrossEncoder(name)  # Load the model
  
  def score(self, querry_doc_pairs: List[tuple[str, str]]) -> float:
    """Computes the relevance score for a single query-document pair."""
    return self.model.predict(querry_doc_pairs)