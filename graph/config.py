from pathlib import Path

"""
Config for AI Agent graph
"""

APP_HOME = Path(__file__).parent.parent

class Config:

  class Data:
    DATABASE = APP_HOME / "database"
    DOCUMENTS = APP_HOME / "documents"
    FAISS_INDEX = DATABASE / "sql_faiss" / "context_index.bin"
    SQL_DB = DATABASE / "sql_faiss" / "data.db"
  
  class Model:
    EMBEDDINGS = "multi-qa-mpnet-base-dot-v1"
    RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ROUTER = "mistral:latest"
    GEN_LLM = "llama3.2:3b"
    OBJ_ANALYSIS = APP_HOME / "models" / "objective_analysis"

    ROUTER_TEMP = 0.0
    GEN_TEMP = 0.0
    
    MAX_TOKENS = 2048