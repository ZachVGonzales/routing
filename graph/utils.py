from langchain_core.documents import Document
from typing import List



def format_documents(documents: List[Document]) -> str:
  texts = []
  for doc in documents:
    texts.append(doc.page_content)
    texts.append("---")
  
  return "\n".join(texts)