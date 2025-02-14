from semantic_router.llms.ollama import OllamaLLM
from semantic_router import Route, RouteLayer

from langchain_ollama import ChatOllama
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker

from sentence_transformers import SentenceTransformer
import faiss

from graph.custom_encoder import STEncoder
from graph.vectorStore import SQLiteVectorStore, CustomCrossEncoder
from graph.config import Config
from graph.state import GraphState
from graph.utils import format_documents

from typing import List, Literal
from operator import itemgetter


SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question.
If the answer is not found within the context, state that the answer cannot be found.
Use a list where applicable and keep the response to a maximum length of 3 paragraphs.
The contextual information is organized with the most relevant source appearing first.
Each source is separated by a horizontal rule (---).

Context:
{context}

Use markdown formatting where appropriate.
"""


class Router():
  def __init__(self, routes : List[Route], k = 5):
    self.embedding_model = STEncoder(name=Config.Model.EMBEDDINGS)
    self.route_llm = OllamaLLM(llm_name=Config.Model.ROUTER, temperature=Config.Model.ROUTER_TEMP, max_tokens=Config.Model.MAX_TOKENS)
    self.route_layer = RouteLayer(encoder=self.embedding_model, llm=self.route_llm, routes=routes, top_k=k)

  def find_route(self, state: GraphState) -> GraphState:
    route =  self.route_layer(str(state['input_querry']))

    if route is None:
      state["next_node"] = None
      state["function_call"] = None
    else:
      state["next_node"] = route.name
      state['function_call'] = route.function_call
    
    return state
  

class RagChain():
  def __init__(self):
    self.model = ChatOllama(
      model=Config.Model.GEN_LLM,
      temperature=Config.Model.GEN_TEMP,
      max_tokens=Config.Model.MAX_TOKENS
    )

    self.retriever = ContextualCompressionRetriever(
      base_compressor=CrossEncoderReranker(model=CustomCrossEncoder(Config.Model.RERANKER), top_n=4),
      base_retriever=SQLiteVectorStore(
        SentenceTransformer(Config.Model.EMBEDDINGS),
        faiss.read_index(str(Config.Data.FAISS_INDEX)),
        str(Config.Data.SQL_DB),
      ).as_retriever()
    )

    self.prompt = SYSTEM_PROMPT
    self.chain = self.__create_chain__()

  def __create_chain__(self):
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
      ]
    )

    chain = (
      RunnablePassthrough.assign(
        context=itemgetter("question")
        | self.retriever.with_config({"run_name": "context_retriever"})
        | format_documents
      )
      | prompt
      | self.model
    )

    return chain

  def ask_question(self, data: dict) -> str:
    return self.chain.invoke(data)