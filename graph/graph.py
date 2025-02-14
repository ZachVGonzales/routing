from langgraph.graph import StateGraph, START, END

from graph.routes import time_route, objective_route, get_time, analyze_objective
from graph.state import GraphState
from graph.models import Router, RagChain

from typing import Literal


R1_NODES = ['analyze_objective', 'get_time']


def objective_analysis(state: GraphState) -> GraphState:
  function_call = state["function_call"]
  
  for kwarg in function_call:
    objective = kwarg.get("objective")
    if objective is not None:
      break
  
  if objective is None:
    return state
  
  state["objective_analysis_response"].append(analyze_objective(objective))
  return state


def time_tool(state: GraphState) -> GraphState:
  function_call = state["function_call"]
  for kwarg in function_call:
    timezone = kwarg.get("timezone")
    if timezone is not None:
      break
  
  if timezone is None:
    return state
  
  state["time_response"].append(get_time(timezone))
  return state


# TODO: implement chat history functionality
def rag_qa(state: GraphState, rag_chain: RagChain) -> GraphState:
  question = state["input_querry"]
  input_data = {"question" : question, "chat_history": []}
  state["rag_response"].append(rag_chain.ask_question(input_data))
  return state


def parse_r1(state: GraphState) -> Literal['analyze_objective', 'get_time', 'default']:
  next_node = state["next_node"]

  if (next_node is None) or not (next_node in R1_NODES):
    return "default"
  return next_node


def create_graph() -> StateGraph:
  graph = StateGraph(GraphState)
  router = Router([time_route, objective_route])
  rag_chain = RagChain()
  
  graph.add_node("router", lambda state: router.find_route(state))
  graph.add_node("analyze_objective", lambda state: objective_analysis(state))
  graph.add_node("get_time", lambda state: time_tool(state))
  graph.add_node("default", lambda state: rag_qa(state, rag_chain))
  
  graph.add_edge(START, "router")
  graph.add_edge("analyze_objective", END)
  graph.add_edge("get_time", END)
  graph.add_edge("default", END)

  graph.add_conditional_edges(
    "router",
    lambda state: parse_r1(state=state)
  )

  return graph

