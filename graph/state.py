from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Define the state object for the agent graph
class GraphState(TypedDict):
  input_querry: str
  next_node: str
  function_call: list[dict]
  objective_analysis_response: Annotated[list, add_messages]
  time_response: Annotated[list, add_messages]
  rag_response: Annotated[list, add_messages]
    
state = {
    "input_querry": "",
    "next_node": "",
    "function_call": [],
    "objective_analysis_response": [],
    "time_response": [],
    "rag_response": [],
}
