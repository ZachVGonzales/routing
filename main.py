from graph.graph import create_graph
import uvicorn
from slm_hub.service import app
import threading


def start_server():
    """Starts the FastAPI server in a separate thread."""
    uvicorn.run(app=app, host="0.0.0.0", port=8208)


if __name__ == "__main__":
  # launch the SLM hub
  server_thread = threading.Thread(target=start_server, daemon=True)
  server_thread.start()

  graph = create_graph()
  graph = graph.compile()

  while True:
    query = input("Please enter your research question: ")
    if query.lower() == "exit":
      break

    dict_inputs = {"input_querry": query}

    for event in graph.stream(dict_inputs):
      print("\nState Dictionary:", event)
  