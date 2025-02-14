from graph.graph import create_graph


if __name__ == "__main__":
  graph = create_graph()
  graph = graph.compile()

  while True:
    query = input("Please enter your research question: ")
    if query.lower() == "exit":
      break

    dict_inputs = {"input_querry": query}

    for event in graph.stream(dict_inputs):
      print("\nState Dictionary:", event)