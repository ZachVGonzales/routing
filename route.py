from datetime import datetime
from zoneinfo import ZoneInfo

from semantic_router import Route
from semantic_router.utils.function_call import get_schema
from semantic_router import RouteLayer
from graph.custom_encoder import STEncoder
from semantic_router.llms.ollama import OllamaLLM

enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)

def get_time(timezone: str) -> str:
    """Finds the current time in a specific timezone.

    :param timezone: The timezone to find the current time in, should
        be a valid timezone from the IANA Time Zone Database like
        "America/New_York" or "Europe/London". Do NOT put the place
        name itself like "rome", or "new york", you must provide
        the IANA format.
    :type timezone: str
    :return: The current time in the specified timezone."""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime("%H:%M")


time_schema = get_schema(get_time)
print(time_schema)

time = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schemas=[time_schema],
)

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

routes = [politics, chitchat, time]
encoder = STEncoder()

"""
_llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_gpu_layers=0,
    n_ctx=4096,
)
_llm.verbose = False
llm = LlamaCppLLM(name="Mistral-7B-v0.2-Instruct", llm=_llm, max_tokens=None)
"""

llm = OllamaLLM(llm_name="mistral:latest", temperature=0, max_tokens=2048)

rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)

out = rl("what is the time in new york?")
print("----------------------------------------------------------------------------------------")
print(out)
print(get_time(**out.function_call[0]))