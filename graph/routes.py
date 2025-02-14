"""
Defines the functionality for nodes of the AI Agent's Graph
"""

import requests
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from semantic_router import Route
from semantic_router.utils.function_call import get_schema

async def analyze_objective(objective: str) -> str:
    """Preforms an analysis of the provided objective.

    :param objective: The objective to perform an analysis on; should
        take the form of a sentance like "the student shall run a mile"
        or "the professor shall grade the essays without help". the
        objective MUST be extracted directly from the user's querry. do
        NOT edit the extracted objective in any way.
    :type objective: str
    :return: Analysis scores of the objective."""
    
    response = await asyncio.to_thread(requests.post, "http://localhost:8208/predict", json={"model_name":"objective_analysis", "text":objective})
    response = response.json()
    analysis = response["prediction"]
    return f"the objective scorded: {analysis}"

objective_schema = get_schema(analyze_objective)
objective_route = Route(name="analyze_objective",
                        utterances=[
                            "analyze the following objective: the student shall run a mile",
                            "perform an analysis on the objective: the student shall write a report in less than 300 words"
                            "give me an analysis on, \"given a cheatsheet the trainee will perform steam turbine analysis with an accuracy of 100%\""
                        ],
                        function_schemas=[objective_schema])


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
time_route = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schemas=[time_schema],
)