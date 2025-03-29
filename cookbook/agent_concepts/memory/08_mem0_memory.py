"""
This example demonstrates how to use the mem0 MemoryClient to store and retrieve memory.

Sign up for mem0 at https://mem0.ai/

Export the MEM0_API_KEY environment variable before running the script.
"""

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response
from mem0 import MemoryClient

client = MemoryClient()

user_id = "agno"
messages = [
    {"role": "user", "content": "My name is John Billings."},
    {"role": "user", "content": "I live in NYC."},
    {"role": "user", "content": "I'm going to a concert tomorrow."},
]
# Comment out the following line after running the script once
client.add(messages, user_id=user_id, output_format="v1.1")

agent = Agent(
    model=OpenAIChat(),
    context={"memory": client.get_all(user_id=user_id)},
    add_context=True,
)
run: RunResponse = agent.run("What do you know about me?")

pprint_run_response(run)

messages = [{"role": i.role, "content": str(i.content)} for i in (run.messages or [])]
client.add(messages, user_id=user_id, output_format="v1.1")
