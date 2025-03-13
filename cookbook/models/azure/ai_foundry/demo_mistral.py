from agno.agent import Agent, RunResponse  # noqa
from agno.models.azure.ai_foundry import AzureAIFoundry

agent = Agent(model=AzureAIFoundry(id="Mistral-Large-2411"), markdown=True)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story")
# print(run.content)

# Print the response on the terminal
agent.print_response("Share a 2 sentence horror story")
