# CitifyAI Agno Scripts

This directory contains Python scripts that implement the Agno agent integration for the CitifyAI project.

## Available Scripts

- `run_agent.py`: The main script that powers the Agno agent integration. It takes a query and agent type, processes the query using Agno, and returns the response in a format suitable for the Next.js frontend.

## Usage

These scripts are designed to be called from the Next.js API routes. You typically won't need to run them manually, but for testing or debugging, you can use the following commands:

```bash
# Activate the virtual environment
source ../../../agno_venv/bin/activate

# Run the agent with a test query
python run_agent.py --query "Where can I find good pizza in Philadelphia?" --agent restaurant

# Run with context data
python run_agent.py --query "What services does the city provide?" --agent government --context '{"user_location": "Center City"}'

# Run with the default agent
python run_agent.py --query "Tell me about Philadelphia"
```

## Environment Setup

This script requires the Agno Python library and an active OpenAI API key. The Agno library is already installed in the project's virtual environment (`agno_venv`).

For the OpenAI API key, ensure that it's set in your environment before running the script:

```bash
# Set OpenAI API key (replace with your actual key)
export OPENAI_API_KEY=your_openai_api_key_here
```

## Future Enhancements

- Add support for web search using Agno's search tools
- Implement knowledge bases for Philadelphia-specific information
- Add support for multi-step workflows using Agno's planning capabilities
