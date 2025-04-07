#!/usr/bin/env python3
"""
CitifyAI Agent Runner Script

This script manages the interface between the Next.js application and Agno's agent framework.
It receives a query and agent type, processes it with the appropriate Agno agent,
and returns the response in a format suitable for the frontend.

Usage:
    python run_agent.py --query "Where can I find good pizza in Philadelphia?" --agent restaurant
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

# Add the Agno library to the Python path
agno_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs/agno'))
if agno_path not in sys.path:
    sys.path.append(agno_path)

from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Define default agent descriptions and instructions
AGENT_CONFIGS = {
    "default": {
        "description": "You are a helpful assistant for the city of Philadelphia. Provide general information about the city.",
        "instructions": [
            "Always introduce yourself as Philadelphia's digital assistant.",
            "If you don't know something, say so and offer to help with something else.",
            "Keep responses concise and helpful.",
            "When discussing locations, include the neighborhood if possible."
        ]
    },
    "restaurant": {
        "description": "You are a restaurant guide for Philadelphia. Help users find great places to eat and drink.",
        "instructions": [
            "Always introduce yourself as Philadelphia's restaurant guide.",
            "Focus on providing information about restaurants, cafes, and bars.",
            "Include cuisine type, neighborhood, and any special features when recommending places.",
            "Suggest alternatives when appropriate."
        ]
    },
    "government": {
        "description": "You are a guide for government services in Philadelphia. Help users navigate city services and resources.",
        "instructions": [
            "Always introduce yourself as Philadelphia's government services assistant.",
            "Focus on providing accurate information about city services, permits, licenses, etc.",
            "When applicable, include information about how to contact the relevant department.",
            "Provide links or phone numbers when referring to specific services."
        ]
    }
}

def initialize_agent(agent_type: str) -> Agent:
    """
    Initialize an Agno agent based on the specified type.
    
    Args:
        agent_type: Type of agent to initialize (default, restaurant, government)
        
    Returns:
        An initialized Agno agent
    """
    config = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["default"])
    
    # For now, we're using a simple agent with OpenAI
    # In the future, we'll add tools and knowledge bases
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description=config["description"],
        instructions=config["instructions"],
        markdown=True
    )
    
    return agent

def process_query(query: str, agent_type: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user query with the appropriate agent.
    
    Args:
        query: The user's query text
        agent_type: The type of agent to use
        context: Optional context information (e.g., user history)
        
    Returns:
        A dictionary containing the agent's response
    """
    # Initialize the appropriate agent
    agent = initialize_agent(agent_type)
    
    # Process the query
    response = agent.run(query)
    
    # Format the response for the frontend
    result = {
        "agentType": agent_type,
        "query": query,
        "response": response.content if hasattr(response, 'content') else str(response),
        "timestamp": response.timestamp if hasattr(response, 'timestamp') else None,
    }
    
    return result

def main():
    """Command line interface for the agent runner."""
    parser = argparse.ArgumentParser(description="Run CitifyAI Agents")
    parser.add_argument("--query", required=True, help="User query to process")
    parser.add_argument("--agent", default="default", choices=["default", "restaurant", "government"], 
                        help="Type of agent to use")
    parser.add_argument("--context", help="Optional JSON context string")
    
    args = parser.parse_args()
    
    context = json.loads(args.context) if args.context else None
    result = process_query(args.query, args.agent, context)
    
    # Print the result as JSON for the API to parse
    print(json.dumps(result))

if __name__ == "__main__":
    main()
