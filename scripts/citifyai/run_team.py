#!/usr/bin/env python3
"""
CitifyAI Agent Team Runner

This script implements a team of specialized agents that work together
to provide comprehensive information about Philadelphia. It uses the Agno
framework's team architecture for coordinated agent responses.

Usage:
    python run_team.py --query "Where can I find good pizza in Philadelphia?" --session "user123"
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional, List
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

# Import Agno modules
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.storage.json import JsonStorage
import tempfile

# Import custom tools
from tools.critique_labs import CritiqueLabsTool
from tools.web_search import WebSearchTool

# Team configuration
TEAM_CONFIG = {
    "name": "PhiladelphiaInfoTeam",
    "description": "Team of specialists providing information about Philadelphia",
    "instructions": [
        "Work together to provide accurate, helpful information about Philadelphia.",
        "Route queries to the most appropriate specialist.",
        "Provide comprehensive responses that cite sources when possible.",
        "If uncertain, acknowledge limitations and suggest where to find more information."
    ]
}

# Agent configurations
AGENT_CONFIGS = {
    "restaurant": {
        "name": "restaurant_agent",
        "description": "Restaurant information specialist for Philadelphia",
        "instructions": [
            "You are Philadelphia's restaurant expert.",
            "Provide detailed information about restaurants, cafes, and bars.",
            "Include cuisine type, neighborhood, and special features when recommending places.",
            "Use the critique_labs_restaurant tool to get accurate, up-to-date information.",
            "Use the web_search tool for additional details or recent information."
        ],
        "tools": ["critique_labs_restaurant", "web_search"]
    },
    
    "government": {
        "name": "government_agent",
        "description": "Government services specialist for Philadelphia",
        "instructions": [
            "You are Philadelphia's government services expert.",
            "Provide accurate information about city services, permits, licenses, etc.",
            "Include contact information for relevant departments when available.",
            "Use the critique_labs_government tool to get accurate, up-to-date information.",
            "Use the web_search tool for additional details or recent information."
        ],
        "tools": ["critique_labs_government", "web_search"]
    },
    
    "default": {
        "name": "general_agent",
        "description": "General information specialist for Philadelphia",
        "instructions": [
            "You are Philadelphia's general information assistant.",
            "Provide helpful information about the city, its history, attractions, and more.",
            "If you don't know something, acknowledge that and offer to help with something else.",
            "Use the critique_labs_general tool to get accurate, up-to-date information.",
            "Use the web_search tool for additional details or recent information."
        ],
        "tools": ["critique_labs_general", "web_search"]
    }
}

def create_tool(tool_name: str) -> Any:
    """
    Create and return a tool instance based on the tool name.
    
    Args:
        tool_name: Name of the tool to create
        
    Returns:
        Instantiated tool object
    """
    try:
        if tool_name.startswith("critique_labs"):
            category = tool_name.split("_")[-1] if "_" in tool_name else "general"
            return CritiqueLabsTool(category=category)
        elif tool_name == "web_search":
            return WebSearchTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        print(f"Error creating tool {tool_name}: {str(e)}")
        # Return a simple no-op tool as fallback
        return None

def initialize_agent_team(memory: Optional[JsonStorage] = None) -> Team:
    """
    Initialize a team of specialized agents for Philadelphia information.
    
    Args:
        memory: Optional memory storage for session persistence
        
    Returns:
        Configured agent team
    """
    # Create a temporary storage directory if not provided
    if memory is None:
        storage_dir = os.path.join(tempfile.gettempdir(), "citifyai_storage")
        os.makedirs(storage_dir, exist_ok=True)
        memory = JsonStorage(storage_dir, "team")
    
    # Create agents for the team
    agents = []
    for agent_type, config in AGENT_CONFIGS.items():
        # Create tools for this agent
        tools = []
        for tool_name in config.get("tools", []):
            tool = create_tool(tool_name)
            if tool:
                tools.append(tool)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment")
        
        # Create the agent with tools (without memory to avoid create_user_memories error)
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=api_key),
            name=config["name"],
            description=config["description"],
            instructions=config["instructions"],
            tools=tools,
            markdown=True
        )
        agents.append(agent)
    
    # Create the team with agents as members (correct parameter name)
    # Note: Removing memory parameter as current version of JsonStorage doesn't support create_user_memories
    team = Team(
        name=TEAM_CONFIG["name"],
        description=TEAM_CONFIG["description"],
        instructions=TEAM_CONFIG["instructions"],
        members=agents,  # Using 'members' instead of 'agents'
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))  # Using 'model' instead of 'router'
    )
    
    return team

def process_query(query: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user query with the agent team.
    
    Args:
        query: The user's query text
        session_id: Unique session ID for context management
        context: Optional additional context information
        
    Returns:
        A dictionary containing the agent's response
    """
    try:
        # Initialize memory storage for persistent context
        storage_dir = os.path.join(tempfile.gettempdir(), "citifyai_storage")
        os.makedirs(storage_dir, exist_ok=True)
        memory = JsonStorage(storage_dir, "team")
        
        # Initialize agent team
        team = initialize_agent_team(memory)
        
        # Process the query
        response = team.run(query, session_id=session_id, context=context)
        
        # Determine which agent handled the query
        agent_type = "team"
        if hasattr(response, 'agent_name'):
            if response.agent_name == "restaurant_agent":
                agent_type = "restaurant"
            elif response.agent_name == "government_agent":
                agent_type = "government"
            elif response.agent_name == "general_agent":
                agent_type = "default"
        
        # Format the response for the frontend
        result = {
            "query": query,
            "response": response.content if hasattr(response, 'content') else str(response),
            "timestamp": response.timestamp if hasattr(response, 'timestamp') else time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sessionId": session_id,
            "agentType": agent_type
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        
        # Return error response
        return {
            "query": query,
            "response": f"I apologize, but I encountered an error while processing your request. Please try again or ask a different question.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sessionId": session_id,
            "agentType": "default",
            "error": str(e)
        }

def main():
    """Command line interface for the team runner."""
    parser = argparse.ArgumentParser(description="Run CitifyAI Agent Team")
    parser.add_argument("--query", required=True, help="User query to process")
    parser.add_argument("--session", default=None, help="Session ID for memory persistence")
    parser.add_argument("--context", help="Optional JSON context string")
    parser.add_argument("--context-base64", help="Optional base64-encoded JSON context string")
    
    args = parser.parse_args()
    
    # Generate session ID if not provided
    session_id = args.session or f"session_{int(time.time())}"
    
    # Parse context if provided (either from regular context or base64-encoded context)
    context = None
    if args.context_base64:
        import base64
        try:
            decoded_context = base64.b64decode(args.context_base64).decode('utf-8')
            context = json.loads(decoded_context)
        except Exception as e:
            print(f"Error decoding base64 context: {str(e)}")
    elif args.context:
        try:
            context = json.loads(args.context)
        except Exception as e:
            print(f"Error parsing JSON context: {str(e)}")
    
    # Process the query
    result = process_query(args.query, session_id, context)
    
    # Print the result as JSON for the API to parse
    print(json.dumps(result))

if __name__ == "__main__":
    main()
