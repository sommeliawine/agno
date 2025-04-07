"""
Base Tool Implementation

This module provides a base Tool class that simulates the Agno Tool interface
for custom tool implementations in CitifyAI.
"""

from typing import Any, Callable, Dict, Optional


class Tool:
    """Base class for implementing custom tools for CitifyAI."""
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            function: The async function to execute when the tool is used
        """
        self.name = name
        self.description = description
        self.function = function
        self.__name__ = name  # Add __name__ attribute expected by Agno
    
    async def __call__(self, *args, **kwargs) -> Any:
        """
        Call the tool's function.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of calling the function
        """
        return await self.function(*args, **kwargs)
