"""
Critique Labs API Tool for Agno

This module provides a custom Agno tool for querying the Critique Labs API,
which provides structured information about cities.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from .tool_base import Tool

class CritiqueLabsTool(Tool):
    """Tool for querying the Critique Labs API for structured city information."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, category: str = "general"):
        """
        Initialize the Critique Labs tool.
        
        Args:
            api_key: Critique Labs API key (defaults to env var CRITIQUE_LABS_API_KEY)
            base_url: Base URL for the Critique Labs API (defaults to env var CRITIQUE_LABS_URL or standard URL)
            category: Category of information to query (restaurant, government, general)
        """
        # Use the provided category in the tool name and description
        tool_name = f"critique_labs_{category}"
        tool_description = f"Query Critique Labs API for structured {category} information about cities"
        
        super().__init__(
            name=tool_name,
            description=tool_description,
            function=self.query_critique_labs
        )
        
        # Use provided values or fall back to environment variables
        self.api_key = api_key or os.getenv("CRITIQUE_LABS_API_KEY")
        self.base_url = base_url or os.getenv("CRITIQUE_LABS_URL") or "https://api.critique-labs.ai/v1/search"
        self.category = category
        
        if not self.api_key:
            raise ValueError("Critique Labs API key must be provided or set as CRITIQUE_LABS_API_KEY environment variable")
    
    def get_output_format(self) -> Dict[str, Any]:
        """Generate output format for API request based on category."""
        if self.category == "restaurant":
            return {
                "restaurants": [{
                    "name": "string",
                    "description": "string",
                    "address": "string",
                    "phone": "string",
                    "website": "string",
                    "cuisine": ["string"],
                    "priceRange": "string",
                    "rating": "number",
                    "neighborhood": "string"
                }]
            }
        elif self.category == "government":
            return {
                "governmentServices": [{
                    "name": "string",
                    "description": "string",
                    "department": "string",
                    "location": "string",
                    "contactInfo": {
                        "phone": "string",
                        "email": "string",
                        "website": "string"
                    },
                    "hours": "string",
                    "requirements": ["string"]
                }]
            }
        else:  # general or any other category
            return {
                "information": [{
                    "title": "string",
                    "description": "string",
                    "category": "string",
                    "source": "string",
                    "lastUpdated": "string"
                }]
            }
    
    def generate_prompt(self, query: str, city: str = "Philadelphia", neighborhood: Optional[str] = None, limit: int = 3) -> str:
        """
        Generate a prompt based on search options.
        
        Args:
            query: The user's query
            city: City to search for information about
            neighborhood: Specific neighborhood in the city (for restaurant queries)
            limit: Maximum number of results to return
            
        Returns:
            Formatted prompt string
        """
        if self.category == "restaurant":
            location_spec = f"in the {neighborhood} neighborhood of {city}" if neighborhood else f"in {city}"
            return f"Find {limit} restaurants {location_spec} that match the following query: \"{query}\". For each restaurant, provide the name, a detailed description (3-4 sentences), full address, phone number, website, cuisine types, and price range."
        
        elif self.category == "government":
            return f"Find {limit} government services in {city} related to the following query: \"{query}\". For each service, provide the name, description, department, location, contact information (phone, email, website), hours of operation, and any requirements."
        
        else:  # general or any other category
            return f"Find {limit} pieces of information about {city} related to the following query: \"{query}\". For each, provide a title, detailed description, category, source, and last updated date."
    
    async def query_critique_labs(self, query: str, city: str = "Philadelphia", neighborhood: Optional[str] = None, limit: int = 3) -> Dict[str, Any]:
        """
        Query Critique Labs API for structured information.
        
        Args:
            query: The user's query
            city: City to search for information about
            neighborhood: Specific neighborhood in the city (for restaurant queries)
            limit: Maximum number of results to return
            
        Returns:
            Structured response from Critique Labs
        """
        try:
            output_format = self.get_output_format()
            prompt = self.generate_prompt(query, city, neighborhood, limit)
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            payload = {
                "prompt": prompt,
                "output_format": output_format
            }
            
            print(f"Sending request to Critique Labs API: {payload}")
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # The API returns a nested structure where the actual response may be in a 'response' property as a JSON string
            data = response_data
            if 'response' in response_data and isinstance(response_data['response'], str):
                try:
                    # Try to parse the nested JSON string
                    data = json.loads(response_data['response'])
                except json.JSONDecodeError:
                    # If parsing fails, use the original response
                    print("Error parsing nested JSON response, using original response")
            
            # Extract the relevant data based on category
            if self.category == "restaurant":
                results = data.get("restaurants", [])
            elif self.category == "government":
                results = data.get("governmentServices", [])
            else:  # general or any other category
                results = data.get("information", [])
            
            # Structure the response to include metadata
            return {
                "query": query,
                "city": city,
                "category": self.category,
                "results": results,
                "result_count": len(results),
                "success": True
            }
            
        except requests.RequestException as e:
            print(f"Error querying Critique Labs API: {str(e)}")
            return {
                "query": query,
                "city": city,
                "category": self.category,
                "error": str(e),
                "success": False,
                "results": []
            }
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                "query": query,
                "city": city,
                "category": self.category,
                "error": f"Unexpected error: {str(e)}",
                "success": False,
                "results": []
            }
