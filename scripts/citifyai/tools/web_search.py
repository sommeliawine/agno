"""
Web Search Tool for Agno

This module provides a custom Agno tool for performing web searches to retrieve
up-to-date information about Philadelphia. In a production environment, this would
be integrated with an actual search API like Google, Bing, or DuckDuckGo.
"""

import os
from typing import Dict, Any, List, Optional
from .tool_base import Tool

class WebSearchTool(Tool):
    """Tool for performing web searches to get up-to-date information."""
    
    def __init__(self):
        """Initialize the web search tool."""
        super().__init__(
            name="web_search",
            description="Search the web for up-to-date information about Philadelphia",
            function=self.search_web
        )
    
    async def search_web(self, query: str, city: str = "Philadelphia", num_results: int = 3) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            city: City to focus the search on (default: Philadelphia)
            num_results: Number of results to return (default: 3)
            
        Returns:
            Search results
        """
        try:
            # Add the city to the query if not already present
            search_query = query
            if city.lower() not in query.lower():
                search_query = f"{city} {query}"
            
            print(f"Performing web search for: {search_query}")
            
            # In a real implementation, this would call a search API
            # For now, we'll return mock results based on the query
            mock_results = self._get_mock_results(search_query)
            
            # Return only the requested number of results
            limited_results = mock_results[:num_results]
            
            return {
                "query": search_query,
                "results": limited_results,
                "result_count": len(limited_results),
                "success": True
            }
            
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "success": False,
                "results": []
            }
    
    def _get_mock_results(self, query: str) -> List[Dict[str, str]]:
        """
        Generate mock search results based on the query keywords.
        
        Args:
            query: Search query
            
        Returns:
            List of mock search results
        """
        # Default results for general Philadelphia queries
        default_results = [
            {
                "title": "Visit Philadelphia - Official Visitor Website",
                "url": "https://www.visitphilly.com",
                "snippet": "Official visitor information for Philadelphia, PA. Comprehensive guide to attractions, restaurants, events, and things to do throughout the city."
            },
            {
                "title": "City of Philadelphia - Official Website",
                "url": "https://www.phila.gov",
                "snippet": "The official website of the City of Philadelphia, with information about city services, departments, programs, and initiatives."
            },
            {
                "title": "Philadelphia Tourism and Travel Guide",
                "url": "https://www.philadelphia.travel",
                "snippet": "Plan your trip to Philadelphia with our comprehensive travel guide, including hotels, attractions, tours, and local tips."
            }
        ]
        
        # Restaurant-related results
        restaurant_keywords = ["restaurant", "food", "eat", "dining", "cuisine", "cafe", "bar"]
        if any(keyword in query.lower() for keyword in restaurant_keywords):
            return [
                {
                    "title": "Best Restaurants in Philadelphia - OpenTable",
                    "url": "https://www.opentable.com/philadelphia-restaurants",
                    "snippet": "Find the best places to eat in Philadelphia. Book popular restaurants and read reviews from diners. Discover top-rated dining experiences in the city."
                },
                {
                    "title": "50 Best Restaurants in Philadelphia - Philadelphia Magazine",
                    "url": "https://www.phillymag.com/foobooz/50-best-restaurants-in-philadelphia/",
                    "snippet": "Our annual list of the best restaurants in Philadelphia, from fine dining establishments to casual neighborhood favorites."
                },
                {
                    "title": "Where to Eat in Philadelphia - Eater Philly",
                    "url": "https://philly.eater.com/maps/best-restaurants-philadelphia",
                    "snippet": "The essential guide to dining in Philadelphia, featuring the hottest new restaurants, classic establishments, and hidden gems throughout the city."
                },
                {
                    "title": "Philadelphia Restaurant Week",
                    "url": "https://www.ccdphila.org/center-city-fit/restaurant-week/",
                    "snippet": "Experience the best of Philadelphia's dining scene during Restaurant Week, with special menus and pricing at participating restaurants."
                }
            ]
        
        # Government-related results
        government_keywords = ["government", "city hall", "permit", "license", "department", "services", "official"]
        if any(keyword in query.lower() for keyword in government_keywords):
            return [
                {
                    "title": "City of Philadelphia - Services",
                    "url": "https://www.phila.gov/services/",
                    "snippet": "Find information on city services, including permits, licenses, taxes, and more. Access online applications and forms for residents and businesses."
                },
                {
                    "title": "Philadelphia City Departments",
                    "url": "https://www.phila.gov/departments/",
                    "snippet": "Directory of Philadelphia city departments and agencies, with contact information, hours of operation, and service details."
                },
                {
                    "title": "Philadelphia Business Services",
                    "url": "https://business.phila.gov",
                    "snippet": "Resources for businesses in Philadelphia, including permits, licenses, regulations, and incentive programs for small business owners."
                },
                {
                    "title": "Philadelphia City Council",
                    "url": "https://phlcouncil.com",
                    "snippet": "Official website of the Philadelphia City Council, with information about council members, meetings, legislation, and public hearings."
                }
            ]
        
        # Tourism-related results
        tourism_keywords = ["visit", "tourism", "attraction", "museum", "historic", "tour", "liberty bell"]
        if any(keyword in query.lower() for keyword in tourism_keywords):
            return [
                {
                    "title": "Top 25 Attractions in Philadelphia - Visit Philadelphia",
                    "url": "https://www.visitphilly.com/articles/philadelphia/top-25-attractions-in-philadelphia/",
                    "snippet": "Discover Philadelphia's must-see attractions, including the Liberty Bell, Independence Hall, Philadelphia Museum of Art, and more."
                },
                {
                    "title": "Philadelphia Museum Guide",
                    "url": "https://www.visitphilly.com/museums-in-philadelphia/",
                    "snippet": "Comprehensive guide to Philadelphia's world-class museums, including art collections, science centers, historic sites, and special exhibitions."
                },
                {
                    "title": "Historic Philadelphia Tours",
                    "url": "https://www.nps.gov/inde/planyourvisit/tours.htm",
                    "snippet": "Guided tours of Philadelphia's historic sites in Independence National Historical Park, including ranger-led programs and self-guided options."
                },
                {
                    "title": "Things to Do in Philadelphia - TripAdvisor",
                    "url": "https://www.tripadvisor.com/Attractions-g60795-Activities-Philadelphia_Pennsylvania.html",
                    "snippet": "Browse hundreds of attractions, tours, and activities in Philadelphia. Read reviews from travelers and book your perfect Philadelphia experience."
                }
            ]
        
        # Return default results if no specific category is matched
        return default_results
