"""Unit tests for the WebSearchTool."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime

from scholar_verse.sub_agents.deep_search.tools.web_search import WebSearchTool, WebSearchParameters


@pytest.fixture
def web_search_tool():
    """Fixture that provides a WebSearchTool instance for testing."""
    return WebSearchTool()


@pytest.mark.asyncio
async def test_web_search_basic(web_search_tool):
    """Test basic web search functionality."""
    # Arrange
    test_query = "test query"
    
    # Act
    result = await web_search_tool._call({
        "query": test_query,
        "search_type": "general",
        "max_results": 3
    })
    
    # Assert
    assert result["success"] is True
    assert result["query"] == test_query
    assert len(result["results"]) > 0
    assert len(result["results"]) <= 3
    assert "metadata" in result


@pytest.mark.asyncio
async def test_web_search_with_domain_filters(web_search_tool):
    """Test web search with domain filters."""
    # Arrange
    test_query = "test query"
    include_domains = ["example.com"]
    
    # Act
    result = await web_search_tool._call({
        "query": test_query,
        "search_type": "general",
        "max_results": 5,
        "include_domains": include_domains
    })
    
    # Assert
    assert result["success"] is True
    assert len(result["results"]) > 0
    assert all("example.com" in result["url"] for result in result["results"])


@pytest.mark.asyncio
async def test_web_search_with_exclude_domains(web_search_tool):
    """Test web search with excluded domains."""
    # Arrange
    test_query = "test query"
    exclude_domains = ["wikipedia.org"]
    
    # Act
    result = await web_search_tool._call({
        "query": test_query,
        "search_type": "general",
        "max_results": 5,
        "exclude_domains": exclude_domains
    })
    
    # Assert
    assert result["success"] is True
    if result["results"]:  # Only check if we have results
        assert all("wikipedia.org" not in result["url"] for result in result["results"])


@pytest.mark.asyncio
async def test_web_search_academic_type(web_search_tool):
    """Test academic search type."""
    # Arrange
    test_query = "quantum computing"
    
    # Act
    result = await web_search_tool._call({
        "query": test_query,
        "search_type": "academic",
        "max_results": 3
    })
    
    # Assert
    assert result["success"] is True
    assert result["search_type"] == "academic"
    assert len(result["results"]) > 0
    assert all("citation_count" in result for result in result["results"])


@pytest.mark.asyncio
async def test_web_search_news_type(web_search_tool):
    """Test news search type."""
    # Arrange
    test_query = "latest research"
    
    # Act
    result = await web_search_tool._call({
        "query": test_query,
        "search_type": "news",
        "max_results": 2
    })
    
    # Assert
    assert result["success"] is True
    assert result["search_type"] == "news"
    assert len(result["results"]) > 0
    assert all("publish_date" in result for result in result["results"])


@pytest.mark.asyncio
async def test_web_search_error_handling(web_search_tool):
    """Test error handling in web search."""
    # Arrange - Test with invalid parameters
    
    # Act
    with patch.object(web_search_tool, '_simulate_search_results', side_effect=Exception("Test error")):
        result = await web_search_tool._call({
            "query": "test query",
            "search_type": "invalid_type"
        })
    
    # Assert
    assert result["success"] is False
    assert "error" in result
    assert result["query"] == "test query"


def test_web_search_parameters_validation():
    """Test WebSearchParameters validation."""
    # Test valid parameters
    params = WebSearchParameters(
        query="test",
        search_type="academic",
        max_results=10,
        include_domains=["example.com"],
        exclude_domains=[]
    )
    
    assert params.query == "test"
    assert params.search_type == "academic"
    assert params.max_results == 10
    assert params.include_domains == ["example.com"]
    assert params.exclude_domains == []
    
    # Test max_results bounds
    params = WebSearchParameters(query="test", max_results=100)
    assert params.max_results == 20  # Should be clamped to max 20
    
    params = WebSearchParameters(query="test", max_results=0)
    assert params.max_results == 1  # Should be clamped to min 1
