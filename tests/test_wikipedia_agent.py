import os
import pytest
from unittest.mock import Mock, patch
from core.langgraph_agent import WikipediaAgent
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def agent():
    """Create a WikipediaAgent instance for testing."""
    return WikipediaAgent()

def test_agent_initialization(agent):
    """Test that the agent initializes with the correct components."""
    assert hasattr(agent, 'tools')
    assert hasattr(agent, 'memory')
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'agent_executor')

@patch('core.langgraph_agent.WikipediaQueryRun')
def test_wikipedia_search(mock_wiki, agent):
    """Test that the agent can perform Wikipedia searches."""
    mock_wiki.return_value.run.return_value = "Test result"
    
    response = agent.query("Tell me about quantum physics. Use just one query in your tool call.")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.skipif(os.getenv("USE_RAG", "false").lower() != "true", reason="RAG not enabled")
def test_rag_functionality(agent):
    """Test RAG functionality when enabled."""
    assert any(tool.name == "wikipedia_rag" for tool in agent.tools)
    
    response = agent.query("Tell me about quantum physics. Use just one query in your tool call.")
    assert isinstance(response, str)
    assert len(response) > 0 