"""
GraphRAG module for handling graph-based RAG operations.

This module provides functionality for setting up, creating, and querying
graph-based RAG systems using the GraphRAG library.
"""

import os
import subprocess
import logging
from dotenv import load_dotenv
from core.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
GRAPHRAG_API_KEY = os.getenv("GRAPHRAG_API_KEY")


class GraphRAG:
    """Class to handle graph-based RAG operations."""
    
    def __init__(self, input_files=None):
        """
        Initialize GraphRAG with input files.
        
        Args:
            input_files (list[str], optional): List of paths to input files. 
                                             If None, will use all .txt files in rag_data/.
        """
        if input_files is None:
            # Default to all txt files in rag_data directory
            rag_data_dir = os.path.join(os.path.dirname(os.getcwd()), "rag_data")
            self.input_files = [
                os.path.join(rag_data_dir, f) 
                for f in os.listdir(rag_data_dir) 
                if f.endswith('.txt')
            ]
        else:
            self.input_files = input_files
        
        logger.info(f"Initialized GraphRAG with input files: {self.input_files}")

    def setup(self):
        """Set up GraphRAG environment and dependencies."""
        logger.info("Setting up GraphRAG...")
        setup_graph_rag(self.input_files)

    def create_graph(self):
        """Create graph from input file."""
        logger.info("Creating the graph...")
        create_graph(self.input_files)

    def query_graph(self, query: str, method: str = 'local'):
        """
        Query the constructed graph.
        
        Args:
            query (str): Query string to process
            method (str): Query method, either 'local' or 'global'
        """
        logger.info(f"Querying the graph with method '{method}': '{query}'")
        result = use_constructed_graph(query, method=method)
        logger.info("Query completed")
        print("Result:")
        print(result)


def setup_graph_rag(input_files: list[str]):
    """
    Set up GraphRAG environment and dependencies.
    
    Args:
        input_files (list[str]): List of paths to input files
    """
    logger.info("Creating directories and installing dependencies")
    
    # Create directory and navigate into it
    os.makedirs(os.path.join(os.getcwd(), "graph_rag"), exist_ok=True)
    os.chdir(os.path.join(os.getcwd(), "graph_rag"))

    # Install GraphRAG
    try:
        subprocess.run([
            "pip", "install", "openai", "networkx", "leidenalg",
            "cdlib", "python-igraph", "python-dotenv"
        ], check=True)
        logger.info("Successfully installed dependencies")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise

    # Create input directory and setup files
    try:
        _setup_input_files(input_files)
        _update_settings()
        logger.info("GraphRAG setup completed successfully")
    except Exception as e:
        logger.error(f"Failed to complete GraphRAG setup: {e}")
        raise


def create_graph(input_files: list[str]):
    """
    Create graph from input files.
    
    Args:
        input_files (list[str]): List of paths to input files
    """
    logger.info(f"Creating graph from input files: {input_files}")
    
    # Ensure input files exist
    input_dir = os.path.join(os.getcwd(), "ragtest", "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Combine all input files into one
    with open(os.path.join(input_dir, "input.txt"), "w", encoding="utf-8") as outfile:
        for input_file in input_files:
            logger.info(f"Processing input file: {input_file}")
            with open(input_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")  # Add separation between files
    
    try:
        subprocess.run([
            "python", "-m", "graphrag.index",
            "--root", os.path.join(os.getcwd(), "ragtest")
        ], check=True)
        logger.info("Graph creation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create graph: {e}")
        raise


def use_constructed_graph(query: str, method: str = 'local') -> str:
    """
    Query the constructed graph.
    
    Args:
        query (str): Query string to process
        method (str): Query method, either 'local' or 'global'
    
    Returns:
        str: Query result
    
    Raises:
        ValueError: If method is not 'local' or 'global'
    """
    if method not in ['local', 'global']:
        logger.error(f"Invalid method specified: {method}")
        raise ValueError("Method must be either 'local' or 'global'")
    
    logger.info(f"Querying graph with method '{method}': {query}")
    try:
        result = subprocess.run(
            [
                "python", "-m", "graphrag.query",
                "--root", os.path.join(os.getcwd(), "ragtest"),
                "--method", method, query
            ],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Query completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to query graph: {e}")
        raise


def _setup_input_files(input_files: list[str]):
    """
    Set up input files for GraphRAG.
    
    Args:
        input_files (list[str]): List of paths to input files
    """
    input_dir = os.path.join(os.getcwd(), "ragtest", "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Combine all input files into one
    with open(os.path.join(input_dir, "input.txt"), "w", encoding="utf-8") as outfile:
        for input_file in input_files:
            logger.info(f"Processing input file: {input_file}")
            with open(input_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")  # Add separation between files
    
    # Initialize GraphRAG
    subprocess.run([
        "python", "-m", "graphrag.index", "--init",
        "--root", os.path.join(os.getcwd(), "ragtest")
    ], check=True)
    
    # Write .env file
    with open(os.path.join(os.getcwd(), "ragtest", ".env"), "w") as env_file:
        env_file.write(f'GRAPHRAG_API_KEY="{GRAPHRAG_API_KEY}"')


def _update_settings():
    """Update GraphRAG settings."""
    settings_path = os.path.join(os.getcwd(), "ragtest", "settings.yaml")
    with open(settings_path, "r") as file:
        settings_content = file.read()
    
    # Update model
    settings_content = settings_content.replace(
        "model: gpt-4-turbo-preview",
        "model: gpt-4o"
    )
    
    with open(settings_path, "w") as file:
        file.write(settings_content)
    
    logger.info("Updated settings.yaml: Changed model to gpt-4o")


if __name__ == "__main__":
    # Example usage with the provided RAG data files
    graph_rag = GraphRAG()  # Will automatically use all txt files in rag_data/
    graph_rag.setup()
    graph_rag.create_graph()
    
    # Test queries relevant to the content
    test_queries = [
        "What are the main risks and ethical concerns with AI?",
        "How is physics used in engineering and applied sciences?",
        "What is deep learning and how has it impacted AI development?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        graph_rag.query_graph(query, method='local')