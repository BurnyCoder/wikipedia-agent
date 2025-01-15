from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import glob
import logging
from .logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def create_rag_tool():
    """
    Create and return the RAG tool for knowledge base search.
    
    This method sets up a FAISS vector store with the documents from the rag_data
    directory and creates a retriever tool for semantic search.
    
    Returns:
        Tool: A LangChain tool for RAG-based retrieval
    """
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(embeddings)
    
    # Load and process RAG data files
    documents = []
    rag_files = glob.glob("rag_data/*.txt")
    for file_path in rag_files:
        logger.info(f"create_rag_tool :: Loading RAG data from {file_path}")
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    
    # Split documents into chunks using semantic chunker
    texts = text_splitter.create_documents([doc.page_content for doc in documents])
    logger.info(f"create_rag_tool :: Created {len(texts)} text chunks from RAG data")
    
    # Create FAISS database with documents
    db = FAISS.from_documents(texts, embeddings)
    logger.info("create_rag_tool :: Created FAISS database with RAG data")
    
    retriever = db.as_retriever()
    wikipedia_rag = create_retriever_tool(
        retriever,
        "wikipedia_rag",
        "Searches through the knowledge base for relevant information.",
    )
    
    return wikipedia_rag 