from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.tools.retriever import create_retriever_tool
import os
from dotenv import load_dotenv
import logging
from .logging_config import setup_logging
import glob
from .rag import create_rag_tool

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

LLM = os.getenv("LLM", "claude-3-5-sonnet-latest")
USE_RAG = os.getenv("USE_RAG", "false").lower() == "true"

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful research assistant that specializes in searching Wikipedia for information.
You aim to provide accurate, well-researched answers based on Wikipedia content.
When asked a question, you should break it down into multiple search queries if needed to gather comprehensive information.
For complex topics, you should make multiple Wikipedia searches to cross-reference and combine information from different articles.
If you're unsure about something or can't find information on Wikipedia, you'll be honest about it.
You'll maintain a professional and informative tone while being engaging and clear in your responses.
Always give sources of the information you provide, which pages you got the information from, and links to the pages.
Remember to synthesize information from multiple sources when appropriate to provide the most complete and accurate answer.
""") if not USE_RAG else os.getenv("SYSTEM_PROMPT", """You are a helpful research assistant that specializes in searching Wikipedia for information.
You aim to provide accurate, well-researched answers based on Wikipedia content.
When asked a question, you will search Wikipedia to find relevant information.
You have access to a RAG tool that allows you to search through Wikipedia content efficiently.
If you're unsure about something or can't find information on Wikipedia, you'll be honest about it.
You'll maintain a professional and informative tone while being engaging and clear in your responses.
""")

class WikipediaAgent:
    """
    A research assistant that leverages LLMs and Wikipedia to provide comprehensive answers.
    
    This agent can use either direct Wikipedia search or RAG (Retrieval Augmented Generation)
    to provide well-researched answers based on Wikipedia content. It supports multiple LLM
    providers including OpenAI, Anthropic, and Google.

    Attributes:
        tools (list): List of tools available to the agent
        memory (MemorySaver): Memory component for maintaining conversation state
        model (Union[ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI]): The LLM model
        agent_executor: The LangGraph agent executor
    """

    def __init__(self) -> None:
        """Initialize the WikipediaAgent with the configured LLM and tools."""
        logger.info("WikipediaAgent.__init__ :: initializing...")

        logger.info(f"WikipediaAgent.__init__ :: Using model: {LLM}")

        # Initialize tools based on USE_RAG setting
        if USE_RAG:
            rag_tool = create_rag_tool()
            self.tools = [rag_tool]
            logger.info("WikipediaAgent.__init__ :: Using RAG tool")
        else:
            @tool
            def wikipedia_search(queries: list[str]) -> str:
                """
                Search Wikipedia for information about multiple topics.
                You can pass multiple queries to gather comprehensive information.
                
                :param queries: List of search query strings
                :return: Combined Wikipedia article content from all queries
                """
                wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                results = []
                for query in queries:
                    results.append(wikipedia.run(query))
                return "\n\n".join(results)
            
            self.tools = [wikipedia_search]
            logger.info("WikipediaAgent.__init__ :: Using Wikipedia Search tool")
        
        # Create the agent
        logger.info("WikipediaAgent.__init__ :: Creating agent components...")
        self.memory = MemorySaver()
        logger.info("WikipediaAgent.__init__ :: Memory initialized")

        if LLM.startswith("claude"):
            self.model = ChatAnthropic(model_name=LLM, api_key=ANTHROPIC_API_KEY)
            logger.info(
                f"WikipediaAgent.__init__ :: ChatAnthropic model initialized with {LLM}"
            )
        elif LLM.startswith("gemini"):
            self.model = ChatGoogleGenerativeAI(
                model=LLM,
                api_key=GOOGLE_API_KEY
            )
            logger.info(
                f"WikipediaAgent.__init__ :: ChatGoogleGenerativeAI model initialized with {LLM}"
            )
        else:
            self.model = ChatOpenAI(model=LLM, api_key=OPENAI_API_KEY)
            logger.info(
                f"WikipediaAgent.__init__ :: ChatOpenAI model initialized with {LLM}"
            )

        logger.info(f"WikipediaAgent.__init__ :: Using system prompt: {SYSTEM_PROMPT}")
        self.agent_executor = create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory,
            state_modifier=SYSTEM_PROMPT
        )
        logger.info("WikipediaAgent.__init__ :: Agent executor created")

        logger.info("WikipediaAgent initialized successfully")

    def query(self, input_text: str, thread_id: str = "abc123") -> str:
        """
        Process a user query using the Wikipedia agent.
        
        Args:
            input_text (str): The user's question or query
            thread_id (str, optional): Unique identifier for the conversation thread. 
                                     Defaults to "abc123".
        
        Returns:
            str: The agent's response to the query
            
        Raises:
            Exception: If there's an error during agent execution
        """
        logger.info(f"WikipediaAgent.query :: received query: {input_text}")
        logger.debug(f"WikipediaAgent.query :: using thread_id: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        try:
            inputs = {"messages": [
                HumanMessage(content=input_text)
            ]}

            stream = self.agent_executor.stream(
                inputs, config, stream_mode="values"
            )

            for s in stream:
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
                logger.debug(f"Message: {message}")

            logger.info(f"WikipediaAgent.query :: agent response: {message.content}")
            return message.content

        except Exception as e:
            logger.exception(
                f"WikipediaAgent.query :: Error during agent execution: {str(e)}"
            )
            return f"An error occurred: {str(e)}"
