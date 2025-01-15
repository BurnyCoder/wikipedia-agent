import core.langgraph_agent as WikipediaAgent
import logging

logger = logging.getLogger(__name__)

def run_terminal():
    # Initialize the WikipediaAgent
    logger.info("Initializing WikipediaAgent")
    
    while True:
        user_input = input("You: ")
        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            logger.info("User requested exit")
            print("Exiting...")
            break
        
        # Use the agent's query method
        logger.debug(f"Processing user input: {user_input}")
        output = WikipediaAgent.query(user_input)
        logger.debug(f"Agent response received")
        print("\n\nAssistant:\n", output)

if __name__ == "__main__":
    logger.info("Starting terminal interface")
    run_terminal()
