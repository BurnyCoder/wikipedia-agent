#!/usr/bin/env python3
"""
Terminal interface for the Wikipedia Research Assistant.

This module provides a command-line interface for interacting with the
Wikipedia Research Assistant.
"""

from core.langgraph_agent import WikipediaAgent
import logging
from core.logging_config import setup_logging
import sys
from typing import NoReturn
import os

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Print welcome message and instructions."""
    print("\n" + "="*70)
    print("🔍 Wikipedia Research Assistant")
    print("="*70)
    print("\nAsk me anything and I'll search Wikipedia to find relevant information!")
    print("\nType 'exit' or 'quit' to end the session, or 'clear' to clear the screen.")
    print("-"*70 + "\n")

def main() -> NoReturn:
    """Run the terminal interface."""
    # Configure logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize agent
    agent = WikipediaAgent()
    
    clear_screen()
    print_welcome()

    while True:
        try:
            # Use a custom prompt
            user_input = input("\033[1m\033[34m❯\033[0m ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using Wikipedia Research Assistant. Goodbye! 👋\n")
                sys.exit(0)
            
            if user_input.lower() == 'clear':
                clear_screen()
                print_welcome()
                continue
                
            if not user_input:
                continue

            # Show "thinking" indicator
            print("\n\033[90m🤔 Searching Wikipedia...\033[0m")
            
            response = agent.query(user_input)
            
            # # Print response with formatting
            # print("\n\033[92m💡 Answer:\033[0m")
            # print(f"{response}\n")
            # print("-"*70)

        except KeyboardInterrupt:
            print("\n\nExiting gracefully... 👋\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n\033[91m❌ Error: {str(e)}\033[0m\n")

if __name__ == "__main__":
    main()
