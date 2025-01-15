import streamlit as st
from core.langgraph_agent import WikipediaAgent
import uuid
import logging
from core.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Wikipedia Research Assistant", page_icon="📚")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    logger.info("Initializing messages in session state")
    st.session_state.messages = []

if "agent" not in st.session_state:
    logger.info("Initializing WikipediaAgent")
    st.session_state.agent = WikipediaAgent()

if "thread_id" not in st.session_state:
    logger.info("Generating new thread ID")
    st.session_state.thread_id = str(uuid.uuid4())

st.title("📚 Wikipedia Research Assistant")
st.write("Ask me anything and I'll search Wikipedia to find relevant information!")

# Display chat messages from history
logger.debug(f"Displaying {len(st.session_state.messages)} messages from history")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    logger.info(f"Received user input: {prompt}")
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    logger.debug("Added user message to chat history")
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching Wikipedia..."):
            logger.info("Querying WikipediaAgent")
            response = st.session_state.agent.query(prompt, thread_id=st.session_state.thread_id)
            logger.debug(f"Received response: {response}")
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    logger.debug("Added assistant response to chat history")