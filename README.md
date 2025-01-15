# Wikipedia Research Assistant

An AI-powered research assistant that uses Wikipedia's vast knowledge base to deliver comprehensive, well-researched answers to your questions.

Featuring a ReAct (Reasoning and Action) agent architecture that uses Wikipedia search via API, or RAG (retrieval augmented generation) enhanced by semantic chunking and embeddings.

Built on a modern stack including LangChain's LangGraph, Streamlit, FAISS vector storage, and support for leading LLM providers (OpenAI, Anthropic, Google). Offers flexible access through both terminal and web interfaces.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🤖 Multiple LLM Support from OpenAI, Anthropic, and Google, such as:
  - OpenAI: gpt-4o, chatgpt-4o-latest, o1-2024-12-17, gpt-4-turbo
  - Anthropic: claude-3-5-sonnet-latest
  - Google: gemini-exp-1206, gemini-1.5-pro, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-1219
- 🔍 Wikipedia search capabilities
- 📚 RAG (Retrieval Augmented Generation) support
- 💾 Conversation memory and thread management
- 🖥️ Both terminal and web interfaces
- 🔄 Streaming responses for real-time interaction
- 📝 Logging system

## Architecture

The project uses a modular architecture with:
- LangGraph's ReAct agent for agent orchestration
- LangChain for LLM interactions and tool integration
- FAISS for vector storage (when using RAG)
- Streamlit for the web interface

## Setup

1. Clone the repository:```bash
git clone https://github.com/yourusername/wikipedia-agent.git
cd wikipedia-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY="your_key_here" # if you want to use OpenAI
ANTHROPIC_API_KEY="your_key_here" # if you want to use Anthropic
GOOGLE_API_KEY="your_key_here" # if you want to use Google
LLM="claude-3-5-sonnet-latest"  # or other supported models
USE_RAG="false"  # set to true to enable RAG, is false by default
SYSTEM_PROMPT="your_prompt_here" # if you want to change the default prompt
```

## Usage

### Web Interface
```bash
python -m streamlit run app.py 
```
or
```bash
streamlit run app.py
```

Then visit `http://localhost:8501` in your browser.

### Terminal Interface
```bash
python app_terminal.py
```

## RAG Setup

To use RAG functionality instead of Wikipedia search:
1. Place wikipedia pages text files in the `rag_data/` directory (by default includes only artificial_intelligence.txt, physics.txt)
2. Set `USE_RAG=true` in your `.env` file

## Configuration

The agent can be configured through environment variables:
- `LLM`: Choose the LLM model (claude-3-5-sonnet-latest, gpt-4-turbo-preview, gemini-pro)
- `USE_RAG`: Enable/disable RAG functionality
- `SYSTEM_PROMPT`: Customize the agent's behavior

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## TODO

- [ ] Implement automatic Wikipedia scraping for RAG data collection
- [ ] Add GraphRAG support for improved knowledge retrieval
- [ ] Integrate text-to-speech capabilities
- [ ] Add support for more LLM providers (e.g., Mistral, Cohere)
- [ ] Create a Docker container for easy deployment
- [ ] Implement rate limiting and API usage tracking
- [ ] Add support for custom knowledge bases beyond Wikipedia
- [ ] Create a web API endpoint for programmatic access
- [ ] Implement caching to reduce API calls
- [ ] Add support for image and diagram generation
- [ ] Add benchmarking tools for different LLM models
- [ ] Add support for concurrent user sessions
- [ ] Create a configuration UI for easy setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
