# VerbalCodeAI 🤖

[![GitHub stars](https://img.shields.io/github/stars/vibheksoni/VerbalCodeAi?style=social)](https://github.com/vibheksoni/VerbalCodeAi/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/vibheksoni/VerbalCodeAi?style=social)](https://github.com/vibheksoni/VerbalCodeAi/network/members)
[![GitHub issues](https://img.shields.io/github/issues/vibheksoni/VerbalCodeAi)](https://github.com/vibheksoni/VerbalCodeAi/issues)
[![GitHub license](https://img.shields.io/github/license/vibheksoni/VerbalCodeAi)](https://github.com/vibheksoni/VerbalCodeAi/blob/master/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11.6-blue)](https://www.python.org/downloads/release/python-3116/)
[![Website](https://img.shields.io/badge/website-verbalcode.xyz-brightgreen)](https://verbalcode.xyz)

A powerful AI-powered code assistant that helps you understand, navigate, and work with codebases more efficiently. VerbalCodeAI uses advanced embedding techniques and LLM integration to provide intelligent code analysis and assistance directly in your terminal.


```yaml
.       .      .          .    .--.        .           .
 \     /       |          |   :            |          / \     o
  \   /.-. .--.|.-.  .-.  |   |    .-.  .-.| .-.     /___\    .
   \ /(.-' |   |   )(   ) |   :   (   )(   |(.-'    /     \   |
    '  `--''   '`-'  `-'`-`-   `--'`-'  `-'`-`--'  '       `-' `-'
                       [AI Assistant for Code]
```

<div align="center">
  <img src="Showcase/VerbalCodeShowcase.gif" alt="VerbalCodeAI Showcase" style="max-width: 800px; max-height: 400px; width: 100%; height: auto;"/>
</div>

## ✨ Features

- 🔍 **Intelligent Code Search**: Find relevant code snippets using natural language queries
- 📊 **Code Analysis**: Get insights about your codebase structure and dependencies
- 🤖 **Agent Mode**: Let the AI explore and understand your codebase using various tools
- 💬 **Chat with AI**: Ask questions about your code and get detailed explanations
- 🌐 **Web Search Integration**: Search the web for code-related information without leaving the terminal
- 🧠 **Memory System**: The AI remembers important information about your project
- 🔄 **Version Control Integration**: Analyze git history and changes
- 📝 **File Description**: Generate concise descriptions of code files
- 🛠️ **Command Runner**: Execute system commands with AI assistance

## 🚀 Getting Started

### Prerequisites

- Python 3.11.6 (tested and recommended version)
- [Ollama](https://ollama.com/download) (for local model execution)

> **Note:** The application has been specifically tested with Python 3.11.6 on Windows. While it may work with other versions, for the best experience, we recommend using the tested version.

### Installation

#### Windows

1. Clone the repository:
   ```
   git clone https://github.com/vibheksoni/VerbalCodeAi.git
   cd VerbalCodeAi
   ```

2. Run the setup script:
   ```
   setup_windows.bat
   ```

#### Linux/macOS

1. Clone the repository:
   ```
   git clone https://github.com/vibheksoni/VerbalCodeAi.git
   cd VerbalCodeAi
   ```

2. Make the setup script executable and run it:
   ```
   chmod +x setup_linux.sh
   ./setup_linux.sh
   ```

### Manual Installation

If you prefer to set up manually:

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration (see `.env.example` for reference)

## 🖥️ Usage

### Starting the Application

After installation, activate your virtual environment and run:

```
python app.py
```

### Indexing a Project

When you first start VerbalCodeAI, you'll be prompted to select a directory to index. This process analyzes your codebase and creates embeddings for efficient searching.

### Main Menu Options

- **Chat with AI**: Ask questions about your code
- **Agent Mode**: Use AI with tools to explore your codebase
- **Reindex Project**: Update the index after code changes
- **Project Info**: View information about the indexed project
- **Settings**: Configure application settings
- **Exit**: Quit the application

### Agent Mode Tools

Agent Mode provides access to powerful tools:

- **Search tools**: `embed_search`, `semantic_search`, `grep`, `regex_advanced_search`, `file_type_search`
- **File tools**: `read_file`, `file_stats`, `directory_tree`, `get_file_description`, `get_file_metadata`
- **Code analysis**: `find_functions`, `find_classes`, `find_usage`, `cross_reference`, `code_analysis`, `get_functions`, `get_classes`, `get_variables`, `get_imports`, `explain_code`
- **Version control**: `git_history`, `version_control_search`, `search_imports`
- **Project tools**: `get_project_description`, `get_instructions`, `create_instructions_template`
- **Memory tools**: `add_memory`, `get_memories`, `search_memories`
- **System tools**: `run_command`, `read_terminal`, `kill_terminal`, `list_terminals`
- **Helper tools**: `ask_buddy` (with context-aware second opinions)
- **Web tools**: `google_search`, `ddg_search`, `bing_news_search`, `fetch_webpage`, `get_base_knowledge`

> **💡 Pro Tip**: Agent Mode is the most cost-effective option when using cloud-based LLM providers. It makes fewer API calls compared to Chat Mode, which helps avoid rate limits and reduces costs. For the best experience with minimal expenses, consider using Agent Mode when working with paid API services.

## ⚙️ Configuration

VerbalCodeAI can be configured through the `.env` file:

```
# Provider can be: ollama, google, openai, anthropic, groq, or openrouter
AI_CHAT_PROVIDER=ollama
AI_EMBEDDING_PROVIDER=ollama
AI_DESCRIPTION_PROVIDER=ollama
AI_AGENT_BUDDY_PROVIDER=ollama

# API Keys for each functionality (only needed if using that provider)
# The same key will be used for the selected provider in each category
AI_CHAT_API_KEY=None
AI_EMBEDDING_API_KEY=None
AI_DESCRIPTION_API_KEY=None
AI_AGENT_BUDDY_API_KEY=None

# Model names for each provider
# For ollama: llama2, codellama, mistral, etc. (embedding)
# For OpenAI: gpt-4, gpt-3.5-turbo, text-embedding-ada-002 (embedding)
# For OpenRouter: anthropic/claude-3-opus, openai/gpt-4-turbo, google/gemini-pro, etc.
# For Google: gemini-pro, gemini-pro-vision
# For Anthropic: claude-3-5-sonnet-latest, claude-3-opus-20240229, claude-3-haiku-20240307
# For Groq: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
CHAT_MODEL=llama2
EMBEDDING_MODEL=all-minilm:33m
DESCRIPTION_MODEL=llama2
AI_AGENT_BUDDY_MODEL=llama3.2

# Optional: Site information for OpenRouter rankings
SITE_URL=http://localhost:3000
SITE_NAME=Local Development

# Performance settings (LOW, MEDIUM, MAX)
# LOW: Minimal resource usage, suitable for low-end systems
# MEDIUM: Balanced resource usage, suitable for most systems
# MAX: Maximum resource usage, suitable for high-end systems
PERFORMANCE_MODE=MEDIUM
# Maximum number of threads to use (will be calculated automatically if not set)
MAX_THREADS=16
# Cache size for embedding queries (higher values use more memory but improve performance)
EMBEDDING_CACHE_SIZE=1000
# Similarity threshold for embedding search (lower values return more results but may be less relevant)
EMBEDDING_SIMILARITY_THRESHOLD=0.05

# Maximum number of threads to use (will be calculated automatically if not set)
# MAX_THREADS=16

# UI Settings
# Enable/disable markdown rendering (TRUE/FALSE)
ENABLE_MARKDOWN_RENDERING=TRUE
# Show thinking blocks in AI responses (TRUE/FALSE)
SHOW_THINKING_BLOCKS=FALSE
# Enable streaming mode for AI responses (TRUE/FALSE) # Tends to be slower for some reason # Broken for openrouter TODO: Fix this at some point !
ENABLE_STREAMING_MODE=FALSE
# Enable chat logging to save conversations (TRUE/FALSE)
CHAT_LOGS=FALSE
# Enable memory for AI conversations (TRUE/FALSE)
MEMORY_ENABLED=TRUE
# Maximum number of memory items to store
MAX_MEMORY_ITEMS=10
# Execute commands without confirmation (TRUE/FALSE)
# When FALSE, the user will be prompted to confirm before executing any command
# When TRUE, commands will execute automatically without confirmation
COMMANDS_YOLO=FALSE
```

### Supported LLM Providers

- **Ollama**: Local models (default)
- **Google AI**: Cloud-based models (requires API key)
- **OpenAI**: OpenAI models for both chat and embeddings (requires API key)
- **Anthropic**: Claude models for chat with streaming support (requires API key)
- **Groq**: High-performance LLMs with extremely low latency (requires API key)
- **OpenRouter**: Various cloud models (requires API key)

#### Recommended Ollama Setup

For the best local experience without any API costs, the developer recommends using these Ollama models:
- **Chat/Description**: `gemma3` - Google's Gemma 3 model provides excellent code understanding and generation
- **Embeddings**: `all-minilm` - Efficient and accurate embeddings for code search and retrieval

To use these models, make sure they're installed in Ollama:
```bash
ollama pull gemma3
ollama pull all-minilm
```

Then configure your `.env` file:
```
AI_CHAT_PROVIDER=ollama
AI_EMBEDDING_PROVIDER=ollama
AI_DESCRIPTION_PROVIDER=ollama
CHAT_MODEL=gemma3
EMBEDDING_MODEL=all-minilm:33m
DESCRIPTION_MODEL=gemma3
```

#### Anthropic Claude Models

Anthropic's Claude models are particularly strong at understanding and generating code. Available models include:
- **claude-3-5-sonnet-latest**: Latest version of Claude 3.5 Sonnet, excellent balance of performance and speed
- **claude-3-opus-20240229**: Most powerful Claude model with advanced reasoning capabilities
- **claude-3-haiku-20240307**: Fastest and most cost-effective Claude model

Note: Anthropic does not provide embedding capabilities, so you'll need to use a different provider for embeddings.

#### Groq Models

Groq provides ultra-fast inference for popular open-source models. Available models include:
- **llama3-8b-8192**: Smaller Llama 3 model with 8B parameters, good balance of performance and speed
- **llama3-70b-8192**: Larger Llama 3 model with 70B parameters, excellent reasoning capabilities
- **mixtral-8x7b-32768**: Mixtral model with 8x7B parameters and 32k context window

Note: Groq does not provide embedding capabilities, so you'll need to use a different provider for embeddings.

### Performance Settings

- **LOW**: Minimal resource usage, suitable for low-end systems
- **MEDIUM**: Balanced resource usage, suitable for most systems
- **MAX**: Maximum resource usage, suitable for high-end systems

### UI Settings

- **ENABLE_MARKDOWN_RENDERING**: Enable/disable markdown rendering in responses
- **SHOW_THINKING_BLOCKS**: Show AI's thinking process in responses
- **ENABLE_STREAMING_MODE**: Enable streaming responses (may be slower)
- **CHAT_LOGS**: Save conversation logs to disk
- **MEMORY_ENABLED**: Enable AI memory for conversations
- **MAX_MEMORY_ITEMS**: Maximum number of memory items to store
- **COMMANDS_YOLO**: When FALSE, prompts for confirmation before executing commands



## 📺 Video Showcase

Check out a quick video demonstration of VerbalCodeAI in action:

<p align="center">
  <a href="https://youtu.be/5EnenUD-4E4?si=WIw-C4ne4JCP5dfZ" target="_blank">
    <img src="https://img.youtube.com/vi/5EnenUD-4E4/hqdefault.jpg" alt="VerbalCodeAI Video Showcase" width="480"/>
  </a>
</p>

Or watch directly on [YouTube](https://youtu.be/5EnenUD-4E4?si=WIw-C4ne4JCP5dfZ).

---

## 🖼️ Image Showcase

Below are some screenshots and showcase images of VerbalCodeAI in action:

<div align="center">
  <img src="Showcase/Agent%20Mode%20New%20commands.PNG" alt="Agent Mode New commands" width="350" style="margin: 10px;"/>
  <img src="Showcase/Agent%20Showcase.png" alt="Agent Showcase" width="350" style="margin: 10px;"/>
  <img src="Showcase/First%20Implementation%20Chat%20With%20Ai.png" alt="First Implementation Chat With Ai" width="350" style="margin: 10px;"/>
  <img src="Showcase/Main%20Menu%20Showcase.PNG" alt="Main Menu Showcase" width="350" style="margin: 10px;"/>
  <img src="Showcase/Second%20Agent.PNG" alt="Second Agent" width="350" style="margin: 10px;"/>
  <img src="Showcase/Second%20Implementation%20Chat%20With%20Ai.PNG" alt="Second Implementation Chat With Ai" width="350" style="margin: 10px;"/>
</div>

---

## 🧩 Project Structure

```
VerbalCodeAi/
├── app.py                  # Main application entry point
├── mods/                   # Core modules
│   ├── banners.py          # ASCII art banners
│   ├── llms.py             # LLM integration
│   ├── terminal_ui.py      # Terminal UI components
│   ├── terminal_utils.py   # Terminal utilities
│   └── code/               # Code processing modules
│       ├── agent_mode.py   # Agent mode implementation
│       ├── decisions.py    # AI decision making
│       ├── directory.py    # Directory structure handling
│       ├── embed.py        # Embedding generation and search
│       ├── indexer.py      # File indexing
│       ├── memory.py       # Memory management
│       ├── terminal.py     # Terminal command execution
│       └── tools.py        # Agent tools
├── requirements.txt        # Python dependencies
├── setup_windows.bat       # Windows setup script
└── setup_linux.sh          # Linux setup script
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Website](https://verbalcode.xyz)
- [GitHub Repository](https://github.com/vibheksoni/VerbalCodeAi)
- [Report Issues](https://github.com/vibheksoni/VerbalCodeAi/issues)

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/) for local model execution
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing
- All the open-source libraries that make this project possible

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/vibheksoni">Vibhek Soni</a>
</p>
