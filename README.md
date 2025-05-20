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

## ⚙️ Configuration

VerbalCodeAI can be configured through the `.env` file:

```
# Provider can be: ollama, google, openai, or openrouter
AI_CHAT_PROVIDER=ollama
AI_EMBEDDING_PROVIDER=ollama
AI_DESCRIPTION_PROVIDER=ollama

# API Keys for each provider (only needed if using that provider)
AI_CHAT_API_KEY=None
AI_EMBEDDING_API_KEY=None
AI_DESCRIPTION_API_KEY=None

# Model names for each provider
# For ollama: llama2, codellama, mistral, etc.
# For OpenAI: gpt-4, gpt-3.5-turbo, text-embedding-ada-002
# For OpenRouter: anthropic/claude-3-opus, openai/gpt-4-turbo, google/gemini-pro, etc.
# For Google: gemini-pro, gemini-pro-vision
CHAT_MODEL=llama2
EMBEDDING_MODEL=all-minilm:33m
DESCRIPTION_MODEL=llama2

# Performance settings (LOW, MEDIUM, MAX)
PERFORMANCE_MODE=MEDIUM

# UI Settings
ENABLE_MARKDOWN_RENDERING=TRUE
SHOW_THINKING_BLOCKS=FALSE
ENABLE_STREAMING_MODE=FALSE
CHAT_LOGS=FALSE
MEMORY_ENABLED=TRUE
MAX_MEMORY_ITEMS=10
COMMANDS_YOLO=FALSE
```

### Supported LLM Providers

- **Ollama**: Local models (default)
- **Google AI**: Cloud-based models (requires API key)
- **OpenAI**: OpenAI models (requires API key)
- **OpenRouter**: Various cloud models (requires API key)

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
