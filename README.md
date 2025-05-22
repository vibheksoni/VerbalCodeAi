# VerbalCodeAI 🤖

[![GitHub stars](https://img.shields.io/github/stars/vibheksoni/VerbalCodeAi?style=social)](https://github.com/vibheksoni/VerbalCodeAi/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/vibheksoni/VerbalCodeAi?style=social)](https://github.com/vibheksoni/VerbalCodeAi/network/members)
[![GitHub issues](https://img.shields.io/github/issues/vibheksoni/VerbalCodeAi)](https://github.com/vibheksoni/VerbalCodeAi/issues)
[![GitHub license](https://img.shields.io/github/license/vibheksoni/VerbalCodeAi)](https://github.com/vibheksoni/VerbalCodeAi/blob/master/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11.6-blue)](https://www.python.org/downloads/release/python-3116/)
[![Website](https://img.shields.io/badge/website-verbalcode.xyz-brightgreen)](https://verbalcode.xyz)
<a href="https://discord.gg/KpjSDEwWCF" target="_blank">
  <img src="https://img.shields.io/static/v1?label=Discord&message=Join%20Us&color=7289DA&logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord"/>
</a>

Join our Discord server to hang out, ask questions, or offer help! We're currently looking for assistance in setting up the server, so if you're experienced with Discord setup, your help would be greatly appreciated! Click [here](https://discord.gg/KpjSDEwWCF) to join.

Ever felt lost in a complex codebase? Wish you had a smart assistant to help you navigate and understand code right in your terminal? VerbalCodeAI is here to help! This powerful AI-powered tool acts as your personal code companion, leveraging advanced embedding techniques and Large Language Model (LLM) integration. It offers intelligent code analysis, helps you search and understand your project, and provides assistance directly within your command-line interface, making your development workflow smoother and more efficient.

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


<details>
<summary><strong>Table of Contents</strong></summary>

- [VerbalCodeAI 🤖](#verbalcodeai-)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
      - [Windows](#windows)
      - [Linux/macOS](#linuxmacos)
    - [Manual Installation](#manual-installation)
  - [Usage](#usage)
    - [Starting the Application](#starting-the-application)
    - [Indexing a Project](#indexing-a-project)
    - [Main Menu Options](#main-menu-options)
    - [Agent Mode Tools](#agent-mode-tools)
    - [HTTP API Server](#http-api-server)
    - [MCP Integration](#mcp-integration)
      - [Setting Up the MCP Server](#setting-up-the-mcp-server)
      - [Using with Claude Desktop](#using-with-claude-desktop)
      - [Using with Cursor](#using-with-cursor)
  - [Configuration](#configuration)
    - [Supported LLM Providers](#supported-llm-providers)
      - [Recommended Ollama Setup](#recommended-ollama-setup)
      - [Anthropic Claude Models](#anthropic-claude-models)
      - [Groq Models](#groq-models)
    - [Performance Settings](#performance-settings)
    - [UI Settings](#ui-settings)
  - [Video Showcase](#video-showcase)
  - [Image Showcase](#image-showcase)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Links](#links)
  - [Acknowledgements](#acknowledgements)

</details>

- 🔍 **Intelligent Code Search**: Find relevant code snippets using natural language queries
- 📊 **Code Analysis**: Get insights about your codebase structure and dependencies
- 🤖 **Agent Mode**: Let the AI explore and understand your codebase using various tools
- 💬 **Chat with AI**: Ask questions about your code and get detailed explanations
- 🌐 **Web Search Integration**: Search the web for code-related information without leaving the terminal
- 🧠 **Memory System**: The AI remembers important information about your project
- 🔄 **Version Control Integration**: Analyze git history and changes
- 📝 **File Description**: Generate concise descriptions of code files
- 🛠️ **Command Runner**: Execute system commands with AI assistance

## Getting Started

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

## Usage

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

### HTTP API Server

VerbalCodeAI includes a built-in HTTP API server that allows you to access its functionality programmatically. This is useful for integrating VerbalCodeAI with other tools or creating custom interfaces.

To start the HTTP API server:

```bash
python app.py --serve [PORT]
```

Where `[PORT]` is the port number you want the server to listen on (default is 8000).

The server provides the following endpoints:

- `GET /api/health` - Health check
- `POST /api/initialize` - Initialize a directory
- `POST /api/ask` - Ask the agent a question
- `POST /api/index/start` - Start indexing a directory
- `GET /api/index/status` - Get indexing status

Example usage with curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Initialize a directory
curl -X POST http://localhost:8000/api/initialize -H "Content-Type: application/json" -d '{"directory_path": "D:/path/to/your/project"}'

# Ask a question
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{"question": "What does this codebase do?"}'

# Start indexing
curl -X POST http://localhost:8000/api/index/start -H "Content-Type: application/json" -d '{"directory_path": "D:/path/to/your/project"}'

# Get indexing status
curl http://localhost:8000/api/index/status
```

By default, the server only accepts connections from localhost (127.0.0.1). To allow connections from any IP address, set the `HTTP_ALLOW_ALL_ORIGINS` environment variable to `TRUE` in your `.env` file.

### MCP Integration

VerbalCodeAI supports the Model Context Protocol (MCP), allowing you to connect it to Claude Desktop and other MCP-compatible AI assistants. This integration enables Claude to directly interact with your codebase, providing a powerful AI-assisted development experience.

<div align="center">
  <img src="Showcase/MCP Showcase.PNG" alt="MCP Integration with Claude Desktop" style="width: 500px; height: 500px; max-width: 100%; object-fit: contain; margin: 10px;"/>
  <img src="Showcase/MCP Showcase 2.PNG" alt="MCP Showcase - Example 2" style="width: 500px; height: 500px; max-width: 100%; object-fit: contain; margin: 10px;"/>
</div>

#### Setting Up the MCP Server

The MCP server wraps the HTTP API server and provides tools for Claude to interact with VerbalCodeAI. Here's how to set it up:

1. **Start the HTTP API Server**:
   First, start the HTTP API server if it's not already running:

   ```bash
   python app.py --serve 8000
   ```

   You should see output confirming the server is running:

   ```
   Starting HTTP API server on 127.0.0.1:8000
   Available endpoints:
   - GET  /api/health - Health check
   - POST /api/initialize - Initialize a directory
   - POST /api/ask - Ask the agent a question
   - POST /api/index/start - Start indexing a directory
   - GET  /api/index/status - Get indexing status
   Server is only accessible from localhost.
   ```

2. **Start the MCP Server**:
   In a new terminal window, start the MCP server:

   ```bash
   python mcp_server.py
   ```

   The MCP server will automatically check if the HTTP API server is running and start it if needed.

3. **Configure the MCP Server** (Optional):
   You can configure the MCP server by setting the following environment variables in your `.env` file:

   ```
   # MCP Server Settings
   MCP_API_URL=http://localhost:8000
   MCP_HTTP_PORT=8000
   ```

#### Using with Claude Desktop

To use VerbalCodeAI with Claude Desktop:

1. **Install the MCP SDK**:
   ```bash
   pip install mcp
   ```

2. **Install the MCP Server in Claude Desktop**:

   **Method 1: Using the command line**
   ```bash
   mcp install mcp_server.py
   ```

   **Method 2: Using JSON configuration**

   You can also configure Claude Desktop to use VerbalCodeAI by adding an entry to the MCP servers configuration file. This is especially useful if you're using a virtual environment or need custom configuration.

   1. Locate your Claude Desktop configuration directory:
      - Windows: `%APPDATA%\Claude Desktop\User Data\Default\`
      - macOS: `~/Library/Application Support/Claude Desktop/User Data/Default/`
      - Linux: `~/.config/Claude Desktop/User Data/Default/`

   2. Create or edit the `mcp_servers.json` file in this directory:
      ```json
      {
        "mcpServers": {
          "VerbalCodeAI": {
            "command": "YOUR_VENV_PATH\\Scripts\\uv.EXE",
            "args": [
              "run",
              "--with",
              "mcp[cli]",
              "--with",
              "python-dotenv",
              "--with",
              "requests",
              "mcp",
              "run",
              "PATH_TO_VERBALCODEAI\\mcp_server.py"
            ]
          }
        }
      }
      ```

      Replace:
      - `YOUR_VENV_PATH` with the path to your Python virtual environment (e.g., `D:\\VerbalCodeAi\\.venv`)
      - `PATH_TO_VERBALCODEAI` with the path to your VerbalCodeAI installation (e.g., `D:\\VerbalCodeAi`)

      Example for a Windows installation:
      ```json
      {
        "mcpServers": {
          "VerbalCodeAI": {
            "command": "D:\\VerbalCodeAi\\.venv\\Scripts\\uv.EXE",
            "args": [
              "run",
              "--with",
              "mcp[cli]",
              "--with",
              "python-dotenv",
              "--with",
              "requests",
              "mcp",
              "run",
              "D:\\VerbalCodeAi\\mcp_server.py"
            ]
          }
        }
      }
      ```

   3. Restart Claude Desktop for the changes to take effect.

3. **Open Claude Desktop**:
   - Launch Claude Desktop
   - Click on the "+" button to start a new conversation
   - Select "VerbalCodeAI" from the list of available tools

4. **Initialize a Project**:
   In Claude Desktop, you can now use the following tools:
   - `initialize_directory(directory_path)` - Initialize a directory for use with VerbalCodeAI
   - `ask_agent(question)` - Ask the agent a question about the codebase
   - `start_indexing(directory_path)` - Start indexing a directory
   - `get_indexing_status()` - Get the status of the indexing process
   - `health_check()` - Check if the HTTP API server is running
   - `start_http_server_tool(port)` - Start the HTTP API server if it's not running
   - `set_api_url(url)` - Set the API URL for the HTTP API server

   Example conversation:

   ```
   User: Can you help me analyze my project?

   Claude: I'd be happy to help you analyze your project using VerbalCodeAI. First, let's check if the HTTP API server is running.

   [Tool: health_check]
   Result: {"status": "ok", "message": "HTTP API server is running"}

   Great! Now let's initialize your project. What's the path to your project directory?

   User: D:/my-project

   Claude: I'll initialize your project now.

   [Tool: initialize_directory]
   Input: {"directory_path": "D:/my-project"}
   Result: {"status": "ok", "message": "Directory initialized successfully"}

   Now I'll start indexing your project to analyze the codebase.

   [Tool: start_indexing]
   Input: {"directory_path": "D:/my-project"}
   Result: {"status": "ok", "message": "Indexing started"}

   Let's check the indexing status.

   [Tool: get_indexing_status]
   Result: {"status": "in_progress", "progress": 45, "message": "Indexing in progress"}

   The indexing is currently at 45%. Once it's complete, I can help you analyze your codebase. What would you like to know about your project?
   ```

#### Using with Cursor

[Cursor](https://cursor.sh/) is an AI-powered code editor that supports MCP. To use VerbalCodeAI with Cursor:

1. **Install Cursor** if you haven't already from [cursor.sh](https://cursor.sh/)

2. **Start the MCP Server**:
   ```bash
   python mcp_server.py
   ```

3. **Connect Cursor to the MCP Server**:
   - Open Cursor
   - Go to Settings > AI > Model Context Protocol
   - Add a new MCP server with the URL `http://localhost:3000` (or the port your MCP server is running on)
   - Select "VerbalCodeAI" from the list of available tools

4. **Use VerbalCodeAI in Cursor**:
   - Open your project in Cursor
   - Use the AI chat to interact with VerbalCodeAI
   - You can ask questions about your codebase, get explanations, and more

## Configuration

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

# Model Tempratures
CHAT_MODEL_TEMPERATURE=0.7
DESCRIPTION_MODEL_TEMPERATURE=0.3
AI_AGENT_BUDDY_MODEL_TEMPERATURE=0.7
INTENT_DETECTION_TEMPERATURE=0.1

# Model Max Tokens
CHAT_MODEL_MAX_TOKENS=4096
DESCRIPTION_MODEL_MAX_TOKENS=4096
AI_AGENT_BUDDY_MODEL_MAX_TOKENS=4096
INTENT_DETECTION_MAX_TOKENS=4096

# Other Model Settings
CHAT_MODEL_TOP_P=0.95
CHAT_MODEL_TOP_K=40
DESCRIPTION_MODEL_TOP_P=0.95
DESCRIPTION_MODEL_TOP_K=40
INTENT_DETECTION_TOP_P=0.95
INTENT_DETECTION_TOP_K=40

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

# API Rate Limiting Settings
# Delay in milliseconds between embedding API calls to prevent rate limiting
# Recommended: 100ms for Google, 0ms for OpenAI/Ollama (set to 0 to disable)
EMBEDDING_API_DELAY_MS=100
# Delay in milliseconds between description generation API calls to prevent rate limiting
# Recommended: 100ms for Google, 0ms for OpenAI/Ollama (set to 0 to disable)
DESCRIPTION_API_DELAY_MS=100

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

# HTTP API Server Settings
# Allow connections from any IP address (TRUE/FALSE)
# When FALSE, the server only accepts connections from localhost (127.0.0.1)
# When TRUE, the server accepts connections from any IP address (0.0.0.0)
# WARNING: Setting this to TRUE may expose your API to the internet
HTTP_ALLOW_ALL_ORIGINS=FALSE

# MCP Server Settings
# URL of the HTTP API server
MCP_API_URL=http://localhost:8000
# Port to run the HTTP API server on
MCP_HTTP_PORT=8000
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



## Video Showcase

Check out a quick video demonstration of VerbalCodeAI in action:

<p align="center">
  <a href="https://youtu.be/5EnenUD-4E4?si=WIw-C4ne4JCP5dfZ" target="_blank">
    <img src="https://img.youtube.com/vi/5EnenUD-4E4/hqdefault.jpg" alt="VerbalCodeAI Video Showcase" width="480"/>
  </a>
</p>

Or watch directly on [YouTube](https://youtu.be/5EnenUD-4E4?si=WIw-C4ne4JCP5dfZ).

---

## Image Showcase

Below are some screenshots and showcase images of VerbalCodeAI in action:

<div align="center">
  <img src="Showcase/Agent Mode New commands.PNG" alt="Agent Mode New commands" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
  <img src="Showcase/Agent Showcase.png" alt="Agent Showcase" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
  <img src="Showcase/First Implementation Chat With Ai.png" alt="First Implementation Chat With Ai" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
  <img src="Showcase/Main Menu Showcase.PNG" alt="Main Menu Showcase" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
  <img src="Showcase/Second Agent.PNG" alt="Second Agent" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
  <img src="Showcase/Second Implementation Chat With Ai.PNG" alt="Second Implementation Chat With Ai" style="width: 350px; max-width: 100%; height: auto; margin: 10px;"/>
</div>

---

## Project Structure

```
VerbalCodeAi/
├── app.py                  # Main application entry point
├── mcp_server.py           # MCP server wrapper
├── mcp_server_http.py      # HTTP-based MCP server implementation
├── mods/                   # Core modules
│   ├── banners.py          # ASCII art banners
│   ├── http_api.py         # HTTP API server implementation
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- [Website](https://verbalcode.xyz)
- [GitHub Repository](https://github.com/vibheksoni/VerbalCodeAi)
- [Report Issues](https://github.com/vibheksoni/VerbalCodeAi/issues)

## Acknowledgements

- [Ollama](https://ollama.com/) for local model execution
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing
- All the open-source libraries that make this project possible

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/vibheksoni">Vibhek Soni</a>
</p>