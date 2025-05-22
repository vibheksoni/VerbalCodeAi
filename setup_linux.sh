#!/bin/sh

# Check if we're running in a terminal that supports colors
if [ -t 1 ]; then
    # ANSI color codes
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    # No colors if not in a terminal
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

printf "${CYAN}"
echo "========================================================"
echo " _    __          __          __   ______          __        ___    _"
echo "| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)"
echo "| | / / _ \/ ___/ __ \/ __ \`/ /  / /   / __ \/ __  / _ \   / /| | / /"
echo "| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /"
echo "|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/"
echo ""
echo "========================================================"
echo "Linux Setup Script for VerbalCodeAI"
echo "========================================================"
printf "${NC}"

# Check if Python is installed
python3 --version &> /dev/null
if [ $? -ne 0 ]; then
    printf "${RED}[ERROR] Python 3 is not installed.${NC}\n"
    echo "Please install Python 3.11.6 or later using your package manager."
    echo "For example: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
printf "${BLUE}[INFO] Detected Python version: ${PYTHON_VERSION}${NC}\n"

# Verify Python version is 3.11.6 or later
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
PATCH=$(echo $PYTHON_VERSION | cut -d. -f3)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
    printf "${YELLOW}[WARNING] Python version $PYTHON_VERSION may be too old.${NC}\n"
    echo "This application was tested with Python 3.11.6."
    echo "You may encounter issues with older versions."
    echo ""
    read -p "Do you want to continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
printf "${YELLOW}[STEP 1] Creating virtual environment...${NC}\n"
if [ -d "venv" ]; then
    printf "${BLUE}[INFO] Virtual environment already exists. Skipping creation.${NC}\n"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        printf "${RED}[ERROR] Failed to create virtual environment.${NC}\n"
        echo "Make sure python3-venv is installed."
        echo "For example: sudo apt install python3-venv"
        exit 1
    fi
    printf "${GREEN}[SUCCESS] Virtual environment created.${NC}\n"
fi

# Activate virtual environment
echo ""
printf "${YELLOW}[STEP 2] Activating virtual environment...${NC}\n"
if [ -f "venv/bin/activate" ]; then
    # Use . instead of source for better compatibility
    . venv/bin/activate
    if [ $? -ne 0 ]; then
        printf "${RED}[ERROR] Failed to activate virtual environment.${NC}\n"
        exit 1
    fi
    printf "${GREEN}[SUCCESS] Virtual environment activated.${NC}\n"
else
    printf "${RED}[ERROR] Virtual environment activation script not found.${NC}\n"
    exit 1
fi

# Upgrade pip
echo ""
printf "${YELLOW}[STEP 3] Upgrading pip...${NC}\n"
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    printf "${YELLOW}[WARNING] Failed to upgrade pip, but continuing with installation.${NC}\n"
fi

# Install dependencies
echo ""
printf "${YELLOW}[STEP 4] Installing dependencies...${NC}\n"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    printf "${RED}[ERROR] Failed to install dependencies.${NC}\n"
    exit 1
fi
printf "${GREEN}[SUCCESS] Dependencies installed.${NC}\n"

# Check if Ollama is installed
echo ""
printf "${YELLOW}[STEP 5] Checking for Ollama...${NC}\n"
if ! command -v ollama &> /dev/null; then
    printf "${YELLOW}[WARNING] Ollama is not installed or not in PATH.${NC}\n"
    echo "You will need to install Ollama to use local models."
    echo "Installation instructions: https://ollama.com/download"
    echo ""
    read -p "Would you like to install Ollama now? (y/n): " INSTALL_OLLAMA
    if [ "$INSTALL_OLLAMA" = "y" ] || [ "$INSTALL_OLLAMA" = "Y" ]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if [ $? -ne 0 ]; then
            printf "${YELLOW}[WARNING] Failed to install Ollama automatically.${NC}\n"
            echo "Please install manually from: https://ollama.com/download"
        else
            printf "${GREEN}[SUCCESS] Ollama installed.${NC}\n"
        fi
    fi
else
    printf "${GREEN}[SUCCESS] Ollama is installed.${NC}\n"
fi

# Create .env file if it doesn't exist
echo ""
printf "${YELLOW}[STEP 6] Setting up environment variables...${NC}\n"
if [ ! -f ".env" ]; then
    printf "${BLUE}[INFO] Creating .env file with default settings...${NC}\n"
    cat > .env << EOL
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
EOL
    printf "${GREEN}[SUCCESS] Created .env file with default settings.${NC}\n"
else
    printf "${BLUE}[INFO] .env file already exists. Skipping creation.${NC}\n"
fi

# Make the app.py executable
chmod +x app.py

echo ""
printf "${CYAN}========================================================\n"
echo "VerbalCodeAI setup completed successfully!"
echo ""
echo "To start the application, run:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "For more information, see the README.md file."
printf "========================================================${NC}\n"
echo ""

# Offer to run the application
read -p "Would you like to run VerbalCodeAI now? (y/n): " RUN_APP
if [ "$RUN_APP" = "y" ] || [ "$RUN_APP" = "Y" ]; then
    echo ""
    echo "Starting VerbalCodeAI..."
    python app.py
fi
