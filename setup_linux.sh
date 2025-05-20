#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
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
echo -e "${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 is not installed.${NC}"
    echo "Please install Python 3.11.6 or later using your package manager."
    echo "For example: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${BLUE}[INFO] Detected Python version: ${PYTHON_VERSION}${NC}"

# Verify Python version is 3.11.6 or later
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
PATCH=$(echo $PYTHON_VERSION | cut -d. -f3)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
    echo -e "${YELLOW}[WARNING] Python version $PYTHON_VERSION may be too old.${NC}"
    echo "This application was tested with Python 3.11.6."
    echo "You may encounter issues with older versions."
    echo ""
    read -p "Do you want to continue anyway? (y/n): " CONTINUE
    if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo -e "${YELLOW}[STEP 1] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${BLUE}[INFO] Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment.${NC}"
        echo "Make sure python3-venv is installed."
        echo "For example: sudo apt install python3-venv"
        exit 1
    fi
    echo -e "${GREEN}[SUCCESS] Virtual environment created.${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[STEP 2] Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}[SUCCESS] Virtual environment activated.${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}[STEP 3] Upgrading pip...${NC}"
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[WARNING] Failed to upgrade pip, but continuing with installation.${NC}"
fi

# Install dependencies
echo ""
echo -e "${YELLOW}[STEP 4] Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to install dependencies.${NC}"
    exit 1
fi
echo -e "${GREEN}[SUCCESS] Dependencies installed.${NC}"

# Check if Ollama is installed
echo ""
echo -e "${YELLOW}[STEP 5] Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}[WARNING] Ollama is not installed or not in PATH.${NC}"
    echo "You will need to install Ollama to use local models."
    echo "Installation instructions: https://ollama.com/download"
    echo ""
    read -p "Would you like to install Ollama now? (y/n): " INSTALL_OLLAMA
    if [[ $INSTALL_OLLAMA == "y" || $INSTALL_OLLAMA == "Y" ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}[WARNING] Failed to install Ollama automatically.${NC}"
            echo "Please install manually from: https://ollama.com/download"
        else
            echo -e "${GREEN}[SUCCESS] Ollama installed.${NC}"
        fi
    fi
else
    echo -e "${GREEN}[SUCCESS] Ollama is installed.${NC}"
fi

# Create .env file if it doesn't exist
echo ""
echo -e "${YELLOW}[STEP 6] Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${BLUE}[INFO] Creating .env file with default settings...${NC}"
    cat > .env << EOL
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

# Optional: Site information for OpenRouter rankings
SITE_URL=http://localhost:3000
SITE_NAME=Local Development

# Performance settings (LOW, MEDIUM, MAX)
# LOW: Minimal resource usage, suitable for low-end systems
# MEDIUM: Balanced resource usage, suitable for most systems
# MAX: Maximum resource usage, suitable for high-end systems
PERFORMANCE_MODE=MEDIUM

# Maximum number of threads to use (will be calculated automatically if not set)
# MAX_THREADS=16

# UI Settings
# Enable/disable markdown rendering (TRUE/FALSE)
ENABLE_MARKDOWN_RENDERING=TRUE
# Show thinking blocks in AI responses (TRUE/FALSE)
SHOW_THINKING_BLOCKS=FALSE
# Enable streaming mode for AI responses (TRUE/FALSE)
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
EOL
    echo -e "${GREEN}[SUCCESS] Created .env file with default settings.${NC}"
else
    echo -e "${BLUE}[INFO] .env file already exists. Skipping creation.${NC}"
fi

# Make the app.py executable
chmod +x app.py

echo ""
echo -e "${CYAN}========================================================"
echo "VerbalCodeAI setup completed successfully!"
echo ""
echo "To start the application, run:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "For more information, see the README.md file."
echo "========================================================${NC}"
echo ""

# Offer to run the application
read -p "Would you like to run VerbalCodeAI now? (y/n): " RUN_APP
if [[ $RUN_APP == "y" || $RUN_APP == "Y" ]]; then
    echo ""
    echo "Starting VerbalCodeAI..."
    python app.py
fi
