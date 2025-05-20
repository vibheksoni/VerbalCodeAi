#!/bin/bash

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
    if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
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
source venv/bin/activate
if [ $? -ne 0 ]; then
    printf "${RED}[ERROR] Failed to activate virtual environment.${NC}\n"
    exit 1
fi
printf "${GREEN}[SUCCESS] Virtual environment activated.${NC}\n"

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
    if [[ $INSTALL_OLLAMA == "y" || $INSTALL_OLLAMA == "Y" ]]; then
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
if [[ $RUN_APP == "y" || $RUN_APP == "Y" ]]; then
    echo ""
    echo "Starting VerbalCodeAI..."
    python app.py
fi
