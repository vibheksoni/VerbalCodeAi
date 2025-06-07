@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================
echo  _    __          __          __   ______          __        ___    _
echo ^| ^|  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   ^|  (_)
echo ^| ^| / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /^| ^| / /
echo ^| ^|/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ ^|/ /
echo ^|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  ^|_/_/
echo.
echo ========================================================
echo Windows Setup Script for VerbalCodeAI
echo ========================================================
echo.

:: Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11.6 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYTHON_VERSION=%%V
echo [INFO] Detected Python version: %PYTHON_VERSION%

:: Extract version components
set MAJOR=0
set MINOR=0
set PATCH=0
for /f "tokens=1,2,3 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
    set PATCH=%%c
)

if %MAJOR% LSS 3 (
    echo [WARNING] Python version %PYTHON_VERSION% may be too old.
    echo This application was tested with Python 3.11.6.
    echo You may encounter issues with older versions.
    echo.
    set /p CONTINUE="Do you want to continue anyway? (Y/N): "
    if /i "%CONTINUE%" NEQ "Y" exit /b 1
) else (
    if %MAJOR% EQU 3 (
        if %MINOR% LSS 11 (
            echo [WARNING] Python version %PYTHON_VERSION% may be too old.
            echo This application was tested with Python 3.11.6.
            echo You may encounter issues with older versions.
            echo.
            set /p CONTINUE="Do you want to continue anyway? (Y/N): "
            if /i "%CONTINUE%" NEQ "Y" exit /b 1
        )
    )
)

:: Create virtual environment
echo.
echo [STEP 1] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

:: Activate virtual environment
echo.
echo [STEP 2] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)
echo [SUCCESS] Virtual environment activated.

:: Upgrade pip
echo.
echo [STEP 3] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip, but continuing with installation.
)

:: Install dependencies
echo.
echo [STEP 4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
)
echo [SUCCESS] Dependencies installed.

:: Check if Ollama is installed
echo.
echo [STEP 5] Checking for Ollama...
where ollama > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not installed or not in PATH.
    echo You will need to install Ollama to use local models.
    echo Download from: https://ollama.com/download
    echo.
    echo Would you like to open the Ollama download page? (Y/N)
    set /p OPEN_OLLAMA="Enter your choice (Y/N): "
    if /i "%OPEN_OLLAMA%"=="Y" (
        start https://ollama.com/download
    )
) else (
    echo [SUCCESS] Ollama is installed.
)

:: Create .env file if it doesn't exist
echo.
echo [STEP 6] Setting up environment variables...
if not exist .env (
    echo [INFO] Creating .env file with default settings...
    (
        echo # Provider can be: ollama, google, openai, anthropic, groq, or openrouter
        echo AI_CHAT_PROVIDER=ollama
        echo AI_EMBEDDING_PROVIDER=ollama
        echo AI_DESCRIPTION_PROVIDER=ollama
        echo AI_AGENT_BUDDY_PROVIDER=ollama
        echo.
        echo # Azure OpenAI
        echo # when using openai as a provider and you want to use Azure OpenAI, set these variables. Both values must be defined
        echo # the endpoint should be in the format: https://<your-resource-name>.openai.azure.com/
        echo AZURE_OPENAI_ENDPOINT=None
        echo # The API version to access your OpenAI resource. E.g. 2024-07-01-preview
        echo OPENAI_API_VERSION=None
        echo.
        echo # API Keys for each functionality (only needed if using that provider)
        echo # The same key will be used for the selected provider in each category
        echo AI_CHAT_API_KEY=None
        echo AI_EMBEDDING_API_KEY=None
        echo AI_DESCRIPTION_API_KEY=None
        echo AI_AGENT_BUDDY_API_KEY=None
        echo.
        echo # Model names for each provider
        echo # For ollama: llama2, codellama, mistral, etc. (embedding)
        echo # For OpenAI: gpt-4, gpt-3.5-turbo, text-embedding-ada-002 (embedding)
        echo # For OpenRouter: anthropic/claude-3-opus, openai/gpt-4-turbo, google/gemini-pro, etc.
        echo # For Google: gemini-pro, gemini-pro-vision
        echo # For Anthropic: claude-3-5-sonnet-latest, claude-3-opus-20240229, claude-3-haiku-20240307
        echo # For Groq: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
        echo CHAT_MODEL=llama2
        echo EMBEDDING_MODEL=all-minilm:33m
        echo DESCRIPTION_MODEL=llama2
        echo AI_AGENT_BUDDY_MODEL=llama3.2
        echo.
        echo # Model Tempratures
        echo CHAT_MODEL_TEMPERATURE=0.7
        echo DESCRIPTION_MODEL_TEMPERATURE=0.3
        echo AI_AGENT_BUDDY_MODEL_TEMPERATURE=0.7
        echo INTENT_DETECTION_TEMPERATURE=0.1
        echo.
        echo # Model Max Tokens
        echo CHAT_MODEL_MAX_TOKENS=4096
        echo DESCRIPTION_MODEL_MAX_TOKENS=4096
        echo AI_AGENT_BUDDY_MODEL_MAX_TOKENS=4096
        echo INTENT_DETECTION_MAX_TOKENS=4096
        echo.
        echo # Other Model Settings
        echo CHAT_MODEL_TOP_P=0.95
        echo CHAT_MODEL_TOP_K=40
        echo DESCRIPTION_MODEL_TOP_P=0.95
        echo DESCRIPTION_MODEL_TOP_K=40
        echo INTENT_DETECTION_TOP_P=0.95
        echo INTENT_DETECTION_TOP_K=40
        echo.
        echo # Optional: Site information for OpenRouter rankings
        echo SITE_URL=http://localhost:3000
        echo SITE_NAME=Local Development
        echo.
        echo # Performance settings (LOW, MEDIUM, MAX)
        echo # LOW: Minimal resource usage, suitable for low-end systems
        echo # MEDIUM: Balanced resource usage, suitable for most systems
        echo # MAX: Maximum resource usage, suitable for high-end systems
        echo PERFORMANCE_MODE=MEDIUM
        echo # Maximum number of threads to use (will be calculated automatically if not set)
        echo MAX_THREADS=16
        echo # Cache size for embedding queries (higher values use more memory but improve performance)
        echo EMBEDDING_CACHE_SIZE=1000
        echo # Similarity threshold for embedding search (lower values return more results but may be less relevant)
        echo EMBEDDING_SIMILARITY_THRESHOLD=0.05
        echo.
        echo # API Rate Limiting Settings
        echo # Delay in milliseconds between embedding API calls to prevent rate limiting
        echo # Recommended: 100ms for Google, 0ms for OpenAI/Ollama (set to 0 to disable)
        echo EMBEDDING_API_DELAY_MS=100
        echo # Delay in milliseconds between description generation API calls to prevent rate limiting
        echo # Recommended: 100ms for Google, 0ms for OpenAI/Ollama (set to 0 to disable)
        echo DESCRIPTION_API_DELAY_MS=100
        echo.
        echo # Maximum number of threads to use (will be calculated automatically if not set)
        echo # MAX_THREADS=16
        echo.
        echo # UI Settings
        echo # Enable/disable markdown rendering (TRUE/FALSE)
        echo ENABLE_MARKDOWN_RENDERING=TRUE
        echo # Show thinking blocks in AI responses (TRUE/FALSE)
        echo SHOW_THINKING_BLOCKS=FALSE
        echo # Enable streaming mode for AI responses (TRUE/FALSE) # Tends to be slower for some reason # Broken for openrouter TODO: Fix this at some point !
        echo ENABLE_STREAMING_MODE=FALSE
        echo # Enable chat logging to save conversations (TRUE/FALSE)
        echo CHAT_LOGS=FALSE
        echo # Enable memory for AI conversations (TRUE/FALSE)
        echo MEMORY_ENABLED=TRUE
        echo # Maximum number of memory items to store
        echo MAX_MEMORY_ITEMS=10
        echo # Execute commands without confirmation (TRUE/FALSE)
        echo # When FALSE, the user will be prompted to confirm before executing any command
        echo # When TRUE, commands will execute automatically without confirmation
        echo COMMANDS_YOLO=FALSE
        echo.
        echo # HTTP API Server Settings
        echo # Allow connections from any IP address (TRUE/FALSE)
        echo # When FALSE, the server only accepts connections from localhost (127.0.0.1)
        echo # When TRUE, the server accepts connections from any IP address (0.0.0.0)
        echo # WARNING: Setting this to TRUE may expose your API to the internet
        echo HTTP_ALLOW_ALL_ORIGINS=FALSE
        echo.
        echo # MCP Server Settings
        echo # URL of the HTTP API server
        echo MCP_API_URL=http://localhost:8000
        echo # Port to run the HTTP API server on
        echo MCP_HTTP_PORT=8000
    ) > .env
    echo [SUCCESS] Created .env file with default settings.
) else (
    echo [INFO] .env file already exists. Skipping creation.
)

echo.
echo ========================================================
echo VerbalCodeAI setup completed successfully!
echo.
echo To start the application, run:
echo   venv\Scripts\activate
echo   python app.py
echo.
echo For more information, see the README.md file.
echo ========================================================
echo.

:: Offer to run the application
echo Would you like to run VerbalCodeAI now? (Y/N)
set /p RUN_APP="Enter your choice (Y/N): "
if /i "%RUN_APP%"=="Y" (
    echo.
    echo Starting VerbalCodeAI...
    python app.py
)

endlocal
