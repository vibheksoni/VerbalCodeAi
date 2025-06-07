#!/usr/bin/env python
"""
VerbalCodeAI - Terminal Application

A simple terminal-based interface for the VerbalCodeAI code assistant.
Also provides an HTTP API server when run with the --serve option.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import colorama
import uvicorn
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

try:
    from mods.banners import display_animated_banner
    from mods.llms import get_current_provider
    from mods.code.agent_mode import AgentMode
    from mods.code.decisions import ChatHandler, FileSelector, ProjectAnalyzer
    from mods.code.indexer import FileIndexer
    from mods.http_api import create_app
    from mods.terminal_ui import StreamingResponseHandler, display_response
    from mods.terminal_utils import (
        clear_screen,
        create_progress_bar,
        create_spinner,
        get_terminal_size,
        render_markdown,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def setup_logging() -> logging.Logger:
    """Set up logging for the application.

    Returns:
        logging.Logger: The logger for the application.
    """
    colorama.init()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    try:
        for file in logs_dir.glob("*"):
            try:
                file.unlink()
            except (PermissionError, OSError):
                pass
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not clean up log directory: {e}{Style.RESET_ALL}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"verbalcodeai_{timestamp}.log"
    print(f"{Fore.CYAN}Logging to: {Style.BRIGHT}{log_file}{Style.RESET_ALL}")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    class StatisticsFilter(logging.Filter):
        """Filter log records to separate statistics and errors."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Filter log records to separate statistics and errors.

            Args:
                record (logging.LogRecord): The log record to filter.

            Returns:
                bool: True if the record should be included, False otherwise.
            """
            return (hasattr(record, "msg") and "[STAT]" in str(record.msg)) or record.levelno >= logging.ERROR

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(StatisticsFilter())
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    try:
        indexer_log_file = logs_dir / f"indexer_{timestamp}.log"
        indexer_handler = logging.FileHandler(indexer_log_file, mode="w", encoding="utf-8")
        indexer_handler.setLevel(logging.DEBUG)
        indexer_handler.setFormatter(file_formatter)
        logging.getLogger("VerbalCodeAI.Indexer").addHandler(indexer_handler)
    except (PermissionError, OSError) as e:
        print(f"{Fore.YELLOW}Warning: Could not create indexer log file: {e}{Style.RESET_ALL}")
        logging.getLogger("VerbalCodeAI.Indexer").addHandler(file_handler)

    logging.getLogger("VerbalCodeAI").setLevel(logging.DEBUG)
    logging.getLogger("VerbalCodeAI.Indexer").setLevel(logging.DEBUG)

    return logging.getLogger("VerbalCodeAI")


class VerbalCodeAI:
    """Main application class for the VerbalCodeAI terminal interface."""

    def __init__(self) -> None:
        """Initialize the application."""
        load_dotenv()

        self.logger: logging.Logger = logging.getLogger("VerbalCodeAI")
        self.indexer: Optional[FileIndexer] = None
        self.file_selector: Optional[FileSelector] = None
        self.project_analyzer: Optional[ProjectAnalyzer] = None
        self.chat_handler: Optional[ChatHandler] = None
        self.settings_path: str = os.path.join(os.path.dirname(__file__), ".app_settings.json")
        self.last_directory: str = self._load_last_directory()
        self.project_info: Dict[str, Any] = {}
        self.recent_projects: List[Dict[str, str]] = self._load_recent_projects()
        self.index_outdated: bool = False
        self.chat_history: List[Dict[str, str]] = []
        self.agent_mode_instance: Optional[AgentMode] = None

        self.enable_markdown_rendering: bool = self._get_env_bool("ENABLE_MARKDOWN_RENDERING", True)
        self.show_thinking_blocks: bool = self._get_env_bool("SHOW_THINKING_BLOCKS", False)
        self.enable_streaming_mode: bool = self._get_env_bool("ENABLE_STREAMING_MODE", False)

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean value from environment variables.

        Args:
            key (str): The environment variable key.
            default (bool): The default value if the key is not found. Defaults to False.

        Returns:
            bool: The boolean value.
        """
        value: str = os.getenv(key, str(default)).upper()
        return value in ("TRUE", "YES", "1", "Y", "T")

    def _load_last_directory(self) -> str:
        """Load the last indexed directory from settings.

        Returns:
            str: The last indexed directory.
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    settings: Dict[str, Any] = json.load(f)
                    return settings.get("last_directory", "")
            return ""
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return ""

    def _save_last_directory(self, directory: str) -> None:
        """Save the last indexed directory to settings.

        Args:
            directory (str): The directory to save.
        """
        try:
            settings: Dict[str, Any] = {}
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    settings = json.load(f)
            settings["last_directory"] = directory
            self._add_to_recent_projects(directory)

            with open(self.settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def _load_recent_projects(self) -> List[Dict[str, str]]:
        """Load the list of recent projects from settings.

        Returns:
            List[Dict[str, str]]: List of recent projects with path and timestamp.
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    settings: Dict[str, Any] = json.load(f)
                    return settings.get("recent_projects", [])
            return []
        except Exception as e:
            self.logger.error(f"Error loading recent projects: {e}")
            return []

    def _add_to_recent_projects(self, directory: str) -> None:
        """Add a directory to the recent projects list.

        Args:
            directory (str): The directory to add.
        """
        try:
            project_entry: Dict[str, str] = {
                "path": directory,
                "name": os.path.basename(directory),
                "timestamp": datetime.now().isoformat(),
            }

            self.recent_projects = [p for p in self.recent_projects if p.get("path") != directory]
            self.recent_projects.insert(0, project_entry)
            self.recent_projects = self.recent_projects[:10]

            settings: Dict[str, Any] = {}
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    settings = json.load(f)

            settings["recent_projects"] = self.recent_projects

            with open(self.settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error adding to recent projects: {e}")

    def display_menu(self) -> None:
        """Display the main menu."""
        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "VerbalCodeAI - Code Assistant" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        if not self.indexer:
            print(Fore.YELLOW + "No code indexed yet. Please select option 1 to index code." + Style.RESET_ALL)
        elif self.index_outdated:
            print(
                Fore.RED
                + Style.BRIGHT
                + "[INDEX OUTDATED] "
                + Style.RESET_ALL
                + Fore.YELLOW
                + "Please select option 1 to update the index."
                + Style.RESET_ALL
            )

        print(
            Fore.YELLOW
            + "WARNING: Consider using a smaller OpenRouter model for descriptions (Free One) to reduce costs."
            + Style.RESET_ALL
        )
        print(
            Fore.YELLOW
            + "WARNING: Chat with AI and Max Chat Mode are expensive. Use Agent Mode for cheaper and faster responses."
            + Style.RESET_ALL
        )

        print(Fore.GREEN + "1. " + Style.BRIGHT + "Index Code" + Style.RESET_ALL)
        print(Fore.GREEN + "2. " + Style.BRIGHT + "Chat with AI (Expensive)" + Style.RESET_ALL)
        print(Fore.GREEN + "3. " + Style.BRIGHT + "Max Chat Mode (Expensive)" + Style.RESET_ALL)
        print(Fore.GREEN + "4. " + Style.BRIGHT + "Agent Mode (Cheapest Option)" + Style.RESET_ALL)
        print(Fore.GREEN + "5. " + Style.BRIGHT + "Force Reindex" + Style.RESET_ALL)
        print(Fore.GREEN + "6. " + Style.BRIGHT + "View Indexed Files" + Style.RESET_ALL)
        print(Fore.GREEN + "7. " + Style.BRIGHT + "View Project Info" + Style.RESET_ALL)
        print(Fore.GREEN + "8. " + Style.BRIGHT + "Recent Projects" + Style.RESET_ALL)
        print(Fore.GREEN + "9. " + Style.BRIGHT + "Enhance Prompt" + Style.RESET_ALL)

        print(Fore.CYAN + "-" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Settings" + Style.RESET_ALL)

        markdown_status: str = (
            f"{Fore.GREEN}Enabled{Style.RESET_ALL}"
            if self.enable_markdown_rendering
            else f"{Fore.RED}Disabled{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}10. {Style.BRIGHT}Toggle Markdown Rendering{Style.RESET_ALL} [{markdown_status}]")

        thinking_status: str = (
            f"{Fore.GREEN}Enabled{Style.RESET_ALL}"
            if self.show_thinking_blocks
            else f"{Fore.RED}Disabled{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}11. {Style.BRIGHT}Toggle Thinking Blocks{Style.RESET_ALL} [{thinking_status}]")

        streaming_status: str = (
            f"{Fore.GREEN}Enabled{Style.RESET_ALL}"
            if self.enable_streaming_mode
            else f"{Fore.RED}Disabled{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}12. {Style.BRIGHT}Toggle Streaming Mode{Style.RESET_ALL} [{streaming_status}]")

        print(f"{Fore.GREEN}13. {Style.BRIGHT}Clear Screen{Style.RESET_ALL}")
        print(f"{Fore.GREEN}14. {Style.BRIGHT}Exit{Style.RESET_ALL}")

        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

    def index_directory(self, force_reindex: bool = False) -> None:
        """Index a directory.

        Args:
            force_reindex (bool): Whether to force reindexing of all files. Defaults to False.
        """
        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(
            Fore.CYAN
            + Style.BRIGHT
            + ("Force Reindex Directory" if force_reindex else "Index Code Directory")
            + Style.RESET_ALL
        )
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        if self.indexer and (self.index_outdated or force_reindex):
            directory: str = self.indexer.root_path
            print(f"{Fore.YELLOW}Using current directory: {Fore.CYAN}{directory}{Style.RESET_ALL}")
        else:
            default_dir: str = self.last_directory if self.last_directory else os.getcwd()
            print(
                f"{Fore.YELLOW}Enter directory path {Fore.CYAN}(default: {default_dir}){Fore.YELLOW}:{Style.RESET_ALL}"
            )
            directory: str = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()

            if not directory:
                directory = default_dir

            if not os.path.isdir(directory):
                print(f"{Fore.RED}Error: '{directory}' is not a valid directory.{Style.RESET_ALL}")
                return

        print(f"{Fore.CYAN}Indexing directory: {Style.BRIGHT}{directory}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This may take some time depending on the size of the codebase...{Style.RESET_ALL}")

        try:
            if not self.indexer or self.indexer.root_path != directory:
                self.logger.info(f"Creating indexer for directory: {directory}")
                self.indexer: FileIndexer = FileIndexer(directory)

                self.file_selector: FileSelector = FileSelector()

                self.project_analyzer: ProjectAnalyzer = ProjectAnalyzer(self.indexer)
                self.chat_handler: ChatHandler = ChatHandler(self.indexer, self.file_selector)
            else:
                self.logger.info(f"Using existing indexer for directory: {directory}")

            index_status: Dict[str, Any] = self.indexer.is_index_complete()
            self.index_outdated: bool = not index_status.get("complete", False)

            self.logger.info("Starting indexing process")
            print(f"{Fore.CYAN}Starting indexing process...{Style.RESET_ALL}")

            outdated_files: List[str] = []
            indexed_files: List[str] = []

            indexed_count: List[int] = [0]
            start_time: float = time.time()
            total_files: int = 0

            def progress_callback() -> bool:
                """Callback function to report indexing progress.

                Returns:
                    bool: Always returns False.
                """
                indexed_count[0] += 1
                elapsed: float = time.time() - start_time
                files_per_second: float = indexed_count[0] / elapsed if elapsed > 0 else 0

                if indexed_count[0] % 5 == 0 or indexed_count[0] == total_files:
                    percent: int = int((indexed_count[0] / total_files) * 100) if total_files > 0 else 0
                    eta: float = ((total_files - indexed_count[0]) / files_per_second) if files_per_second > 0 else 0
                    eta_str: str = f"{int(eta / 60)}m {int(eta % 60)}s" if eta > 0 else "0s"

                    terminal_width: int
                    terminal_width, _ = get_terminal_size()
                    bar_width: int = min(50, terminal_width - 30 if terminal_width > 30 else terminal_width)

                    filled_width: int = 0
                    if total_files > 0:
                        filled_width = int(bar_width * indexed_count[0] / total_files)

                    bar: str = f"{Fore.GREEN}{'█' * filled_width}{Fore.WHITE}{'░' * (bar_width - filled_width)}"

                    sys.stdout.write("\r" + " " * terminal_width)

                    sys.stdout.write(
                        f"\r{Fore.CYAN}[{bar}{Fore.CYAN}] {Fore.YELLOW}{percent}%{Fore.CYAN} - "
                        f"{Style.BRIGHT}{indexed_count[0]}/{total_files}{Style.NORMAL} files - "
                        f"{files_per_second:.1f} files/sec - "
                        f"ETA: {Fore.MAGENTA}{eta_str}{Style.RESET_ALL}"
                    )
                    sys.stdout.flush()

                    if indexed_count[0] == total_files:
                        sys.stdout.write("\n")
                        sys.stdout.flush()

                return False

            spinner = create_spinner()

            stop_spinner_flag: List[bool] = [False]

            def spin_fn():
                while not stop_spinner_flag[0]:
                    spinner()
                    time.sleep(0.1)

            spinner_thread: threading.Thread = threading.Thread(target=spin_fn)
            spinner_thread.daemon = True

            if force_reindex:
                self.logger.info("Force reindexing all files")
                print(f"{Fore.YELLOW}Force reindexing all files...{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Preparing to reindex...{Style.RESET_ALL}")

                stop_spinner_flag[0] = False
                spinner_thread.start()

                try:
                    all_files: List[str] = self.indexer._get_all_indexable_files()
                    total_files = len(all_files)
                    self.logger.info(f"[STAT] Estimated {total_files} files to reindex")
                except Exception as e:
                    self.logger.error(f"Error estimating files to reindex: {e}", exc_info=True)
                finally:
                    stop_spinner_flag[0] = True
                    if spinner_thread.is_alive():
                        spinner_thread.join(timeout=1.0)
                    terminal_width, _ = get_terminal_size()
                    sys.stdout.write("\r" + " " * terminal_width + "\r")
                    sys.stdout.flush()

                if total_files > 0:
                    print(
                        f"{Fore.CYAN}Found approximately {Style.BRIGHT}{total_files}{Style.NORMAL} files to reindex{Style.RESET_ALL}"
                    )
                else:
                    print(f"{Fore.YELLOW}Could not estimate file count or no files found to reindex.{Style.RESET_ALL}")

                if total_files > 0:
                    print(f"{Fore.CYAN}Starting reindexing process...{Style.RESET_ALL}")
                    try:
                        indexed_files = self.indexer.force_reindex_all(progress_callback)
                    finally:
                        terminal_width, _ = get_terminal_size()
                        sys.stdout.write("\r" + " " * terminal_width + "\r")
                        sys.stdout.flush()
                else:
                    indexed_files = []

                self.logger.info(f"[STAT] Force reindexed {len(indexed_files)} files")
                print(
                    f"{Fore.GREEN}Force reindexed {Style.BRIGHT}{len(indexed_files)}{Style.NORMAL} files{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.CYAN}Finding files to index...{Style.RESET_ALL}")

                if not spinner_thread.is_alive():
                    spinner_thread = threading.Thread(target=spin_fn)
                    spinner_thread.daemon = True

                stop_spinner_flag[0] = False
                spinner_thread.start()

                try:
                    outdated_files = self.indexer.get_outdated_files()
                    total_files = len(outdated_files)
                except Exception as e:
                    self.logger.error(f"Error getting outdated files: {e}", exc_info=True)
                    outdated_files = []
                    total_files = 0
                    print(f"{Fore.RED}Error getting outdated files: {e}{Style.RESET_ALL}")
                finally:
                    stop_spinner_flag[0] = True
                    if spinner_thread.is_alive():
                        spinner_thread.join(timeout=1.0)
                    terminal_width, _ = get_terminal_size()
                    sys.stdout.write("\r" + " " * terminal_width + "\r")
                    sys.stdout.flush()

                if outdated_files:
                    self.logger.info(f"[STAT] Found {len(outdated_files)} files to index")
                    print(
                        f"{Fore.GREEN}Found {Style.BRIGHT}{len(outdated_files)}{Style.NORMAL} files to index{Style.RESET_ALL}"
                    )
                    print(f"\n{Fore.CYAN}Starting indexing process...{Style.RESET_ALL}")
                    try:
                        indexed_files = self.indexer.index_directory(progress_callback)
                    finally:
                        terminal_width, _ = get_terminal_size()
                        sys.stdout.write("\r" + " " * terminal_width + "\r")
                        sys.stdout.flush()
                else:
                    self.logger.info("[STAT] No files need to be indexed")
                    print(f"{Fore.GREEN}No files need to be indexed{Style.RESET_ALL}")
                    indexed_files = []

            self._save_last_directory(directory)
            self.index_outdated = False

            self.logger.info(f"[STAT] Indexing complete! Successfully indexed {len(indexed_files)} files.")
            print(
                f"\n{Fore.GREEN}{Style.BRIGHT}Indexing complete! {Style.NORMAL}Successfully indexed {Style.BRIGHT}{len(indexed_files)}{Style.NORMAL} files.{Style.RESET_ALL}"
            )

            project_info_path: str = os.path.join(self.indexer.index_dir, "project_info.json")
            update_project_info: bool = True
            if len(indexed_files) > 0:
                if os.path.exists(project_info_path):
                    print(
                        f"\n{Fore.YELLOW}Project information already exists. Would you like to update it? (y/n){Style.RESET_ALL}"
                    )
                    choice: str = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip().lower()
                    update_project_info = choice == "y" or choice == "yes"

                if update_project_info:
                    if self.project_analyzer:
                        self.project_info = self.project_analyzer.collect_project_info()

                        if self.chat_handler:
                            self.chat_handler.set_project_info(self.project_info)
                    else:
                        self.logger.error("ProjectAnalyzer not initialized")

        except Exception as e:
            self.logger.error(f"Error during indexing: {e}", exc_info=True)
            print(f"{Fore.RED}{Style.BRIGHT}Error during indexing: {e}{Style.RESET_ALL}")

    async def agent_mode(self) -> None:
        """Run the AI agent mode for interactive codebase exploration."""
        if not self.indexer:
            print(f"\n{Fore.RED}Error: No code has been indexed yet. Please index a directory first.{Style.RESET_ALL}")
            return

        if not self.agent_mode_instance:
            self.agent_mode_instance: AgentMode = AgentMode(self.indexer)

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Agent Mode" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(f"{Fore.YELLOW}In Agent Mode, the AI can use tools to explore and understand your codebase.")
        print(f"{Fore.YELLOW}Available tools:")
        print(f"{Fore.YELLOW}  Search tools: embed_search, semantic_search, grep, regex_advanced_search, file_type_search")
        print(f"{Fore.YELLOW}  File tools: read_file, file_stats, directory_tree, get_file_description, get_file_metadata")
        print(f"{Fore.YELLOW}  Code analysis: find_functions, find_classes, find_usage, cross_reference, code_analysis,")
        print(f"{Fore.YELLOW}                 get_functions, get_classes, get_variables, get_imports, explain_code")
        print(f"{Fore.YELLOW}  Version control: git_history, version_control_search, search_imports")
        print(f"{Fore.YELLOW}  Project tools: get_project_description, get_instructions, create_instructions_template")
        print(f"{Fore.YELLOW}  Memory tools: add_memory, get_memories, search_memories")
        print(f"{Fore.YELLOW}  System tools: run_command, read_terminal, kill_terminal, list_terminals")
        print(f"{Fore.YELLOW}  Helper tools: ask_buddy (with context-aware second opinions)")
        print(f"{Fore.YELLOW}  Web tools: google_search, ddg_search, bing_news_search, fetch_webpage, get_base_knowledge")
        print(
            f"{Fore.YELLOW}Type your questions about the codebase. Type '{Fore.RED}exit{Fore.YELLOW}' to return to the main menu.{Style.RESET_ALL}"
        )

        while True:
            provider, model = get_current_provider()
            print(
                f"\n{Fore.YELLOW}Current provider: {Fore.CYAN}{provider.upper()}"
                f"{Fore.YELLOW}, model: {Fore.CYAN}{model}{Style.RESET_ALL}"
            )
            print(
                f"\n{Fore.GREEN}Your question (or '{Fore.RED}exit{Fore.GREEN}' to return to menu):{Style.RESET_ALL}"
            )
            user_input: str = input(f"{Fore.CYAN}> {Style.RESET_ALL}").strip()

            if user_input.lower() == "exit":
                break

            if not user_input:
                continue

            try:
                print(f"\n{Fore.YELLOW}Processing your question with Agent Mode...{Style.RESET_ALL}")
                self.logger.info(f"[STAT] Processing question in Agent Mode: {user_input[:50]}...")

                await self.agent_mode_instance.process_query(user_input)

            except Exception as e:
                self.logger.error(f"Error in Agent Mode: {e}", exc_info=True)
                print(f"\n{Fore.RED}Error in Agent Mode: {e}{Style.RESET_ALL}")

    async def enhance_prompt(self) -> None:
        """Run the prompt enhancement feature."""
        if not self.indexer:
            print(f"\n{Fore.RED}Error: No code has been indexed yet. Please index a directory first.{Style.RESET_ALL}")
            return

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Enhance Prompt" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(f"{Fore.YELLOW}This feature will analyze your codebase and enhance your prompt with relevant context.")
        print(f"{Fore.YELLOW}The enhanced prompt will include:")
        print(f"{Fore.YELLOW}  • Expanded and clarified task description")
        print(f"{Fore.YELLOW}  • Relevant code snippets and file references")
        print(f"{Fore.YELLOW}  • Technical context and constraints from the codebase")
        print(f"{Fore.YELLOW}  • Implementation guidance based on existing patterns")
        print(f"{Fore.YELLOW}Type your initial prompt/instruction below.{Style.RESET_ALL}")

        while True:
            provider, model = get_current_provider()
            print(
                f"\n{Fore.YELLOW}Current provider: {Fore.CYAN}{provider.upper()}"
                f"{Fore.YELLOW}, model: {Fore.CYAN}{model}{Style.RESET_ALL}"
            )
            print(
                f"\n{Fore.GREEN}Enter your initial prompt (or '{Fore.RED}exit{Fore.GREEN}' to return to menu):{Style.RESET_ALL}"
            )
            user_input: str = input(f"{Fore.CYAN}> {Style.RESET_ALL}").strip()

            if user_input.lower() == "exit":
                break

            if not user_input:
                continue

            try:
                print(f"\n{Fore.YELLOW}Analyzing codebase and enhancing your prompt...{Style.RESET_ALL}")
                self.logger.info(f"[STAT] Enhancing prompt: {user_input[:50]}...")

                if not hasattr(self, 'prompt_enhancer') or not self.prompt_enhancer:
                    from mods.code.prompt_enhancer import PromptEnhancer
                    self.prompt_enhancer = PromptEnhancer(self.indexer)

                enhanced_prompt = await self.prompt_enhancer.enhance_prompt(user_input)

                terminal_width: int
                terminal_width, _ = get_terminal_size()
                print(f"\n{Fore.BLUE}{Style.BRIGHT}✧ ENHANCED PROMPT ✧{Style.RESET_ALL}")
                print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")

                if self.enable_markdown_rendering:
                    rendered_content = render_markdown(enhanced_prompt, width=terminal_width-2)
                    print(rendered_content)
                else:
                    print(enhanced_prompt)

                print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")

                print(f"\n{Fore.CYAN}Options:{Style.RESET_ALL}")
                print(f"{Fore.GREEN}1. {Style.BRIGHT}Continue with another prompt{Style.RESET_ALL}")
                print(f"{Fore.GREEN}2. {Style.BRIGHT}Save enhanced prompt to file{Style.RESET_ALL}")
                print(f"{Fore.GREEN}3. {Style.BRIGHT}Return to main menu{Style.RESET_ALL}")

                choice = input(f"\n{Fore.YELLOW}Enter your choice (1-3): {Style.RESET_ALL}").strip()

                if choice == "2":
                    filename = f"enhanced_prompt_{int(time.time())}.md"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(enhanced_prompt)
                        print(f"{Fore.GREEN}Enhanced prompt saved to: {filename}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error saving file: {e}{Style.RESET_ALL}")
                elif choice == "3":
                    break

            except Exception as e:
                self.logger.error(f"Error in Enhance Prompt: {e}", exc_info=True)
                print(f"\n{Fore.RED}Error enhancing prompt: {e}{Style.RESET_ALL}")

    def chat_with_ai(self, max_chat_mode: bool = False) -> None:
        """Chat with the AI about code.

        Args:
            max_chat_mode (bool): Whether to use Max Chat mode, which sends full file contents to the AI. Defaults to False.
        """
        if not self.indexer:
            print(f"\n{Fore.RED}Error: No code has been indexed yet. Please index a directory first.{Style.RESET_ALL}")
            return

        if not self.chat_handler:
            self.chat_handler: ChatHandler = ChatHandler(self.indexer, self.file_selector, self.project_info)

        if not self.project_analyzer:
            self.project_analyzer: ProjectAnalyzer = ProjectAnalyzer(self.indexer)

        if not self.project_info:
            print(f"\n{Fore.YELLOW}Loading project information before starting chat...{Style.RESET_ALL}")
            self.project_info = self.project_analyzer.load_project_info()

            if not self.project_info:
                print(f"\n{Fore.YELLOW}Collecting project information before starting chat...{Style.RESET_ALL}")
                self.project_info = self.project_analyzer.collect_project_info()

            self.chat_handler.set_project_info(self.project_info)

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        if max_chat_mode:
            print(Fore.CYAN + Style.BRIGHT + "Max Chat Mode (Token Intensive)" + Style.RESET_ALL)
            print(
                Fore.RED + "WARNING: This mode sends full file contents to the AI and uses more tokens." + Style.RESET_ALL
            )
        else:
            print(Fore.CYAN + Style.BRIGHT + "Chat with AI" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(
            f"{Fore.YELLOW}Type your questions about the codebase. Type '{Fore.RED}exit{Fore.YELLOW}' to return to the main menu.{Style.RESET_ALL}"
        )

        while True:
            print(
                f"\n{Fore.GREEN}Your question (or '{Fore.RED}exit{Fore.GREEN}' to return to menu):{Style.RESET_ALL}"
            )
            user_input: str = input(f"{Fore.CYAN}> {Style.RESET_ALL}").strip()

            if user_input.lower() == "exit":
                break

            if not user_input:
                continue

            try:
                print(f"\n{Fore.YELLOW}Processing your question...{Style.RESET_ALL}")
                self.logger.info(f"[STAT] Processing question: {user_input[:50]}...")

                response: Union[str, Any]
                relevant_files: List[str]
                response, relevant_files = self.chat_handler.process_query(
                    user_input,
                    max_chat_mode=max_chat_mode,
                    streaming=self.enable_streaming_mode,
                )

                if relevant_files:
                    print(f"\n{Fore.CYAN}{Style.BRIGHT}Relevant Files:{Style.RESET_ALL}")
                    display_count: int = min(5, len(relevant_files))
                    for i, file_path in enumerate(relevant_files[:display_count], 1):
                        file_name: str = os.path.basename(file_path)
                        print(f"{Fore.GREEN}{i}. {Fore.WHITE}{file_name} {Fore.CYAN}({file_path}){Style.RESET_ALL}")

                    if len(relevant_files) > 5:
                        remaining: int = len(relevant_files) - 5
                        print(f"{Fore.YELLOW}+ {remaining} more file{'s' if remaining > 1 else ''}{Style.RESET_ALL}")

                    if max_chat_mode:
                        print(
                            f"\n{Fore.YELLOW}Including full content of {len(relevant_files)} files in the context.{Style.RESET_ALL}"
                        )
                else:
                    print(
                        f"{Fore.YELLOW}No relevant files found. Generating response based on general knowledge.{Style.RESET_ALL}"
                    )

                terminal_width: int
                terminal_width, _ = get_terminal_size()
                print(f"\n{Fore.CYAN}{Style.BRIGHT}AI Response:{Style.RESET_ALL}")
                print(Fore.CYAN + "-" * min(terminal_width, 80) + Style.RESET_ALL)

                if self.enable_streaming_mode:
                    self.logger.info("[STAT] Processing streaming AI response")

                    handler: StreamingResponseHandler = StreamingResponseHandler(
                        enable_markdown_rendering=self.enable_markdown_rendering,
                        show_thinking_blocks=self.show_thinking_blocks,
                        logger=self.logger,
                    )

                    def on_complete(complete_response: str):
                        self.chat_handler.add_to_history("assistant", complete_response)

                    try:
                        asyncio.run(handler.process_stream(response, on_complete))

                        self.logger.info("[STAT] Streaming AI response completed successfully")
                    except Exception as e:
                        self.logger.error(f"Error processing streaming response: {e}")
                        print(f"\n{Fore.RED}Error processing response: {e}{Style.RESET_ALL}")
                else:
                    self.logger.info("[STAT] AI response generated successfully")

                    display_response(
                        response,
                        enable_markdown_rendering=self.enable_markdown_rendering,
                        show_thinking_blocks=self.show_thinking_blocks,
                    )

                print(Fore.CYAN + "-" * min(terminal_width, 80) + Style.RESET_ALL)

                self.chat_history = self.chat_handler.get_chat_history(max_messages=10)

            except Exception as e:
                self.logger.error(f"Error generating response: {e}", exc_info=True)
                print(f"\n{Fore.RED}{Style.BRIGHT}Error generating response: {e}{Style.RESET_ALL}")

    def view_indexed_files(self) -> None:
        """View information about indexed files."""
        if not self.indexer:
            print(f"\n{Fore.RED}Error: No code has been indexed yet. Please index a directory first.{Style.RESET_ALL}")
            return

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Indexed Files" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        try:
            self.logger.info("[STAT] Viewing indexed files information")

            index_status: Dict[str, Any] = self.indexer.is_index_complete()
            status_str: str = "Complete" if index_status["complete"] else "Incomplete"
            status_color: str = Fore.GREEN if index_status["complete"] else Fore.YELLOW
            print(f"{Fore.CYAN}Index Status: {status_color}{Style.BRIGHT}{status_str}{Style.RESET_ALL}")

            if not index_status["complete"] and "reason" in index_status:
                print(f"{Fore.YELLOW}Reason: {index_status['reason']}{Style.RESET_ALL}")

            index_dir = self.indexer.index_dir
            print(f"\n{Fore.CYAN}Index Directory: {Fore.WHITE}{index_dir}{Style.RESET_ALL}")

            if os.path.exists(index_dir):
                metadata_dir = os.path.join(index_dir, "metadata")
                embeddings_dir = os.path.join(index_dir, "embeddings")
                descriptions_dir = os.path.join(index_dir, "descriptions")

                print(f"\n{Fore.CYAN}{Style.BRIGHT}Index Directory Structure:{Style.RESET_ALL}")
                header = f"  {Fore.CYAN}{'Directory':<15} {'Exists':<10} {'Files':<10} {'Size':<10}{Style.RESET_ALL}"
                separator = f"  {Fore.CYAN}{'-'*15} {'-'*10} {'-'*10} {'-'*10}{Style.RESET_ALL}"
                print(header)
                print(separator)

                if os.path.exists(metadata_dir):
                    files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]
                    size = sum(os.path.getsize(os.path.join(metadata_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'metadata':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'metadata':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )

                if os.path.exists(embeddings_dir):
                    files = [f for f in os.listdir(embeddings_dir) if f.endswith(".json")]
                    size = sum(os.path.getsize(os.path.join(embeddings_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'embeddings':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'embeddings':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )

                if os.path.exists(descriptions_dir):
                    files = [f for f in os.listdir(descriptions_dir) if f.endswith(".txt")]
                    size = sum(os.path.getsize(os.path.join(descriptions_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'descriptions':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'descriptions':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )
            else:
                print(f"{Fore.RED}Index directory does not exist.{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}{Style.BRIGHT}Sample Indexed Files:{Style.RESET_ALL}")
            try:
                sample_files = self.indexer.get_sample_files(5)
                if sample_files:
                    for i, file in enumerate(sample_files, 1):
                        print(f"{Fore.GREEN}{i}. {Fore.WHITE}{file}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No files have been indexed yet.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error retrieving sample files: {e}{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            input()

        except Exception as e:
            self.logger.error(f"Error viewing indexed files: {e}", exc_info=True)
            print(f"{Fore.RED}{Style.BRIGHT}Error viewing indexed files: {e}{Style.RESET_ALL}")

    def view_project_info(self) -> None:
        """View information about the project."""
        if not self.indexer:
            print(f"\n{Fore.RED}Error: No code has been indexed yet. Please index a directory first.{Style.RESET_ALL}")
            return

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Project Information" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        if not hasattr(self, "project_info") or not self.project_info:
            print(f"{Fore.YELLOW}No project information available yet.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Collecting project information...{Style.RESET_ALL}")

            if not self.project_analyzer:
                self.project_analyzer = ProjectAnalyzer(self.indexer)

            self.project_info = self.project_analyzer.load_project_info()
            if not self.project_info:
                self.project_info = self.project_analyzer.collect_project_info()

            if self.chat_handler:
                self.chat_handler.set_project_info(self.project_info)

        if not self.project_info:
            print(f"{Fore.RED}Failed to collect project information.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
            input()
            return

        print(f"\n{Fore.CYAN}{Style.BRIGHT}Project Summary:{Style.RESET_ALL}")

        terminal_width, _ = get_terminal_size()

        markdown_content = "# Project Summary\n\n"

        fields = [
            ("Project Name", "project_name"),
            ("Purpose", "purpose"),
            ("Languages", "languages"),
            ("Frameworks", "frameworks"),
            ("Structure", "structure"),
            ("Other Notes", "other_notes"),
        ]

        for display_name, field_name in fields:
            if field_name in self.project_info:
                value = self.project_info[field_name]
                if isinstance(value, list):
                    markdown_content += f"## {display_name}\n\n"
                    for item in value:
                        markdown_content += f"- {item}\n"
                    markdown_content += "\n"
                else:
                    markdown_content += f"## {display_name}\n\n{value}\n\n"

        rendered_content = render_markdown(markdown_content, width=terminal_width-2)
        print(rendered_content)

        print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        input()

        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Indexed Files" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        try:
            self.logger.info("[STAT] Viewing indexed files information")

            index_status = self.indexer.is_index_complete()
            status_str = "Complete" if index_status["complete"] else "Incomplete"
            status_color = Fore.GREEN if index_status["complete"] else Fore.YELLOW
            print(f"{Fore.CYAN}Index Status: {status_color}{Style.BRIGHT}{status_str}{Style.RESET_ALL}")

            if not index_status["complete"] and "reason" in index_status:
                print(f"{Fore.YELLOW}Reason: {index_status['reason']}{Style.RESET_ALL}")

            index_dir = self.indexer.index_dir
            print(f"\n{Fore.CYAN}Index Directory: {Fore.WHITE}{index_dir}{Style.RESET_ALL}")

            if os.path.exists(index_dir):
                metadata_dir = os.path.join(index_dir, "metadata")
                embeddings_dir = os.path.join(index_dir, "embeddings")
                descriptions_dir = os.path.join(index_dir, "descriptions")

                print(f"\n{Fore.CYAN}{Style.BRIGHT}Index Directory Structure:{Style.RESET_ALL}")
                header = f"  {Fore.CYAN}{'Directory':<15} {'Exists':<10} {'Files':<10} {'Size':<10}{Style.RESET_ALL}"
                separator = f"  {Fore.CYAN}{'-'*15} {'-'*10} {'-'*10} {'-'*10}{Style.RESET_ALL}"
                print(header)
                print(separator)

                if os.path.exists(metadata_dir):
                    files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]
                    size = sum(os.path.getsize(os.path.join(metadata_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'metadata':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'metadata':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )

                if os.path.exists(embeddings_dir):
                    files = [f for f in os.listdir(embeddings_dir) if f.endswith(".json")]
                    size = sum(os.path.getsize(os.path.join(embeddings_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'embeddings':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'embeddings':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )

                if os.path.exists(descriptions_dir):
                    files = [f for f in os.listdir(descriptions_dir) if f.endswith(".txt")]
                    size = sum(os.path.getsize(os.path.join(descriptions_dir, f)) for f in files)
                    print(
                        f"  {Fore.GREEN}{'descriptions':<15} {Fore.GREEN}{'Yes':<10} {Fore.YELLOW}{len(files):<10} {Fore.MAGENTA}{self._format_size(size):<10}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"  {Fore.GREEN}{'descriptions':<15} {Fore.RED}{'No':<10} {'-':<10} {'-':<10}{Style.RESET_ALL}"
                    )
            else:
                print(f"{Fore.RED}Index directory does not exist.{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}{Style.BRIGHT}Sample Indexed Files:{Style.RESET_ALL}")
            try:
                sample_files = self.indexer.get_sample_files(5)
                if sample_files:
                    for i, file in enumerate(sample_files, 1):
                        print(f"{Fore.GREEN}{i}. {Fore.WHITE}{file}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No files have been indexed yet.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error retrieving sample files: {e}{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            input()

        except Exception as e:
            self.logger.error(f"Error viewing indexed files: {e}", exc_info=True)
            print(f"{Fore.RED}{Style.BRIGHT}Error viewing indexed files: {e}{Style.RESET_ALL}")

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable format.

        Args:
            size_bytes (int): Size in bytes.

        Returns:
            str: Human-readable size.
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        if size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def collect_project_info(self, batch_size: int = 10) -> Dict[str, Any]:
        """Collect and analyze project information to provide context for AI responses.

        This method is now a wrapper around the ProjectAnalyzer.collect_project_info method.

        Args:
            batch_size (int): Number of files to process in each batch. Defaults to 10.

        Returns:
            Dict[str, Any]: Dictionary containing project information.
        """
        if not self.indexer:
            self.logger.warning("Cannot collect project info: No code has been indexed yet")
            return {}

        if not self.project_analyzer:
            self.project_analyzer = ProjectAnalyzer(self.indexer)

        project_info = self.project_analyzer.collect_project_info(batch_size)

        if self.chat_handler:
            self.chat_handler.set_project_info(project_info)

        self.project_info = project_info

        if project_info:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}Project Summary:{Style.RESET_ALL}")
            if "project_name" in project_info:
                print(f"{Fore.WHITE}Project: {Fore.GREEN}{project_info['project_name']}{Style.RESET_ALL}")
            if "purpose" in project_info:
                print(f"{Fore.WHITE}Purpose: {Fore.YELLOW}{project_info['purpose']}{Style.RESET_ALL}")
            if "languages" in project_info and project_info["languages"]:
                print(f"{Fore.WHITE}Languages: {Fore.MAGENTA}{', '.join(project_info['languages'])}{Style.RESET_ALL}")
            if "frameworks" in project_info and project_info["frameworks"]:
                print(f"{Fore.WHITE}Frameworks: {Fore.BLUE}{', '.join(project_info['frameworks'])}{Style.RESET_ALL}")

        return project_info

    def view_recent_projects(self) -> None:
        """View and select from recent projects."""
        print("\n" + Fore.CYAN + "=" * 50 + Style.RESET_ALL)
        print(Fore.CYAN + Style.BRIGHT + "Recent Projects" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

        if not self.recent_projects:
            print(f"{Fore.YELLOW}No recent projects found.{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            input()
            return

        print(f"{Fore.YELLOW}Select a project to load:{Style.RESET_ALL}")

        for i, project in enumerate(self.recent_projects, 1):
            name = project.get("name", os.path.basename(project.get("path", "Unknown")))
            path = project.get("path", "Unknown")
            timestamp = project.get("timestamp", "Unknown")

            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pass

            print(f"{Fore.GREEN}{i}. {Fore.WHITE}{name} {Fore.CYAN}({path}) {Fore.YELLOW}[{timestamp}]{Style.RESET_ALL}")

        print(f"{Fore.GREEN}0. {Fore.WHITE}Cancel{Style.RESET_ALL}")

        while True:
            try:
                choice = input(f"\n{Fore.YELLOW}Enter your choice (0-{len(self.recent_projects)}): {Style.RESET_ALL}").strip()

                if choice == "0":
                    return

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.recent_projects):
                    selected_project = self.recent_projects[choice_idx]
                    project_path = selected_project.get("path")

                    if os.path.isdir(project_path):
                        print(f"{Fore.GREEN}Loading project: {project_path}{Style.RESET_ALL}")
                        self.index_directory()
                        return
                    print(f"{Fore.RED}Error: Directory no longer exists: {project_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid choice. Please enter a number between 0 and {len(self.recent_projects)}.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")

    def toggle_markdown_rendering(self) -> None:
        """Toggle markdown rendering on/off."""
        self.enable_markdown_rendering = not self.enable_markdown_rendering
        status = "enabled" if self.enable_markdown_rendering else "disabled"
        print(f"\n{Fore.CYAN}Markdown rendering is now {Fore.GREEN if self.enable_markdown_rendering else Fore.RED}{status}{Style.RESET_ALL}")

        self._update_env_setting("ENABLE_MARKDOWN_RENDERING", str(self.enable_markdown_rendering).upper())

        print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        input()

    def toggle_thinking_blocks(self) -> None:
        """Toggle showing thinking blocks on/off."""
        self.show_thinking_blocks = not self.show_thinking_blocks
        status = "enabled" if self.show_thinking_blocks else "disabled"
        print(f"\n{Fore.CYAN}Showing thinking blocks is now {Fore.GREEN if self.show_thinking_blocks else Fore.RED}{status}{Style.RESET_ALL}")

        self._update_env_setting("SHOW_THINKING_BLOCKS", str(self.show_thinking_blocks).upper())

        print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        input()

    def toggle_streaming_mode(self) -> None:
        """Toggle streaming mode on/off."""
        self.enable_streaming_mode = not self.enable_streaming_mode
        status = "enabled" if self.enable_streaming_mode else "disabled"
        print(f"\n{Fore.CYAN}Streaming mode is now {Fore.GREEN if self.enable_streaming_mode else Fore.RED}{status}{Style.RESET_ALL}")

        self._update_env_setting("ENABLE_STREAMING_MODE", str(self.enable_streaming_mode).upper())

        print(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        input()

    def _update_env_setting(self, key: str, value: str) -> None:
        """Update a setting in the .env file.

        Args:
            key (str): The key to update.
            value (str): The new value.
        """
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if not os.path.exists(env_path):
                self.logger.warning(f"Cannot update {key} in .env file: File not found")
                return

            with open(env_path, "r") as f:
                lines = f.readlines()

            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break

            if not updated:
                lines.append(f"{key}={value}\n")

            with open(env_path, "w") as f:
                f.writelines(lines)

            self.logger.info(f"Updated {key}={value} in .env file")
        except Exception as e:
            self.logger.error(f"Error updating {key} in .env file: {e}")

    def run(self) -> None:
        """Run the application."""
        if not self.indexer and not self.last_directory:
            print(f"\n{Fore.YELLOW}Welcome to VerbalCodeAI! Let's start by indexing a code directory.{Style.RESET_ALL}")
            self.index_directory()
        elif self.recent_projects and not self.indexer:
            print(f"\n{Fore.YELLOW}Would you like to load a recent project? (y/n){Style.RESET_ALL}")
            choice = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip().lower()
            if choice in ("y", "yes"):
                self.view_recent_projects()

        while True:
            if self.indexer:
                index_status = self.indexer.is_index_complete()
                self.index_outdated = not index_status.get('complete', False)

            self.display_menu()
            choice = input(f"{Fore.YELLOW}Enter your choice (1-14): {Style.RESET_ALL}").strip()

            actions = {
                "1": lambda: self.index_directory(),
                "2": lambda: self.chat_with_ai(),
                "3": lambda: self.chat_with_ai(max_chat_mode=True),
                "4": lambda: asyncio.run(self.agent_mode()),
                "5": lambda: self.index_directory(force_reindex=True),
                "6": lambda: self.view_indexed_files(),
                "7": lambda: self.view_project_info(),
                "8": lambda: self.view_recent_projects(),
                "9": lambda: asyncio.run(self.enhance_prompt()),
                "10": lambda: self.toggle_markdown_rendering(),
                "11": lambda: self.toggle_thinking_blocks(),
                "12": lambda: self.toggle_streaming_mode(),
                "13": lambda: clear_screen()
            }

            if choice in actions or choice == "14":
                if choice == "14":
                    clear_screen()
                    print(f"\n{Fore.GREEN}Exiting VerbalCodeAI. Goodbye!{Style.RESET_ALL}")
                    break
                else:
                    clear_screen()
                    actions[choice]()
            else:
                print(f"\n{Fore.RED}Invalid choice. Please enter a number between 1 and 14.{Style.RESET_ALL}")

def run_http_server(port: int, allow_all_origins: bool = None) -> None:
    """Run the HTTP API server.

    Args:
        port: Port number to listen on
        allow_all_origins: Whether to allow all origins or only localhost.
                          If None, reads from environment variable.
    """
    app_instance = VerbalCodeAI()

    if allow_all_origins is None:
        allow_all_origins = app_instance._get_env_bool("HTTP_ALLOW_ALL_ORIGINS", False)

    app = create_app(allow_all_origins=allow_all_origins)

    host = "0.0.0.0" if allow_all_origins else "127.0.0.1"

    print(f"{Fore.GREEN}Starting HTTP API server on {host}:{port}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Available endpoints:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}- GET  /api/health{Style.RESET_ALL} - Health check")
    print(f"{Fore.CYAN}- POST /api/initialize{Style.RESET_ALL} - Initialize a directory")
    print(f"{Fore.CYAN}- POST /api/ask{Style.RESET_ALL} - Ask the agent a question")
    print(f"{Fore.CYAN}- POST /api/index/start{Style.RESET_ALL} - Start indexing a directory")
    print(f"{Fore.CYAN}- GET  /api/index/status{Style.RESET_ALL} - Get indexing status")

    if allow_all_origins:
        print(f"{Fore.RED}WARNING: Server is accessible from any IP address.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Server is only accessible from localhost.{Style.RESET_ALL}")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="VerbalCodeAI Terminal Application")
        parser.add_argument("directory", nargs="?", help="Directory to index")
        parser.add_argument("--serve", type=int, metavar="PORT", help="Run HTTP API server on specified port")
        parser.add_argument("--allow-all-origins", action="store_true", help="Allow all origins for HTTP API (not just localhost)")
        args = parser.parse_args()

        if args.serve:
            logger = setup_logging()
            logger.info(f"Starting VerbalCodeAI HTTP API Server on port {args.serve}")

            run_http_server(args.serve, args.allow_all_origins)
            sys.exit(0)

        display_animated_banner(frame_delay=0.2)

        print(f"{Fore.CYAN}{Style.BRIGHT}Starting VerbalCodeAI Terminal Application...{Style.RESET_ALL}")

        logger = setup_logging()
        logger.info("Starting VerbalCodeAI Terminal Application")

        app = VerbalCodeAI()

        if args.directory:
            directory = args.directory
            if os.path.isdir(directory):
                print(f"{Fore.CYAN}Using directory from command line: {Style.BRIGHT}{directory}{Style.RESET_ALL}")
                app.indexer = FileIndexer(directory)
                app._save_last_directory(directory)

                index_status = app.indexer.is_index_complete()
                app.index_outdated = not index_status.get('complete', False)
                if app.index_outdated:
                    print(f"{Fore.YELLOW}Index is outdated. Consider reindexing.{Style.RESET_ALL}")

        app.run()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Application terminated by user.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n{Fore.RED}{Style.BRIGHT}CRITICAL ERROR: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}See logs for details.{Style.RESET_ALL}")
        print(f"{Fore.RED}Error details:\n{error_details}{Style.RESET_ALL}")
        sys.exit(1)
