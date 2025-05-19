"""Terminal module for executing system commands.

This module provides functionality for executing system commands, reading terminal output,
killing terminal processes, and listing active terminal sessions.
"""

import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

try:
    import psutil
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    from colorama import Fore, Style, init
    init()
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import Fore, Style, init
    init()

import logging
logger = logging.getLogger("VerbalCodeAI.Terminal")

COMMANDS_YOLO = os.environ.get("COMMANDS_YOLO", "FALSE").upper() in ("TRUE", "YES", "1", "Y", "T")

class TerminalProcess:
    """Class representing a terminal process.

    This class provides methods for managing a terminal process, including
    starting, stopping, reading output, and writing input.
    """

    def __init__(self, command: str, cwd: str = None, env: Dict[str, str] = None):
        """Initialize a terminal process.

        Args:
            command (str): The command to execute.
            cwd (str, optional): The working directory. Defaults to None.
            env (Dict[str, str], optional): Environment variables. Defaults to None.
        """
        self.command = command
        self.cwd = cwd or os.getcwd()
        self.env = env or os.environ.copy()
        self.process = None
        self.stdout_buffer = []
        self.stderr_buffer = []
        self.terminal_id = None
        self.start_time = None
        self.end_time = None
        self.exit_code = None
        self.is_running = False
        self.is_shell = True
        self.terminal_type = self._detect_terminal_type()

    def _detect_terminal_type(self) -> str:
        """Detect the terminal type.

        Returns:
            str: The terminal type (powershell, cmd, bash, etc.).
        """
        if platform.system() == "Windows":
            try:
                subprocess.run(["powershell", "-Command", "echo 'test'"],
                               capture_output=True, text=True, check=True)
                return "powershell"
            except (subprocess.SubprocessError, FileNotFoundError):
                return "cmd"
        else:
            shell = os.environ.get("SHELL", "")
            if "bash" in shell:
                return "bash"
            elif "zsh" in shell:
                return "zsh"
            elif "fish" in shell:
                return "fish"
            else:
                return "sh"

    def start(self) -> bool:
        """Start the terminal process.

        Returns:
            bool: True if the process started successfully, False otherwise.
        """
        if self.is_running:
            logger.warning(f"Process already running: {self.command}")
            return False

        try:
            if self.terminal_type == "powershell":
                cmd = ["powershell", "-Command", self.command]
            elif self.terminal_type == "cmd":
                cmd = ["cmd", "/c", self.command]
            else:
                cmd = [self.terminal_type, "-c", self.command]

            self.process = subprocess.Popen(
                cmd if self.is_shell else self.command,
                shell=not self.is_shell,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.terminal_id = self.process.pid
            self.start_time = time.time()
            self.is_running = True

            self._start_output_readers()

            logger.info(f"Started process {self.terminal_id}: {self.command}")
            return True

        except Exception as e:
            logger.error(f"Error starting process: {e}")
            return False

    def _start_output_readers(self) -> None:
        """Start threads to read stdout and stderr."""
        def read_stdout():
            for line in iter(self.process.stdout.readline, ''):
                self.stdout_buffer.append(line)
            self.process.stdout.close()

        def read_stderr():
            for line in iter(self.process.stderr.readline, ''):
                self.stderr_buffer.append(line)
            self.process.stderr.close()

        stdout_thread = threading.Thread(target=read_stdout)
        stdout_thread.daemon = True
        stdout_thread.start()

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

    def write(self, input_text: str) -> bool:
        """Write input to the process.

        Args:
            input_text (str): The text to write.

        Returns:
            bool: True if the input was written successfully, False otherwise.
        """
        if not self.is_running or not self.process:
            logger.warning("Cannot write to process: Process not running")
            return False

        try:
            if not input_text.endswith('\n'):
                input_text += '\n'

            self.process.stdin.write(input_text)
            self.process.stdin.flush()
            return True

        except Exception as e:
            logger.error(f"Error writing to process: {e}")
            return False

    def read(self) -> str:
        """Read the current output of the process.

        Returns:
            str: The output of the process.
        """
        stdout = ''.join(self.stdout_buffer)
        stderr = ''.join(self.stderr_buffer)

        if stderr:
            return f"{stdout}\n\nErrors:\n{stderr}"
        else:
            return stdout

    def wait(self, timeout: float = None) -> Optional[int]:
        """Wait for the process to complete.

        Args:
            timeout (float, optional): The maximum time to wait in seconds. Defaults to None.

        Returns:
            Optional[int]: The exit code of the process, or None if the timeout expired.
        """
        if not self.is_running or not self.process:
            return self.exit_code

        try:
            exit_code = self.process.wait(timeout=timeout)
            self.exit_code = exit_code
            self.end_time = time.time()
            self.is_running = False
            return exit_code

        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> bool:
        """Kill the process.

        Returns:
            bool: True if the process was killed successfully, False otherwise.
        """
        if not self.is_running or not self.process:
            logger.warning("Cannot kill process: Process not running")
            return False

        try:
            parent = psutil.Process(self.process.pid)

            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            self.process.kill()
            self.is_running = False
            self.end_time = time.time()
            self.exit_code = -1

            logger.info(f"Killed process {self.terminal_id}")
            return True

        except Exception as e:
            logger.error(f"Error killing process: {e}")
            return False

    def get_info(self) -> Dict[str, any]:
        """Get information about the process.

        Returns:
            Dict[str, any]: Information about the process.
        """
        return {
            "terminal_id": self.terminal_id,
            "command": self.command,
            "cwd": self.cwd,
            "terminal_type": self.terminal_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "is_running": self.is_running,
            "runtime": time.time() - self.start_time if self.start_time else None
        }

class TerminalManager:
    """Class for managing terminal processes.

    This class provides methods for executing commands, reading terminal output,
    killing terminal processes, and listing active terminal sessions.
    """

    def __init__(self):
        """Initialize the terminal manager."""
        self.terminals: Dict[int, TerminalProcess] = {}
        self.next_terminal_id = 1
        self.terminal_type = self._detect_terminal_type()

    def _detect_terminal_type(self) -> str:
        """Detect the terminal type.

        Returns:
            str: The terminal type (powershell, cmd, bash, etc.).
        """
        if platform.system() == "Windows":
            try:
                subprocess.run(["powershell", "-Command", "echo 'test'"],
                               capture_output=True, text=True, check=True)
                return "powershell"
            except (subprocess.SubprocessError, FileNotFoundError):
                return "cmd"
        else:
            shell = os.environ.get("SHELL", "")
            if "bash" in shell:
                return "bash"
            elif "zsh" in shell:
                return "zsh"
            elif "fish" in shell:
                return "fish"
            else:
                return "sh"

    def _is_dangerous_command(self, command: str) -> bool:
        """Check if a command is potentially dangerous.

        Args:
            command (str): The command to check.

        Returns:
            bool: True if the command is potentially dangerous, False otherwise.
        """
        dangerous_patterns = [
            r"\brm\s+(-rf?|--recursive)\s+[/\\]",  # rm -rf / or similar
            r"\bdd\s+.*\bof=/dev/(hd|sd|nvme)",    # dd to disk devices
            r"\bmkfs\b",                           # filesystem formatting
            r"\bformat\b",                         # disk formatting
            r"\bdel\s+[/\\].*\s+/[fqs]",           # del with force options
            r"\bfdisk\b",                          # disk partitioning
            r":(){:\|:&};:",                       # fork bomb
            r">(>)?.*/(etc|boot|bin|sbin|dev)",    # redirecting to system directories
            r"\bchmod\s+-[rR].*\s+/",              # removing permissions recursively from /
            r"\bchown\s+-[rR].*\s+/",              # changing ownership recursively from /
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True

        return False

    def _prompt_for_command_confirmation(self, command: str, cwd: str = None) -> bool:
        """Prompt the user to confirm command execution.

        Args:
            command (str): The command to execute.
            cwd (str, optional): The working directory. Defaults to None.

        Returns:
            bool: True if the user confirms, False otherwise.
        """
        print(f"\n{Fore.YELLOW}Command Execution Confirmation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Command:{Style.RESET_ALL} {Fore.WHITE}{command}{Style.RESET_ALL}")

        if cwd:
            print(f"{Fore.CYAN}Working Directory:{Style.RESET_ALL} {Fore.WHITE}{cwd}{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}Do you want to execute this command? (y/n){Style.RESET_ALL}")

        try:
            user_input = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip().lower()
            return user_input in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.RED}Command execution cancelled.{Style.RESET_ALL}")
            return False

    def run_command(self, command: str, cwd: str = None, timeout_seconds: int = 30,
                   check_dangerous: bool = True) -> Dict[str, any]:
        """Run a command in a terminal.

        Args:
            command (str): The command to run.
            cwd (str, optional): The working directory. Defaults to None.
            timeout_seconds (int, optional): The timeout in seconds. Defaults to 30.
            check_dangerous (bool, optional): Whether to check if the command is dangerous. Defaults to True.

        Returns:
            Dict[str, any]: The result of the command execution.
        """
        if check_dangerous and self._is_dangerous_command(command):
            logger.warning(f"Dangerous command detected: {command}")
            return {
                "error": "This command has been blocked for security reasons.",
                "command": command,
                "terminal_id": None,
                "output": None,
                "exit_code": None,
                "success": False
            }

        if not COMMANDS_YOLO:
            if not self._prompt_for_command_confirmation(command, cwd):
                logger.info(f"Command execution cancelled by user: {command}")
                return {
                    "error": "Command execution cancelled by user.",
                    "command": command,
                    "terminal_id": None,
                    "output": None,
                    "exit_code": None,
                    "success": False
                }

        terminal = TerminalProcess(command, cwd)

        if not terminal.start():
            return {
                "error": "Failed to start the command.",
                "command": command,
                "terminal_id": None,
                "output": None,
                "exit_code": None,
                "success": False
            }

        terminal_id = self.next_terminal_id
        self.next_terminal_id += 1
        self.terminals[terminal_id] = terminal
        terminal.terminal_id = terminal_id

        if timeout_seconds > 0:
            exit_code = terminal.wait(timeout=timeout_seconds)

            if exit_code is not None:
                output = terminal.read()

                if not terminal.is_running:
                    del self.terminals[terminal_id]

                return {
                    "terminal_id": terminal_id,
                    "command": command,
                    "output": output,
                    "exit_code": exit_code,
                    "success": exit_code == 0,
                    "runtime": terminal.end_time - terminal.start_time if terminal.end_time else None,
                    "terminal_type": terminal.terminal_type
                }

        return {
            "terminal_id": terminal_id,
            "command": command,
            "message": f"Command is running in terminal {terminal_id}. Use read_terminal to get output.",
            "success": True,
            "is_running": True,
            "terminal_type": terminal.terminal_type
        }

    def read_terminal(self, terminal_id: int, wait: bool = False,
                     max_wait_seconds: int = 0) -> Dict[str, any]:
        """Read output from a terminal.

        Args:
            terminal_id (int): The terminal ID.
            wait (bool, optional): Whether to wait for the command to complete. Defaults to False.
            max_wait_seconds (int, optional): The maximum time to wait in seconds. Defaults to 0.

        Returns:
            Dict[str, any]: The terminal output.
        """
        if terminal_id not in self.terminals:
            return {
                "error": f"Terminal {terminal_id} not found.",
                "terminal_id": terminal_id,
                "output": None,
                "success": False
            }

        terminal = self.terminals[terminal_id]

        if wait and terminal.is_running:
            exit_code = terminal.wait(timeout=max_wait_seconds if max_wait_seconds > 0 else None)

            if exit_code is not None:
                output = terminal.read()

                if not terminal.is_running:
                    del self.terminals[terminal_id]

                return {
                    "terminal_id": terminal_id,
                    "command": terminal.command,
                    "output": output,
                    "exit_code": exit_code,
                    "success": exit_code == 0,
                    "runtime": terminal.end_time - terminal.start_time if terminal.end_time else None,
                    "is_running": False,
                    "terminal_type": terminal.terminal_type
                }

        output = terminal.read()

        return {
            "terminal_id": terminal_id,
            "command": terminal.command,
            "output": output,
            "is_running": terminal.is_running,
            "success": True,
            "terminal_type": terminal.terminal_type
        }

    def write_terminal(self, terminal_id: int, input_text: str) -> Dict[str, any]:
        """Write input to a terminal.

        Args:
            terminal_id (int): The terminal ID.
            input_text (str): The text to write.

        Returns:
            Dict[str, any]: The result of the write operation.
        """
        if terminal_id not in self.terminals:
            return {
                "error": f"Terminal {terminal_id} not found.",
                "terminal_id": terminal_id,
                "success": False
            }

        terminal = self.terminals[terminal_id]

        if not terminal.is_running:
            return {
                "error": f"Terminal {terminal_id} is not running.",
                "terminal_id": terminal_id,
                "success": False
            }

        success = terminal.write(input_text)

        return {
            "terminal_id": terminal_id,
            "command": terminal.command,
            "input": input_text,
            "success": success,
            "terminal_type": terminal.terminal_type
        }

    def kill_terminal(self, terminal_id: int) -> Dict[str, any]:
        """Kill a terminal process.

        Args:
            terminal_id (int): The terminal ID.

        Returns:
            Dict[str, any]: The result of the kill operation.
        """
        if terminal_id not in self.terminals:
            return {
                "error": f"Terminal {terminal_id} not found.",
                "terminal_id": terminal_id,
                "success": False
            }

        terminal = self.terminals[terminal_id]

        if not terminal.is_running:
            return {
                "error": f"Terminal {terminal_id} is not running.",
                "terminal_id": terminal_id,
                "success": False
            }

        success = terminal.kill()

        if success:
            del self.terminals[terminal_id]

        return {
            "terminal_id": terminal_id,
            "command": terminal.command,
            "success": success,
            "terminal_type": terminal.terminal_type
        }

    def list_terminals(self) -> Dict[str, any]:
        """List all active terminal sessions.

        Returns:
            Dict[str, any]: Information about all active terminals.
        """
        terminals_info = []

        for terminal_id, terminal in self.terminals.items():
            if terminal.is_running and terminal.process:
                if terminal.process.poll() is not None:
                    terminal.is_running = False
                    terminal.end_time = time.time()
                    terminal.exit_code = terminal.process.returncode

            terminals_info.append({
                "terminal_id": terminal_id,
                "command": terminal.command,
                "cwd": terminal.cwd,
                "is_running": terminal.is_running,
                "start_time": terminal.start_time,
                "runtime": time.time() - terminal.start_time if terminal.start_time else None,
                "terminal_type": terminal.terminal_type
            })

        return {
            "terminals": terminals_info,
            "count": len(terminals_info),
            "system_terminal_type": self.terminal_type
        }

terminal_manager = TerminalManager()
