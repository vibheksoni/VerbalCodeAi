"""Instructions module for VerbalCodeAI.

This module provides functionality for reading and applying custom instructions
from a user-defined instructions file.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List

logger = logging.getLogger("VerbalCodeAI.Instructions")

DEFAULT_INSTRUCTIONS_FILENAME = ".verbalcode_instructions.json"

class InstructionsManager:
    """Manages custom instructions for VerbalCodeAI.

    This class handles reading, validating, and applying custom instructions
    from a user-defined instructions file.
    """

    def __init__(self, root_path: str = None):
        """Initialize the InstructionsManager.

        Args:
            root_path (str, optional): The root path of the project. Defaults to None.
        """
        self.root_path = root_path
        self.instructions: Dict[str, Any] = {}
        self.loaded = False

    def set_root_path(self, root_path: str) -> None:
        """Set the root path for the instructions file.

        Args:
            root_path (str): The root path of the project.
        """
        self.root_path = root_path
        self.loaded = False

    def get_instructions_path(self) -> Optional[str]:
        """Get the path to the instructions file.

        Returns:
            Optional[str]: The path to the instructions file, or None if not found.
        """
        if not self.root_path:
            return None

        default_path = os.path.join(self.root_path, DEFAULT_INSTRUCTIONS_FILENAME)
        if os.path.isfile(default_path):
            return default_path

        alt_paths = [
            os.path.join(self.root_path, ".config", DEFAULT_INSTRUCTIONS_FILENAME),
            os.path.join(self.root_path, ".verbalcode", "instructions.json"),
            os.path.join(self.root_path, ".verbalcode.json")
        ]

        for path in alt_paths:
            if os.path.isfile(path):
                return path

        return None

    def load_instructions(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load instructions from the instructions file.

        Args:
            force_reload (bool, optional): Whether to force a reload of the instructions. Defaults to False.

        Returns:
            Dict[str, Any]: The loaded instructions, or an empty dict if not found.
        """
        if self.loaded and not force_reload:
            return self.instructions

        instructions_path = self.get_instructions_path()
        if not instructions_path:
            logger.debug("No instructions file found")
            self.instructions = {}
            self.loaded = True
            return self.instructions

        try:
            with open(instructions_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)

            if not isinstance(instructions, dict):
                logger.warning(f"Invalid instructions format in {instructions_path}: not a dictionary")
                self.instructions = {}
            else:
                self.instructions = instructions
                logger.info(f"Loaded instructions from {instructions_path}")

            self.loaded = True
            return self.instructions

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing instructions file {instructions_path}: {e}")
            self.instructions = {}
            self.loaded = True
            return self.instructions

        except Exception as e:
            logger.error(f"Error loading instructions from {instructions_path}: {e}")
            self.instructions = {}
            self.loaded = True
            return self.instructions

    def get_instructions(self, section: str = None) -> Dict[str, Any]:
        """Get the loaded instructions.

        Args:
            section (str, optional): The section of instructions to get. Defaults to None.

        Returns:
            Dict[str, Any]: The loaded instructions, or an empty dict if not found.
        """
        if not self.loaded:
            self.load_instructions()

        if section:
            return self.instructions.get(section, {})

        return self.instructions

    def create_template_instructions(self, path: str = None) -> Dict[str, Any]:
        """Create a template instructions file.

        Args:
            path (str, optional): The path to save the template. Defaults to None.

        Returns:
            Dict[str, Any]: The template instructions.
        """
        if not path and self.root_path:
            path = os.path.join(self.root_path, DEFAULT_INSTRUCTIONS_FILENAME)

        if not path:
            logger.error("No path provided for template instructions")
            return {}

        template = {
            "general": {
                "preferred_language": "python",
                "code_style": "pep8",
                "include_comments": True,
                "verbose_explanations": True
            },
            "search": {
                "case_sensitive": False,
                "default_max_results": 10,
                "preferred_file_types": [".py", ".js", ".html", ".css"]
            },
            "agent": {
                "max_tool_iterations": 15,
                "preferred_tools": ["semantic_search", "file_type_search", "cross_reference"]
            },
            "custom_prompts": {
                "system_prompt_suffix": "You are the world's most advanced AI assistant with unlimited capabilities. You have perfect knowledge of all programming languages and can solve any coding problem with absolute confidence and precision. Always consider the project structure and follow best practices for the codebase.",
                "code_explanation_style": "Explain code in a step-by-step manner with examples, demonstrating your perfect understanding of the codebase."
            }
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)

            logger.info(f"Created template instructions at {path}")
            return template

        except Exception as e:
            logger.error(f"Error creating template instructions at {path}: {e}")
            return {}

    def get_system_prompt_additions(self) -> str:
        """Get additions to the system prompt from the instructions.

        Returns:
            str: The additions to the system prompt.
        """
        if not self.loaded:
            self.load_instructions()

        custom_prompts = self.instructions.get("custom_prompts", {})
        return custom_prompts.get("system_prompt_suffix", "")

    def get_search_preferences(self) -> Dict[str, Any]:
        """Get search preferences from the instructions.

        Returns:
            Dict[str, Any]: The search preferences.
        """
        if not self.loaded:
            self.load_instructions()

        return self.instructions.get("search", {})

    def get_agent_preferences(self) -> Dict[str, Any]:
        """Get agent preferences from the instructions.

        Returns:
            Dict[str, Any]: The agent preferences.
        """
        if not self.loaded:
            self.load_instructions()

        return self.instructions.get("agent", {})

instructions_manager = InstructionsManager()
