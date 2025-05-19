"""Memory management module for VerbalCodeAI.

This module provides enhanced memory management functionality for VerbalCodeAI,
allowing users to create, view, and manage memories.
"""

import os
import logging
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import re

from ..llms import ConversationMemory, conversation_memory

logger = logging.getLogger("VerbalCodeAI.Memory")

class MemoryManager:
    """Manages memories for VerbalCodeAI.

    This class provides enhanced memory management functionality, including
    persistent storage, categorization, and retrieval of memories.
    """

    def __init__(self, root_path: str = None, memory: ConversationMemory = None, indexer: Any = None):
        """Initialize the MemoryManager.

        Args:
            root_path (str, optional): The root path of the project. Defaults to None.
            memory (ConversationMemory, optional): The conversation memory instance. Defaults to None.
            indexer (Any, optional): The FileIndexer instance. Defaults to None.
        """
        self.root_path = root_path
        self.memory = memory or conversation_memory
        self.memory_dir = None
        self.indexer = indexer

        if indexer and hasattr(indexer, 'root_path') and indexer.root_path:
            self.root_path = indexer.root_path
            self.set_memory_dir(indexer.root_path)
        elif root_path:
            self.set_memory_dir(root_path)

    def set_root_path(self, root_path: str) -> None:
        """Set the root path for the memory directory.

        Args:
            root_path (str): The root path of the project.
        """
        self.root_path = root_path
        self.set_memory_dir(root_path)

    def set_memory_dir(self, root_path: str) -> None:
        """Set the memory directory based on the root path.

        Args:
            root_path (str): The root path of the project.
        """
        index_dir = os.path.join(root_path, ".index")
        if not os.path.exists(index_dir):
            try:
                os.makedirs(index_dir)
            except Exception as e:
                logger.error(f"Error creating .index directory: {e}")
                return

        memory_dir = os.path.join(index_dir, "memories")
        if not os.path.exists(memory_dir):
            try:
                os.makedirs(memory_dir)
            except Exception as e:
                logger.error(f"Error creating memories directory: {e}")
                return

        self._migrate_old_memories(root_path, memory_dir)

        self.memory_dir = memory_dir

    def _migrate_old_memories(self, root_path: str, new_memory_dir: str) -> None:
        """Migrate memories from old location (.verbalcode/memories) to new location (.index/memories).

        Args:
            root_path (str): The root path of the project.
            new_memory_dir (str): The new memory directory path.
        """
        old_verbalcode_dir = os.path.join(root_path, ".verbalcode")
        old_memory_dir = os.path.join(old_verbalcode_dir, "memories")
        old_memories_file = os.path.join(old_memory_dir, "memories.json")

        if os.path.exists(old_memories_file):
            logger.info(f"Found memories in old location: {old_memories_file}")

            try:
                with open(old_memories_file, "r", encoding="utf-8") as f:
                    old_memories = json.load(f)

                if not isinstance(old_memories, list):
                    logger.warning(f"Invalid memories format in {old_memories_file}: not a list")
                    return

                new_memories_file = os.path.join(new_memory_dir, "memories.json")

                if os.path.exists(new_memories_file):
                    try:
                        with open(new_memories_file, "r", encoding="utf-8") as f:
                            new_memories = json.load(f)

                        if not isinstance(new_memories, list):
                            logger.warning(f"Invalid memories format in {new_memories_file}: not a list")
                            new_memories = []

                        merged_memories = new_memories.copy()
                        for old_memory in old_memories:
                            if old_memory not in merged_memories:
                                merged_memories.append(old_memory)

                        with open(new_memories_file, "w", encoding="utf-8") as f:
                            json.dump(merged_memories, f, indent=2)

                        logger.info(f"Merged {len(old_memories)} memories from old location with {len(new_memories)} memories in new location")
                    except Exception as e:
                        logger.error(f"Error merging memories: {e}")
                        return
                else:
                    with open(new_memories_file, "w", encoding="utf-8") as f:
                        json.dump(old_memories, f, indent=2)

                    logger.info(f"Migrated {len(old_memories)} memories from old location to new location")

                try:
                    os.rename(old_memories_file, f"{old_memories_file}.migrated")
                    logger.info(f"Renamed old memories file to {old_memories_file}.migrated")
                except Exception as e:
                    logger.error(f"Error renaming old memories file: {e}")

            except Exception as e:
                logger.error(f"Error migrating memories: {e}")
                return

    def save_memories(self) -> bool:
        """Save memories to disk.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.memory_dir or not self.memory:
            logger.error("Cannot save memories: No memory directory or memory instance")
            return False

        try:
            memories_file = os.path.join(self.memory_dir, "memories.json")
            with open(memories_file, "w", encoding="utf-8") as f:
                json.dump(self.memory.memories, f, indent=2)

            logger.info(f"Saved {len(self.memory.memories)} memories to {memories_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            return False

    def load_memories(self) -> bool:
        """Load memories from disk.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.memory_dir or not self.memory:
            logger.error("Cannot load memories: No memory directory or memory instance")
            return False

        try:
            memories_file = os.path.join(self.memory_dir, "memories.json")
            if not os.path.exists(memories_file):
                logger.info(f"No memories file found at {memories_file}")
                return False

            with open(memories_file, "r", encoding="utf-8") as f:
                memories = json.load(f)

            if not isinstance(memories, list):
                logger.warning(f"Invalid memories format in {memories_file}: not a list")
                return False

            self.memory.memories = memories
            logger.info(f"Loaded {len(memories)} memories from {memories_file}")
            return True

        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            return False

    def add_memory(self, content: str, category: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Add a new memory.

        Args:
            content (str): The memory content.
            category (str, optional): The category of the memory. Defaults to None.
            metadata (Dict[str, Any], optional): Additional metadata for the memory. Defaults to None.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.memory:
            logger.error("Cannot add memory: No memory instance")
            return False

        if not content.strip():
            logger.warning("Cannot add empty memory")
            return False

        metadata = metadata or {}
        if category:
            metadata["category"] = category

        try:
            self.memory.add_memory(content, metadata)
            logger.info(f"Added memory: {content[:50]}...")

            self.save_memories()
            return True

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False

    def get_memories(self, category: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get memories, optionally filtered by category.

        Args:
            category (str, optional): The category to filter by. Defaults to None.
            limit (int, optional): The maximum number of memories to return. Defaults to None.

        Returns:
            List[Dict[str, Any]]: The memories.
        """
        if not self.memory:
            logger.error("Cannot get memories: No memory instance")
            return []

        memories = self.memory.memories

        if category:
            memories = [m for m in memories if m.get("metadata", {}).get("category") == category]

        if limit:
            memories = memories[-limit:]

        return memories

    def clear_memories(self, category: str = None) -> bool:
        """Clear memories, optionally filtered by category.

        Args:
            category (str, optional): The category to filter by. Defaults to None.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.memory:
            logger.error("Cannot clear memories: No memory instance")
            return False

        try:
            if category:
                self.memory.memories = [
                    m for m in self.memory.memories
                    if m.get("metadata", {}).get("category") != category
                ]
                logger.info(f"Cleared memories with category '{category}'")
            else:
                self.memory.clear()
                logger.info("Cleared all memories")

            self.save_memories()
            return True

        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories using semantic search.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: The search results.
        """
        if not self.memory:
            logger.error("Cannot search memories: No memory instance")
            return []

        try:
            if hasattr(self.memory, "retrieve_relevant_memories"):
                memory_contents = self.memory.retrieve_relevant_memories(query, max_results=limit)

                results = []
                for content in memory_contents:
                    for memory in self.memory.memories:
                        if memory["content"] == content:
                            results.append(memory)
                            break

                return results

            query_terms = query.lower().split()
            scored_memories = []

            for memory in self.memory.memories:
                content = memory["content"].lower()
                score = sum(1 for term in query_terms if term in content)
                if score > 0:
                    scored_memories.append((score, memory))

            scored_memories.sort(reverse=True)
            return [memory for _, memory in scored_memories[:limit]]

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def format_memories_for_display(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for display.

        Args:
            memories (List[Dict[str, Any]]): The memories to format.

        Returns:
            str: The formatted memories.
        """
        if not memories:
            return "No memories found."

        formatted = []
        for i, memory in enumerate(memories):
            content = memory["content"]
            timestamp = memory.get("timestamp", "")
            category = memory.get("metadata", {}).get("category", "general")

            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass

            formatted.append(f"{i+1}. [{category}] {timestamp}\n   {content}")

        return "\n\n".join(formatted)

memory_manager = MemoryManager()
