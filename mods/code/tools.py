"""Tools module for Agent Mode in VerbalCodeAI.

This module provides a set of tools that can be used by the AI agent to interact
with the codebase, including:
- Embedding-based search
- Regex-based search (grep)
- File reading
- Directory structure visualization
- Code analysis
- Function and class finding
- Git history analysis
- Code explanation
- Import and usage analysis

These tools are designed to be used by the Agent Mode to provide more interactive
and powerful capabilities for exploring and understanding codebases.
"""

import ast
import fnmatch
import json
import logging
import os
import platform
import re
import subprocess
import sys
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytz
import requests
from bs4 import BeautifulSoup

try:
    from googlesearch import search
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "googlesearch-python"])
    from googlesearch import search

try:
    from duckduckgo_search import DDGS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo_search"])
    from duckduckgo_search import DDGS

from .directory import DirectoryEntry, DirectoryParser, EntryType
from .embed import SimilaritySearch
from .instructions import instructions_manager
from .memory import memory_manager
from .terminal import terminal_manager

logger = logging.getLogger("VerbalCodeAI.Tools")

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)

AI_AGENT_BUDDY_MODEL_TEMPERATURE: float = float(os.getenv("AI_AGENT_BUDDY_MODEL_TEMPERATURE", "0.7"))
AI_AGENT_BUDDY_MODEL_MAX_TOKENS: int = int(os.getenv("AI_AGENT_BUDDY_MODEL_MAX_TOKENS", "1024"))

class CodebaseTools:
    """A collection of tools for interacting with the codebase.

    This class provides tools that can be used by the AI agent to search,
    read, and analyze the codebase. It requires an initialized indexer
    to access the indexed codebase.
    """

    def __init__(self, indexer: Any = None):
        """Initialize the CodebaseTools with an indexer.

        Args:
            indexer (Any, optional): The FileIndexer instance used to access the indexed codebase. Defaults to None.
        """
        self.indexer = indexer
        self.similarity_search = None
        self.logger = logging.getLogger("VerbalCodeAI.Tools")

        if self.indexer and hasattr(self.indexer, "similarity_search") and self.indexer.similarity_search:
            self.logger.info("Using shared SimilaritySearch instance from indexer")
            self.similarity_search = self.indexer.similarity_search
        elif self.indexer and self.indexer.index_dir:
            embeddings_dir = os.path.join(self.indexer.index_dir, "embeddings")
            if os.path.exists(embeddings_dir):
                self.logger.warning("Creating new SimilaritySearch instance (not using shared instance)")
                self.similarity_search = SimilaritySearch(embeddings_dir=embeddings_dir)

        if self.indexer and self.indexer.root_path:
            instructions_manager.set_root_path(self.indexer.root_path)

            memory_manager.indexer = self.indexer
            memory_manager.root_path = self.indexer.root_path
            memory_manager.set_memory_dir(self.indexer.root_path)

            memory_manager.load_memories()

    def _collect_file_paths_recursive_from_entry(
        self, entry: DirectoryEntry, current_path_prefix: str, file_list: List[str]
    ):
        """Recursively collects file paths from DirectoryEntry objects.

        Args:
            entry (DirectoryEntry): The directory entry to collect file paths from.
            current_path_prefix (str): The path leading up to the parent of `entry`.
            file_list (List[str]): The list to store the collected file paths.
        """
        entry_full_relative_path = os.path.join(current_path_prefix, entry.name).replace("\\\\", "/")

        if entry.is_file():
            file_list.append(entry_full_relative_path)
        elif entry.is_folder() and entry.children:
            for child_entry in entry.children:
                self._collect_file_paths_recursive_from_entry(child_entry, entry_full_relative_path, file_list)

    def find_closest_file_match(self, user_path: str) -> Optional[str]:
        """Finds the closest match for a given user_path in the indexed files.

        Args:
            user_path (str): The path to search for. Can be a filename, a relative path, or a partial path.

        Returns:
            Optional[str]: The matched relative path or None if no match is found.
        """
        if not self.indexer or not self.indexer.root_path:
            self.logger.warning("find_closest_file_match: No indexer or root_path available.")
            return None

        all_relative_files: List[str] = []

        if hasattr(self.indexer, "file_metadata") and isinstance(self.indexer.file_metadata, dict) and self.indexer.file_metadata:
            self.logger.debug("find_closest_file_match: Using self.indexer.file_metadata.keys()")
            all_relative_files = [p.replace("\\\\", "/") for p in self.indexer.file_metadata.keys()]

        if not all_relative_files and hasattr(self.indexer, "get_all_indexed_relative_files") and callable(
            getattr(self.indexer, "get_all_indexed_relative_files")
        ):
            self.logger.debug("find_closest_file_match: Using self.indexer.get_all_indexed_relative_files()")
            try:
                fetched_paths = self.indexer.get_all_indexed_relative_files()
                if fetched_paths:
                    all_relative_files = [p.replace("\\\\", "/") for p in fetched_paths]
            except Exception as e:
                self.logger.error(f"Error calling self.indexer.get_all_indexed_relative_files: {e}", exc_info=True)

        if not all_relative_files and hasattr(self.indexer, "_get_all_indexable_files") and callable(
            getattr(self.indexer, "_get_all_indexable_files")
        ):
            self.logger.warning("find_closest_file_match: Falling back to self.indexer._get_all_indexable_files.")
            try:
                full_paths: List[str] = self.indexer._get_all_indexable_files()
                temp_relative_files = []
                for fp in full_paths:
                    if isinstance(fp, str):
                        temp_relative_files.append(os.path.relpath(fp, self.indexer.root_path).replace("\\\\", "/"))
                if temp_relative_files:
                    all_relative_files = temp_relative_files
                elif not full_paths:
                    self.logger.warning("_get_all_indexable_files returned no files.")
                elif full_paths and not temp_relative_files:
                    self.logger.error("_get_all_indexable_files provided paths but conversion to relative failed.")
            except Exception as e:
                self.logger.error(f"Error calling or processing self.indexer._get_all_indexable_files: {e}", exc_info=True)

        if not all_relative_files:
            self.logger.warning("find_closest_file_match: Falling back to DirectoryParser.")
            try:
                default_excludes = getattr(self.indexer, "DEFAULT_EXCLUDED_EXTENSIONS", [])
                parser = DirectoryParser(
                    directory_path=self.indexer.root_path,
                    gitignore_path=os.path.join(self.indexer.root_path, ".gitignore")
                    if os.path.exists(os.path.join(self.indexer.root_path, ".gitignore"))
                    else None,
                    parallel=False,
                    hash_files=False,
                    extra_exclude_patterns=default_excludes,
                )
                root_entry = parser.parse()
                parsed_paths: List[str] = []
                if root_entry.is_folder() and root_entry.children:
                    for top_level_child in root_entry.children:
                        self._collect_file_paths_recursive_from_entry(top_level_child, "", parsed_paths)
                if parsed_paths:
                    all_relative_files = parsed_paths
                else:
                    self.logger.warning("DirectoryParser also found no files for find_closest_file_match.")
            except Exception as e_parser:
                self.logger.error(f"Error using DirectoryParser in find_closest_file_match: {e_parser}", exc_info=True)

        if not all_relative_files:
            self.logger.error("find_closest_file_match: Critically failed to obtain list of indexed files.")
            return None

        normalized_user_path = user_path.replace("\\\\", "/").strip("/")

        if normalized_user_path in all_relative_files:
            self.logger.info(f"find_closest_file_match: Exact match for '{user_path}' -> '{normalized_user_path}'")
            return normalized_user_path

        user_basename = os.path.basename(normalized_user_path)
        basename_matches = [p for p in all_relative_files if os.path.basename(p) == user_basename]
        if basename_matches:
            if len(basename_matches) == 1:
                self.logger.info(f"find_closest_file_match: Single basename match for '{user_path}' -> '{basename_matches[0]}'")
                return basename_matches[0]
            else:
                suffix_matches_among_basename = [p for p in basename_matches if p.endswith(normalized_user_path)]
                if len(suffix_matches_among_basename) == 1:
                    self.logger.info(
                        f"find_closest_file_match: Disambiguated basename by suffix for '{user_path}' -> '{suffix_matches_among_basename[0]}'"
                    )
                    return suffix_matches_among_basename[0]
                basename_matches.sort(key=lambda p: (len(p), p))
                self.logger.warning(
                    f"find_closest_file_match: Multiple basename matches for '{user_path}': {basename_matches}. Returning shortest: {basename_matches[0]}"
                )
                return basename_matches[0]

        suffix_matches = [p for p in all_relative_files if p.endswith(normalized_user_path)]
        if suffix_matches:
            shortest_suffix_match = min(suffix_matches, key=len)
            self.logger.info(
                f"find_closest_file_match: Suffix match for '{user_path}' -> '{shortest_suffix_match}' (from {len(suffix_matches)} options)"
            )
            return shortest_suffix_match

        self.logger.info(f"find_closest_file_match: No close match found for '{user_path}'")
        return None

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of the search query to improve search results.

        Args:
            query (str): The original search query

        Returns:
            List[str]: List of query variations including the original
        """
        variations = [query]
        variations.append(query.lower())

        common_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "is",
            "are",
            "that",
            "this",
            "can",
            "will",
        ]
        filtered_query = " ".join([word for word in query.split() if word.lower() not in common_words])
        if filtered_query and filtered_query != query:
            variations.append(filtered_query)

        key_terms = [word for word in query.split() if len(word) > 3]
        if key_terms:
            variations.append(" ".join(key_terms))

        programming_terms = {
            "function": ["def", "method", "func"],
            "method": ["def", "function", "func"],
            "class": ["class", "struct", "interface"],
            "variable": ["var", "let", "const"],
            "authentication": ["auth", "login", "signin", "signup", "register"],
            "registration": ["register", "signup", "create account", "new user"],
            "api": ["endpoint", "route", "handler", "controller"],
            "database": ["db", "storage", "repository", "model"],
            "frontend": ["ui", "interface", "client", "browser"],
            "backend": ["server", "api", "service"],
        }

        query_lower = query.lower()
        for term, alternatives in programming_terms.items():
            if term in query_lower:
                for alt in alternatives:
                    if alt not in query_lower:
                        new_variation = query.replace(term, alt)
                        variations.append(new_variation)

        if "fetch" in query_lower or "api" in query_lower or "request" in query_lower:
            variations.append(query + " http")
            variations.append(query + " axios")
            variations.append(query + " fetch")
            variations.append("api endpoint " + query)

        unique_variations = []
        for v in variations:
            if v and v not in unique_variations:
                unique_variations.append(v)

        self.logger.debug(f"Generated query variations: {unique_variations}")
        return unique_variations

    def embed_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the indexed codebase using vector embeddings.

        Args:
            query (str): The search query string.
            max_results (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of results containing file paths and matching code snippets.
        """
        if not self.indexer or not self.similarity_search:
            self.logger.error("Cannot perform embed_search: No indexer or embeddings available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Performing embedding search for query: {query}")

            query_variations = self._generate_query_variations(query)
            self.logger.info(f"Generated {len(query_variations)} query variations")

            threshold = 0.12

            if len(query_variations) > 1:
                results = self.similarity_search.search_multiple(query_variations, top_k=max_results * 2, threshold=threshold)
            else:
                results = self.similarity_search.search(query, top_k=max_results * 2, threshold=threshold)

            if not results:
                self.logger.warning(f"No results found for query: {query} with threshold {threshold}")
                threshold = 0.08
                self.logger.info(f"Retrying with lower threshold: {threshold}")

                if len(query_variations) > 1:
                    results = self.similarity_search.search_multiple(query_variations, top_k=max_results * 3, threshold=threshold)
                else:
                    results = self.similarity_search.search(query, top_k=max_results * 3, threshold=threshold)

                if not results:
                    self.logger.warning(f"Still no results with lower threshold. Trying individual terms.")
                    individual_terms = [term for term in query.split() if len(term) > 3]
                    if individual_terms:
                        term_results = []
                        for term in individual_terms:
                            term_result = self.similarity_search.search(term, top_k=2, threshold=0.05)
                            if term_result:
                                term_results.extend(term_result)

                        if term_results:
                            results = term_results
                            self.logger.info(f"Found {len(results)} results using individual terms")

            formatted_results = []
            for result in results:
                file_name = result.get("file", "")
                chunk = result.get("chunk", {})
                score = result.get("score", 0.0)

                formatted_result = {
                    "file_path": file_name,
                    "score": score,
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "text": chunk.get("text", ""),
                    "type": chunk.get("type", ""),
                }
                formatted_results.append(formatted_result)

            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            formatted_results = formatted_results[:max_results]

            self.logger.info(f"Found {len(formatted_results)} results for query: {query}")

            return formatted_results
        except Exception as e:
            self.logger.error(f"Error in embed_search: {e}", exc_info=True)
            return [{"error": f"Error performing embedding search: {str(e)}"}]

    def regex_advanced_search(self, search_pattern: str, file_pattern: str = None,
                           case_sensitive: bool = False, whole_word: bool = False,
                           include_context: bool = True, context_lines: int = 2,
                           max_results: int = 100) -> List[Dict[str, Any]]:
        """Perform an advanced regex search with additional options for precision.

        This tool enhances the basic grep functionality with case sensitivity options,
        whole word matching, and context lines.

        Args:
            search_pattern (str): The regex pattern to search for.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.
            case_sensitive (bool, optional): Whether the search is case sensitive. Defaults to False.
            whole_word (bool, optional): Whether to match whole words only. Defaults to False.
            include_context (bool, optional): Whether to include context lines around matches. Defaults to True.
            context_lines (int, optional): Number of context lines to include before and after matches. Defaults to 2.
            max_results (int, optional): Maximum number of results to return. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of matches with file paths, line numbers, and context.
        """
        if not self.indexer:
            self.logger.error("Cannot perform regex_advanced_search: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Performing advanced regex search for pattern: {search_pattern}")
            self.logger.info(f"Options: case_sensitive={case_sensitive}, whole_word={whole_word}, include_context={include_context}")

            if whole_word:
                search_pattern = f"\\b{search_pattern}\\b"

            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(search_pattern, flags)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path

            gitignore_path = os.path.join(root_path, ".gitignore") if os.path.exists(os.path.join(root_path, ".gitignore")) else None
            parser = DirectoryParser(
                directory_path=root_path,
                gitignore_path=gitignore_path,
                parallel=False,
                hash_files=False,
                extra_exclude_patterns=self.indexer.DEFAULT_EXCLUDED_EXTENSIONS,
            )

            root_entry = parser.parse()
            all_files = []

            def collect_files(entry):
                if entry.is_file():
                    if file_pattern and not fnmatch.fnmatch(entry.name, file_pattern):
                        return

                    file_ext = os.path.splitext(entry.name)[1].lower()
                    if file_ext in self.indexer.DEFAULT_EXCLUDED_EXTENSIONS:
                        return

                    all_files.append(entry.path)
                else:
                    for child in entry.children:
                        collect_files(child)

            collect_files(root_entry)

            for file_path in all_files:
                rel_path = os.path.relpath(file_path, root_path)

                try:
                    if not self.indexer._is_text_file(file_path):
                        continue

                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            match_result = {
                                "file_path": rel_path,
                                "line_number": i,
                                "line_text": line.rstrip(),
                                "match": search_pattern
                            }

                            if include_context and context_lines > 0:
                                context_before = []
                                context_after = []

                                start_idx = max(0, i - context_lines - 1)
                                for j in range(start_idx, i - 1):
                                    context_before.append({
                                        "line_number": j + 1,
                                        "line_text": lines[j].rstrip()
                                    })

                                end_idx = min(len(lines), i + context_lines)
                                for j in range(i, end_idx):
                                    context_after.append({
                                        "line_number": j + 1,
                                        "line_text": lines[j].rstrip()
                                    })

                                match_result["context_before"] = context_before
                                match_result["context_after"] = context_after

                            results.append(match_result)

                            if len(results) >= max_results:
                                break

                    if len(results) >= max_results:
                        break

                except (IOError, UnicodeDecodeError) as e:
                    self.logger.debug(f"Could not read file {file_path}: {e}")

            if results:
                file_counts = {}
                for item in results:
                    file_path = item.get("file_path", "unknown")
                    file_counts[file_path] = file_counts.get(file_path, 0) + 1

                summary = {
                    "total_matches": len(results),
                    "files_with_matches": len(file_counts),
                    "pattern": search_pattern,
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word
                }

                results.insert(0, {"summary": summary})

            return results

        except Exception as e:
            self.logger.error(f"Error in regex_advanced_search: {e}", exc_info=True)
            return [{"error": f"Error performing advanced regex search: {str(e)}"}]

    def grep(self, search_pattern: str, file_pattern: str = None) -> List[Dict[str, Any]]:
        """Search the codebase using regex patterns.

        Args:
            search_pattern (str): The regex pattern to search for.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of matches with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform grep: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Performing grep search for pattern: {search_pattern}")

            try:
                regex = re.compile(search_pattern)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path

            gitignore_path = os.path.join(root_path, ".gitignore") if os.path.exists(os.path.join(root_path, ".gitignore")) else None
            parser = DirectoryParser(
                directory_path=root_path,
                gitignore_path=gitignore_path,
                parallel=False,
                hash_files=False,
                extra_exclude_patterns=self.indexer.DEFAULT_EXCLUDED_EXTENSIONS,
            )

            root_entry = parser.parse()

            all_files = []

            def collect_files(entry):
                if entry.is_file():
                    if file_pattern and not fnmatch.fnmatch(entry.name, file_pattern):
                        return

                    file_ext = os.path.splitext(entry.name)[1].lower()
                    if file_ext in self.indexer.DEFAULT_EXCLUDED_EXTENSIONS:
                        return

                    all_files.append(entry.path)
                else:
                    for child in entry.children:
                        collect_files(child)

            collect_files(root_entry)

            for file_path in all_files:
                rel_path = os.path.relpath(file_path, root_path)

                try:
                    if not self.indexer._is_text_file(file_path):
                        continue

                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(
                                    {
                                        "file_path": rel_path,
                                        "line_number": i,
                                        "line_text": line.rstrip(),
                                        "match": search_pattern,
                                    }
                                )
                except (IOError, UnicodeDecodeError) as e:
                    self.logger.debug(f"Could not read file {file_path}: {e}")

            return results[:100]
        except Exception as e:
            self.logger.error(f"Error in grep: {e}", exc_info=True)
            return [{"error": f"Error performing grep search: {str(e)}"}]

    def file_stats(self, path: str) -> Dict[str, Any]:
        """Get statistics about a file in the codebase.

        Args:
            path (str): Path to the file (can be imprecise, partial, or full path).

        Returns:
            Dict[str, Any]: Dictionary with file statistics including full path, line count, size, etc.
        """
        if not self.indexer:
            self.logger.error("Cannot perform file_stats: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting file stats for: {path}")

            resolved_path = self.find_closest_file_match(path)
            if not resolved_path:
                return {"error": f"File not found: {path}"}

            full_path = os.path.join(self.indexer.root_path, resolved_path)
            if not os.path.exists(full_path):
                return {"error": f"File does not exist: {resolved_path}"}

            file_size = os.path.getsize(full_path)
            modification_time = os.path.getmtime(full_path)
            creation_time = os.path.getctime(full_path)

            line_count = 0
            blank_lines = 0
            code_lines = 0
            comment_lines = 0

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line_count += 1
                        stripped = line.strip()
                        if not stripped:
                            blank_lines += 1
                        elif (
                            stripped.startswith("#")
                            or stripped.startswith("//")
                            or stripped.startswith("/*")
                            or stripped.startswith("*")
                        ):
                            comment_lines += 1
                        else:
                            code_lines += 1
            except Exception as e:
                self.logger.warning(f"Error counting lines in {resolved_path}: {e}")

            _, file_extension = os.path.splitext(full_path)

            is_text = self.indexer._is_text_file(full_path) if hasattr(self.indexer, "_is_text_file") else True

            return {
                "file_path": resolved_path,
                "full_path": full_path,
                "file_name": os.path.basename(full_path),
                "directory": os.path.dirname(resolved_path),
                "size_bytes": file_size,
                "size_human": self._format_size(file_size),
                "line_count": line_count,
                "blank_lines": blank_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "extension": file_extension,
                "is_text_file": is_text,
                "modified_time": modification_time,
                "created_time": creation_time,
            }
        except Exception as e:
            self.logger.error(f"Error in file_stats: {e}", exc_info=True)
            return {"error": f"Error getting file stats: {str(e)}"}

    def ask_buddy(self, question: str, context_file_path: str = None, include_project_info: bool = True) -> Dict[str, Any]:
        """Ask the buddy AI model for opinions or suggestions with relevant context.

        This tool provides a second opinion from another AI model. It includes relevant
        context from the codebase to help the buddy AI provide more accurate responses.

        Args:
            question (str): The question or request to ask the buddy AI.
            context_file_path (str, optional): Path to a specific file to include as context. Defaults to None.
            include_project_info (bool, optional): Whether to include project information. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary with the buddy's response.
        """
        try:
            self.logger.info(f"Asking buddy AI: {question}")

            provider = os.environ.get("AI_AGENT_BUDDY_PROVIDER", "")
            api_key = os.environ.get("AI_AGENT_BUDDY_API_KEY", "")
            model = os.environ.get("AI_AGENT_BUDDY_MODEL", "")

            if not provider or not api_key or not model:
                self.logger.warning("Buddy AI environment variables not set")
                return {
                    "error": "Buddy AI not configured. Please set AI_AGENT_BUDDY_PROVIDER, AI_AGENT_BUDDY_API_KEY, and AI_AGENT_BUDDY_MODEL environment variables."
                }

            context_parts = []
            
            if include_project_info and self.indexer:
                try:
                    project_info = self.get_project_description()
                    if project_info and not "error" in project_info:
                        context_parts.append("## Project Information\n")
                        if "name" in project_info and project_info["name"]:
                            context_parts.append(f"Project Name: {project_info['name']}\n")
                        if "description" in project_info and project_info["description"]:
                            context_parts.append(f"Description: {project_info['description']}\n")
                        if "languages" in project_info and project_info["languages"]:
                            context_parts.append(f"Languages: {', '.join(project_info['languages'])}\n")
                        if "frameworks" in project_info and project_info["frameworks"]:
                            context_parts.append(f"Frameworks: {', '.join(project_info['frameworks'])}\n")
                except Exception as e:
                    self.logger.warning(f"Error getting project information: {e}")

            if context_file_path and self.indexer:
                try:
                    resolved_path = self.find_closest_file_match(context_file_path)
                    if resolved_path:
                        full_path = os.path.join(self.indexer.root_path, resolved_path)

                        file_stats = self.file_stats(resolved_path)

                        context_parts.append(f"\n## File Context: {resolved_path}\n")

                        file_description = self.get_file_description(resolved_path)
                        if file_description and not isinstance(file_description, dict):
                            context_parts.append(f"Description: {file_description}\n")

                        if os.path.exists(full_path) and os.path.getsize(full_path) < 50000:
                            try:
                                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                                    content = f.read()

                                if len(content) > 5000:
                                    content = content[:5000] + "\n... (content truncated)"

                                context_parts.append(f"\n```\n{content}\n```\n")
                            except Exception as e:
                                self.logger.warning(f"Error reading file content: {e}")
                        else:
                            try:
                                code_analysis = self.code_analysis(resolved_path)
                                if code_analysis and not "error" in code_analysis:
                                    context_parts.append("\nFile structure summary:\n")

                                    if "functions" in code_analysis and code_analysis["functions"]:
                                        context_parts.append(f"Functions: {', '.join([f['name'] for f in code_analysis['functions']])}\n")

                                    if "classes" in code_analysis and code_analysis["classes"]:
                                        context_parts.append(f"Classes: {', '.join([c['name'] for c in code_analysis['classes']])}\n")

                                    if "imports" in code_analysis and code_analysis["imports"]:
                                        imports_list = []
                                        for imp in code_analysis["imports"][:10]:
                                            if imp.get("type") == "import":
                                                imports_list.append(imp.get("name", ""))
                                            elif imp.get("type") == "from_import":
                                                imports_list.append(f"{imp.get('module', '')}.{imp.get('name', '')}")

                                        if imports_list:
                                            context_parts.append(f"Key imports: {', '.join(imports_list)}\n")
                            except Exception as e:
                                self.logger.warning(f"Error getting code analysis: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing context file: {e}")

            context = "\n".join(context_parts)

            if len(context) > 4000:
                context = context[:4000] + "\n... (context truncated)"

            from .. import llms

            system_message = (
                "You are a helpful AI assistant that provides opinions and suggestions to another AI agent. "
                "You have been provided with context about the project and possibly specific files. "
                "Use this context to provide more accurate and relevant responses. "
                "Be concise and direct in your responses, focusing on practical advice and solutions. "
                "If the context doesn't provide enough information to answer the question, say so clearly."
            )

            user_message = f"Context:\n{context}\n\nQuestion: {question}"

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            response = llms.generate_response(
                messages=messages,
                parse_thinking=False,
                provider=provider,
                api_key=api_key,
                max_tokens=AI_AGENT_BUDDY_MODEL_MAX_TOKENS,
                temperature=AI_AGENT_BUDDY_MODEL_TEMPERATURE,
                model=model,
            )

            return {
                "response": response,
                "provider": provider,
                "model": model,
                "context_included": bool(context_parts),
                "context_file": context_file_path if context_file_path else None
            }

        except Exception as e:
            self.logger.error(f"Error in ask_buddy: {e}", exc_info=True)
            return {"error": f"Error asking buddy AI: {str(e)}"}

    def get_project_languages(self) -> Dict[str, Any]:
        """Get information about programming languages used in the project.

        Returns:
            Dict[str, Any]: Dictionary with language information, including:
                - languages: List of detected programming languages
                - extensions: Dictionary mapping languages to their file extensions
                - primary_language: The most used language in the project
                - extension_counts: Dictionary with counts of each file extension
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_project_languages: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info("Analyzing project languages")

            root_path: str = self.indexer.root_path
            extension_counts: Dict[str, int] = {}
            language_counts: Dict[str, int] = {}
            language_to_extensions: Dict[str, List[str]] = {}

            for root, _, files in os.walk(root_path):
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext:
                        ext = ext.lower()
                        extension_counts[ext] = extension_counts.get(ext, 0) + 1

                        language: str = self._map_extension_to_language(ext)
                        if language != 'Unknown':
                            language_counts[language] = language_counts.get(language, 0) + 1

                            if language not in language_to_extensions:
                                language_to_extensions[language] = []
                            if ext not in language_to_extensions[language]:
                                language_to_extensions[language].append(ext)

            languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
            language_list: List[str] = [lang for lang, _ in languages]

            primary_language: str = language_list[0] if language_list else "Unknown"

            sorted_extensions: Dict[str, int] = {
                k: v for k, v in sorted(extension_counts.items(), key=lambda item: item[1], reverse=True)
            }

            return {
                "languages": language_list,
                "extensions": language_to_extensions,
                "primary_language": primary_language,
                "extension_counts": sorted_extensions,
            }
        except Exception as e:
            self.logger.error(f"Error in get_project_languages: {e}", exc_info=True)
            return {"error": f"Error analyzing project languages: {str(e)}"}

    def get_project_description(self) -> Dict[str, Any]:
        """Get a description of the project using the existing project information.

        Returns:
            Dict[str, Any]: Dictionary with project description, structure, and other relevant information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_project_description: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info("Getting project description from ProjectAnalyzer")

            from .decisions import ProjectAnalyzer

            project_analyzer = ProjectAnalyzer(self.indexer)
            project_info = project_analyzer.load_project_info()

            if not project_info:
                self.logger.info("No existing project info found, collecting new info")
                project_info = project_analyzer.collect_project_info()

            if not project_info:
                self.logger.warning("Failed to get project info from ProjectAnalyzer, using basic info")
                root_path: str = self.indexer.root_path
                project_name: str = os.path.basename(root_path)

                file_count: int = 0
                dir_count: int = 0
                file_types: Dict[str, int] = {}

                for root, dirs, files in os.walk(root_path):
                    dir_count += len(dirs)
                    file_count += len(files)

                    for file in files:
                        _, ext = os.path.splitext(file)
                        if ext:
                            ext = ext.lower()
                            file_types[ext] = file_types.get(ext, 0) + 1

                project_info = {
                    "project_name": project_name,
                    "purpose": "Unknown",
                    "languages": [],
                    "frameworks": [],
                    "structure": "Unknown",
                    "other_notes": "Project information could not be collected automatically.",
                    "file_count": file_count,
                    "directory_count": dir_count,
                    "file_types": {k: v for k, v in sorted(file_types.items(), key=lambda item: item[1], reverse=True)[:10]},
                }

                language_info = self.get_project_languages()
                if "error" not in language_info:
                    project_info["languages"] = language_info.get("languages", [])
                    project_info["primary_language"] = language_info.get("primary_language", "Unknown")
                    project_info["language_extensions"] = language_info.get("extensions", {})

            project_info["root_path"] = self.indexer.root_path

            readme_path: Optional[str] = None
            for readme_name in ["README.md", "README.txt", "README", "readme.md"]:
                potential_path: str = os.path.join(self.indexer.root_path, readme_name)
                if os.path.exists(potential_path):
                    readme_path = potential_path
                    break

            if readme_path:
                try:
                    with open(readme_path, "r", encoding="utf-8", errors="replace") as f:
                        project_info["readme_content"] = f.read()
                except Exception as e:
                    self.logger.warning(f"Error reading README: {e}")

            return project_info

        except Exception as e:
            self.logger.error(f"Error in get_project_description: {e}", exc_info=True)
            return {"error": f"Error getting project description: {str(e)}"}

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes (int): Size in bytes.

        Returns:
            str: Human-readable size string.
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024 or unit == "GB":
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} GB"

    def _map_extension_to_language(self, extension: str) -> str:
        """Map a file extension to a programming language.

        Args:
            extension (str): The file extension (with dot, e.g., '.py')

        Returns:
            str: The corresponding programming language name
        """
        extension_map = {
            # Python
            '.py': 'Python',
            '.pyw': 'Python',
            '.pyx': 'Python',
            '.pyi': 'Python',
            '.ipynb': 'Python',

            # JavaScript/TypeScript
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.mjs': 'JavaScript',
            '.cjs': 'JavaScript',

            # Web
            '.html': 'HTML',
            '.htm': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'SASS',
            '.less': 'LESS',

            # Java
            '.java': 'Java',
            '.class': 'Java',
            '.jar': 'Java',

            # C/C++
            '.c': 'C',
            '.h': 'C',
            '.cpp': 'C++',
            '.cc': 'C++',
            '.cxx': 'C++',
            '.hpp': 'C++',
            '.hxx': 'C++',

            # C#
            '.cs': 'C#',
            '.csproj': 'C#',

            # Ruby
            '.rb': 'Ruby',
            '.erb': 'Ruby',
            '.gemspec': 'Ruby',

            # PHP
            '.php': 'PHP',

            # Go
            '.go': 'Go',

            # Rust
            '.rs': 'Rust',

            # Swift
            '.swift': 'Swift',

            # Kotlin
            '.kt': 'Kotlin',
            '.kts': 'Kotlin',

            # Shell/Bash
            '.sh': 'Shell',
            '.bash': 'Shell',
            '.zsh': 'Shell',

            # Data formats
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.csv': 'CSV',
            '.md': 'Markdown',
            '.markdown': 'Markdown',

            # Configuration
            '.ini': 'INI',
            '.cfg': 'Config',
            '.conf': 'Config',
            '.config': 'Config',
            '.env': 'Environment',

            # Other
            '.sql': 'SQL',
            '.r': 'R',
            '.dart': 'Dart',
            '.lua': 'Lua',
            '.pl': 'Perl',
            '.pm': 'Perl',
            '.hs': 'Haskell',
            '.elm': 'Elm',
            '.ex': 'Elixir',
            '.exs': 'Elixir',
            '.erl': 'Erlang',
            '.fs': 'F#',
            '.fsx': 'F#',
            '.clj': 'Clojure',
            '.scala': 'Scala',
            '.groovy': 'Groovy',
            '.jl': 'Julia',
        }

        return extension_map.get(extension.lower(), 'Unknown')

    def read_file(self, path: str, line_start: int = None, line_end: int = None) -> Dict[str, Any]:
        """Read contents of a specific file.

        Args:
            path (str): Path to the file relative to the project root (can be imprecise).
            line_start (int, optional): Optional starting line number (1-based, inclusive). Defaults to None.
            line_end (int, optional): Optional ending line number (1-based, inclusive). Defaults to None.

        Returns:
            Dict[str, Any]: File content as string (full file or specified lines).
        """
        if not self.indexer:
            self.logger.error("Cannot perform read_file: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Attempting to read file (user input): {path} (lines {line_start}-{line_end})")

            resolved_relative_path = self.find_closest_file_match(path)

            if not resolved_relative_path:
                self.logger.error(f"File not found or no close match for user input: '{path}'")
                return {"error": f"File not found or no close match for: {path}"}

            self.logger.info(f"Resolved path for '{path}' to: '{resolved_relative_path}'")

            full_path = os.path.join(self.indexer.root_path, resolved_relative_path)

            if not os.path.isfile(full_path):
                self.logger.error(f"Resolved file path '{resolved_relative_path}' ({full_path}) does not exist or is not a file.")
                return {"error": f"Resolved file '{resolved_relative_path}' is not accessible."}

            file_size = os.path.getsize(full_path)
            if file_size > 1024 * 1024 and line_start is None and line_end is None:
                self.logger.warning(
                    f"File {resolved_relative_path} is large ({file_size} bytes). Consider using file_stats first and then reading in chunks."
                )

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    if line_start is None and line_end is None:
                        content = f.read()
                        lines_content = content.split("\n")
                        total_lines = len(lines_content)

                        chunk_suggestion = None
                        if total_lines > 200:
                            chunk_suggestion = {
                                "suggestion": f"This file has {total_lines} lines. Consider reading it in chunks of 100-200 lines.",
                                "recommended_chunks": [
                                    {"start": 1, "end": 200},
                                    {"start": 201, "end": min(400, total_lines)},
                                    {"start": 401, "end": min(600, total_lines)},
                                ],
                            }

                        return {
                            "file_path": resolved_relative_path,
                            "content": content,
                            "total_lines": total_lines,
                            "line_start": 1,
                            "line_end": total_lines,
                            "chunk_suggestion": chunk_suggestion,
                        }
                    else:
                        lines_content_all = f.readlines()

                        parsed_line_start = None
                        if line_start is not None:
                            try:
                                parsed_line_start = int(line_start)
                            except ValueError:
                                self.logger.warning(f"Invalid line_start '{line_start}', using default.")

                        parsed_line_end = None
                        if line_end is not None:
                            try:
                                parsed_line_end = int(line_end)
                            except ValueError:
                                self.logger.warning(f"Invalid line_end '{line_end}', using default.")

                        actual_line_start = parsed_line_start if parsed_line_start is not None else 1
                        actual_line_end = parsed_line_end if parsed_line_end is not None and parsed_line_end != -1 else len(lines_content_all)

                        actual_line_start = max(1, actual_line_start)
                        actual_line_end = min(len(lines_content_all), actual_line_end)

                        if actual_line_start > actual_line_end:
                            self.logger.warning(
                                f"Corrected line_start {actual_line_start} is greater than corrected line_end {actual_line_end} for file {resolved_relative_path}. Returning empty content for range."
                            )
                            selected_lines_content = []
                        else:
                            selected_lines_content = lines_content_all[actual_line_start - 1 : actual_line_end]

                        content = "".join(selected_lines_content)

                        next_chunk_suggestion = None
                        if actual_line_end < len(lines_content_all):
                            next_start = actual_line_end + 1
                            chunk_size = actual_line_end - actual_line_start + 1
                            next_end = min(next_start + chunk_size - 1, len(lines_content_all))
                            next_chunk_suggestion = {
                                "next_start": next_start,
                                "next_end": next_end,
                                "remaining_lines": len(lines_content_all) - actual_line_end,
                            }

                        return {
                            "file_path": resolved_relative_path,
                            "content": content,
                            "line_start": actual_line_start,
                            "line_end": actual_line_end,
                            "total_lines": len(lines_content_all),
                            "next_chunk": next_chunk_suggestion,
                        }
            except (IOError, UnicodeDecodeError) as e:
                self.logger.error(f"Could not read file '{resolved_relative_path}': {str(e)}", exc_info=True)
                return {"error": f"Could not read file {resolved_relative_path}: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Error in read_file for path '{path}': {e}", exc_info=True)
            return {"error": f"Error reading file '{path}': {str(e)}"}


    def directory_tree(self, max_depth: int = None) -> Dict[str, Any]:
        """Generate a directory structure of the indexed project.

        Args:
            max_depth (int, optional): Optional maximum depth to display. Defaults to None.

        Returns:
            Dict[str, Any]: String representation of the directory tree.
        """
        if not self.indexer:
            self.logger.error("Cannot generate directory tree: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Generating directory tree with max_depth={max_depth}")

            root_path = self.indexer.root_path
            parser = DirectoryParser(
                directory_path=root_path,
                gitignore_path=os.path.join(root_path, ".gitignore")
                if os.path.exists(os.path.join(root_path, ".gitignore"))
                else None,
                parallel=True,
                hash_files=False,
                extra_exclude_patterns=self.indexer.DEFAULT_EXCLUDED_EXTENSIONS,
            )

            root_entry = parser.parse()

            tree_lines = []
            self._format_directory_tree(root_entry, "", True, tree_lines, 0, max_depth)

            tree_str = "\n".join(tree_lines)

            return {
                "tree": tree_str,
                "root_path": root_path,
                "file_count": self._count_files(root_entry),
                "dir_count": self._count_dirs(root_entry),
            }
        except Exception as e:
            self.logger.error(f"Error generating directory tree: {e}", exc_info=True)
            return {"error": f"Error generating directory tree: {str(e)}"}

    def _format_directory_tree(
        self,
        entry: DirectoryEntry,
        prefix: str,
        is_last: bool,
        lines: List[str],
        current_depth: int,
        max_depth: Optional[int],
    ) -> None:
        """Format a directory entry as a tree structure.

        Args:
            entry (DirectoryEntry): The DirectoryEntry to format.
            prefix (str): The prefix to use for this entry.
            is_last (bool): Whether this is the last entry in its parent.
            lines (List[str]): The list of lines to append to.
            current_depth (int): The current depth in the tree.
            max_depth (Optional[int]): The maximum depth to display.
        """
        if max_depth is not None and current_depth > max_depth:
            return

        if is_last:
            lines.append(f"{prefix}└── {entry.name}")
            new_prefix = f"{prefix}    "
        else:
            lines.append(f"{prefix}├── {entry.name}")
            new_prefix = f"{prefix}│   "

        if entry.is_folder() and entry.children:
            sorted_children = sorted(
                entry.children,
                key=lambda e: (e.entry_type != EntryType.FOLDER, e.name.lower()),
            )

            for i, child in enumerate(sorted_children):
                is_last_child = i == len(sorted_children) - 1
                self._format_directory_tree(
                    child, new_prefix, is_last_child, lines, current_depth + 1, max_depth
                )

    def _count_files(self, entry: DirectoryEntry) -> int:
        """Count the number of files in a directory tree.

        Args:
            entry (DirectoryEntry): The root DirectoryEntry.

        Returns:
            int: The number of files in the tree.
        """
        if entry.is_file():
            return 1

        return sum(self._count_files(child) for child in entry.children)

    def _count_dirs(self, entry: DirectoryEntry) -> int:
        """Count the number of directories in a directory tree.

        Args:
            entry (DirectoryEntry): The root DirectoryEntry.

        Returns:
            int: The number of directories in the tree.
        """
        if entry.is_file():
            return 0

        return 1 + sum(self._count_dirs(child) for child in entry.children)

    def find_functions(self, pattern: str, file_pattern: str = None) -> List[Dict[str, Any]]:
        """Find function definitions matching a pattern.

        Args:
            pattern (str): The regex pattern to match function names.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of function definitions with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform find_functions: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Finding functions matching pattern: {pattern}")
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path
            language_info = self.get_project_languages()
            python_extensions = ['.py']

            if "error" not in language_info and "extensions" in language_info:
                if "Python" in language_info["extensions"]:
                    python_extensions = language_info["extensions"]["Python"]

            for root, _, files in os.walk(root_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() not in python_extensions:
                        continue

                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, root_path)

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            file_content = f.read()

                        try:
                            tree = ast.parse(file_content)

                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef) and regex.search(node.name):
                                    args = []
                                    for arg in node.args.args:
                                        args.append(arg.arg)

                                    docstring = ast.get_docstring(node)

                                    results.append(
                                        {
                                            "file_path": rel_path,
                                            "line_number": node.lineno,
                                            "function_name": node.name,
                                            "arguments": args,
                                            "docstring": docstring,
                                        }
                                    )
                        except SyntaxError:
                            self.logger.debug(f"Syntax error in {file_path}, skipping")
                    except (IOError, UnicodeDecodeError) as e:
                        self.logger.debug(f"Could not read file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error in find_functions: {e}", exc_info=True)
            return [{"error": f"Error finding functions: {str(e)}"}]

    def find_classes(self, pattern: str, file_pattern: str = None) -> List[Dict[str, Any]]:
        """Find class definitions matching a pattern.

        Args:
            pattern (str): The regex pattern to match class names.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of class definitions with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform find_classes: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Finding classes matching pattern: {pattern}")

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path

            language_info = self.get_project_languages()
            python_extensions = ['.py']

            if "error" not in language_info and "extensions" in language_info:
                if "Python" in language_info["extensions"]:
                    python_extensions = language_info["extensions"]["Python"]

            for root, _, files in os.walk(root_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() not in python_extensions:
                        continue

                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, root_path)

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            file_content = f.read()

                        try:
                            tree = ast.parse(file_content)

                            for node in ast.walk(tree):
                                if isinstance(node, ast.ClassDef) and regex.search(node.name):
                                    bases = []
                                    for base in node.bases:
                                        if isinstance(base, ast.Name):
                                            bases.append(base.id)
                                        elif isinstance(base, ast.Attribute):
                                            bases.append(f"{base.value.id}.{base.attr}")

                                    docstring = ast.get_docstring(node)

                                    methods = []
                                    for child_node in ast.iter_child_nodes(node):
                                        if isinstance(child_node, ast.FunctionDef):
                                            methods.append(child_node.name)

                                    results.append(
                                        {
                                            "file_path": rel_path,
                                            "line_number": node.lineno,
                                            "class_name": node.name,
                                            "base_classes": bases,
                                            "methods": methods,
                                            "docstring": docstring,
                                        }
                                    )
                        except SyntaxError:
                            self.logger.debug(f"Syntax error in {file_path}, skipping")
                    except (IOError, UnicodeDecodeError) as e:
                        self.logger.debug(f"Could not read file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error in find_classes: {e}", exc_info=True)
            return [{"error": f"Error finding classes: {str(e)}"}]

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a user-provided path to a valid path in the indexed codebase.

        Args:
            path (str): The path to resolve (can be imprecise, partial, or full path).

        Returns:
            Optional[str]: The resolved relative path or None if no match is found.
        """
        if not self.indexer or not self.indexer.root_path:
            self.logger.warning("_resolve_path: No indexer or root_path available.")
            return None

        resolved_path = self.find_closest_file_match(path)
        if resolved_path:
            return resolved_path

        normalized_path = path.replace("\\\\", "/").strip("/")
        root_path = self.indexer.root_path
        full_path = os.path.join(root_path, normalized_path)

        if os.path.exists(full_path):
            return normalized_path

        self.logger.warning(f"_resolve_path: Could not resolve path '{path}'")
        return None

    def cross_reference(self, symbol: str, reference_type: str = "all", max_results: int = 20) -> Dict[str, Any]:
        """Find all references and definitions of a symbol across the codebase.

        This tool tracks where variables, functions, or classes are defined and used,
        providing a comprehensive view of symbol relationships.

        Args:
            symbol (str): The symbol name to cross-reference (function, class, variable).
            reference_type (str, optional): Type of references to find - "all", "definition",
                                          "usage", "import", "inheritance". Defaults to "all".
            max_results (int, optional): Maximum number of results to return. Defaults to 20.

        Returns:
            Dict[str, Any]: Dictionary with definitions and references of the symbol.
        """
        if not self.indexer:
            self.logger.error("Cannot perform cross_reference: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Cross-referencing symbol: {symbol} with type: {reference_type}")

            result = {
                "symbol": symbol,
                "definitions": [],
                "usages": [],
                "imports": [],
                "inheritance": [],
                "related_symbols": []
            }

            if reference_type in ["all", "definition"]:
                class_defs = self.find_classes(f"^{re.escape(symbol)}$")
                if not isinstance(class_defs, list) or "error" in class_defs:
                    class_defs = []

                for class_def in class_defs:
                    if "error" not in class_def:
                        result["definitions"].append({
                            "type": "class",
                            "file_path": class_def.get("file_path", ""),
                            "line_number": class_def.get("line_number", 0),
                            "name": class_def.get("class_name", ""),
                            "base_classes": class_def.get("base_classes", []),
                            "methods": class_def.get("methods", [])
                        })

                func_defs = self.find_functions(f"^{re.escape(symbol)}$")
                if not isinstance(func_defs, list) or "error" in func_defs:
                    func_defs = []

                for func_def in func_defs:
                    if "error" not in func_def:
                        result["definitions"].append({
                            "type": "function",
                            "file_path": func_def.get("file_path", ""),
                            "line_number": func_def.get("line_number", 0),
                            "name": func_def.get("function_name", ""),
                            "arguments": func_def.get("arguments", [])
                        })

                var_pattern = f"(^|\\s+){re.escape(symbol)}\\s*="
                var_defs = self.grep(var_pattern)
                if isinstance(var_defs, list) and not any("error" in item for item in var_defs):
                    for var_def in var_defs[:max_results//2]:
                        result["definitions"].append({
                            "type": "variable",
                            "file_path": var_def.get("file_path", ""),
                            "line_number": var_def.get("line_number", 0),
                            "line_text": var_def.get("line_text", "")
                        })

            if reference_type in ["all", "usage"]:
                usages = self.find_usage(symbol)
                if isinstance(usages, list) and not any("error" in item for item in usages):
                    for usage in usages[:max_results]:
                        if usage.get("file_path") and usage.get("line_number"):
                            result["usages"].append({
                                "file_path": usage.get("file_path", ""),
                                "line_number": usage.get("line_number", 0),
                                "line_text": usage.get("line_text", ""),
                                "context": usage.get("context", "")
                            })

            if reference_type in ["all", "import"]:
                imports = self.search_imports(symbol)
                if isinstance(imports, list) and not any("error" in item for item in imports):
                    for imp in imports[:max_results]:
                        result["imports"].append({
                            "file_path": imp.get("file_path", ""),
                            "line_number": imp.get("line_number", 0),
                            "import_statement": imp.get("import_statement", "")
                        })

            if reference_type in ["all", "inheritance"] and len(result["definitions"]) > 0:
                for definition in result["definitions"]:
                    if definition["type"] == "class":
                        inheritors = self.find_classes(f".*")
                        if isinstance(inheritors, list) and not any("error" in item for item in inheritors):
                            for inheritor in inheritors:
                                if symbol in inheritor.get("base_classes", []):
                                    result["inheritance"].append({
                                        "type": "child",
                                        "file_path": inheritor.get("file_path", ""),
                                        "line_number": inheritor.get("line_number", 0),
                                        "class_name": inheritor.get("class_name", ""),
                                        "relationship": f"{inheritor.get('class_name', '')} inherits from {symbol}"
                                    })

                related_pattern = f"{re.escape(symbol)}[A-Z]\\w*|[A-Z]\\w*{re.escape(symbol)}"
                related_classes = self.find_classes(related_pattern)
                if isinstance(related_classes, list) and not any("error" in item for item in related_classes):
                    for related in related_classes[:max_results//2]:
                        if related.get("class_name") != symbol:
                            result["related_symbols"].append({
                                "type": "class",
                                "name": related.get("class_name", ""),
                                "file_path": related.get("file_path", ""),
                                "line_number": related.get("line_number", 0)
                            })

                related_funcs = self.find_functions(related_pattern)
                if isinstance(related_funcs, list) and not any("error" in item for item in related_funcs):
                    for related in related_funcs[:max_results//2]:
                        if related.get("function_name") != symbol:
                            result["related_symbols"].append({
                                "type": "function",
                                "name": related.get("function_name", ""),
                                "file_path": related.get("file_path", ""),
                                "line_number": related.get("line_number", 0)
                            })

            result["summary"] = {
                "total_definitions": len(result["definitions"]),
                "total_usages": len(result["usages"]),
                "total_imports": len(result["imports"]),
                "total_inheritance": len(result["inheritance"]),
                "total_related": len(result["related_symbols"]),
                "files_with_references": len(set([item.get("file_path") for item in result["definitions"] + result["usages"] + result["imports"] + result["inheritance"]]))
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in cross_reference: {e}", exc_info=True)
            return {"error": f"Error performing cross-reference: {str(e)}"}

    def version_control_search(self, search_pattern: str, search_type: str = "commit_message",
                            max_results: int = 20, author: str = None,
                            date_range: str = None) -> Dict[str, Any]:
        """Search across git commit history for specific patterns.

        This tool integrates with version control systems (Git) to search across commit history,
        providing historical context for code changes.

        Args:
            search_pattern (str): The pattern to search for.
            search_type (str, optional): Type of search - "commit_message", "code_change",
                                       "author", "file_path". Defaults to "commit_message".
            max_results (int, optional): Maximum number of results to return. Defaults to 20.
            author (str, optional): Filter by commit author. Defaults to None.
            date_range (str, optional): Filter by date range (e.g., "2023-01-01..2023-12-31"). Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary with search results from version control history.
        """
        if not self.indexer:
            self.logger.error("Cannot perform version_control_search: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Performing version control search for pattern: {search_pattern}")
            self.logger.info(f"Search type: {search_type}, Author: {author}, Date range: {date_range}")

            root_path = self.indexer.root_path

            if search_type == "commit_message":
                cmd = [
                    "git",
                    "-C",
                    root_path,
                    "log",
                    "--pretty=format:%H|%an|%ad|%s",
                    "--date=iso",
                ]

                if author:
                    cmd.append(f"--author={author}")

                if date_range:
                    cmd.append(f"--since={date_range.split('..')[0]}" if ".." in date_range else f"--since={date_range}")
                    if ".." in date_range:
                        cmd.append(f"--until={date_range.split('..')[1]}")

                cmd.extend(["-i", "--grep", search_pattern])

                cmd.extend(["-n", str(max_results)])

            elif search_type == "code_change":
                cmd = [
                    "git",
                    "-C",
                    root_path,
                    "log",
                    "--pretty=format:%H|%an|%ad|%s",
                    "--date=iso",
                    "-p",
                ]

                if author:
                    cmd.append(f"--author={author}")

                if date_range:
                    cmd.append(f"--since={date_range.split('..')[0]}" if ".." in date_range else f"--since={date_range}")
                    if ".." in date_range:
                        cmd.append(f"--until={date_range.split('..')[1]}")

                cmd.extend(["-G", search_pattern])

                cmd.extend(["-n", str(max_results)])

            elif search_type == "file_path":
                cmd = [
                    "git",
                    "-C",
                    root_path,
                    "log",
                    "--pretty=format:%H|%an|%ad|%s",
                    "--date=iso",
                    "--name-only",
                ]

                if author:
                    cmd.append(f"--author={author}")

                if date_range:
                    cmd.append(f"--since={date_range.split('..')[0]}" if ".." in date_range else f"--since={date_range}")
                    if ".." in date_range:
                        cmd.append(f"--until={date_range.split('..')[1]}")

                cmd.append("--")
                cmd.append(search_pattern)

                cmd.extend(["-n", str(max_results)])

            else:
                return {"error": f"Invalid search type: {search_type}"}

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout.strip()

                if not output:
                    return {
                        "search_pattern": search_pattern,
                        "search_type": search_type,
                        "author_filter": author,
                        "date_range": date_range,
                        "results": [],
                        "total_results": 0,
                        "message": "No matching commits found."
                    }

                commits = []

                if search_type == "commit_message":
                    for line in output.split("\n"):
                        if not line.strip():
                            continue

                        parts = line.split("|", 3)
                        if len(parts) == 4:
                            commit_hash, author_name, date, message = parts
                            commits.append({
                                "hash": commit_hash,
                                "author": author_name,
                                "date": date,
                                "message": message,
                                "matches": [{"text": message, "type": "commit_message"}]
                            })

                elif search_type == "code_change":
                    current_commit = None
                    current_matches = []

                    for line in output.split("\n"):
                        if line.startswith("commit "):
                            if current_commit:
                                current_commit["matches"] = current_matches
                                commits.append(current_commit)
                                current_matches = []

                            commit_hash = line.split(" ")[1]
                            current_commit = {
                                "hash": commit_hash,
                                "author": "",
                                "date": "",
                                "message": ""
                            }
                        elif line.startswith("Author: "):
                            if current_commit:
                                current_commit["author"] = line[8:].strip()
                        elif line.startswith("Date: "):
                            if current_commit:
                                current_commit["date"] = line[6:].strip()
                        elif line.startswith("    ") and current_commit and not current_commit["message"]:
                            current_commit["message"] = line.strip()
                        elif line.startswith("+") and search_pattern.lower() in line.lower():
                            current_matches.append({
                                "text": line[1:].strip(),
                                "type": "added_code"
                            })
                        elif line.startswith("-") and search_pattern.lower() in line.lower():
                            current_matches.append({
                                "text": line[1:].strip(),
                                "type": "removed_code"
                            })

                    if current_commit:
                        current_commit["matches"] = current_matches
                        commits.append(current_commit)

                elif search_type == "file_path":
                    current_commit = None
                    current_files = []

                    for line in output.split("\n"):
                        if "|" in line and len(line.split("|")) == 4:
                            if current_commit:
                                current_commit["matches"] = [{"file": file, "type": "file_path"} for file in current_files]
                                commits.append(current_commit)
                                current_files = []

                            parts = line.split("|", 3)
                            commit_hash, author_name, date, message = parts
                            current_commit = {
                                "hash": commit_hash,
                                "author": author_name,
                                "date": date,
                                "message": message
                            }
                        elif line.strip() and current_commit and search_pattern.lower() in line.lower():
                            current_files.append(line.strip())

                    if current_commit:
                        current_commit["matches"] = [{"file": file, "type": "file_path"} for file in current_files]
                        if current_files:
                            commits.append(current_commit)

                return {
                    "search_pattern": search_pattern,
                    "search_type": search_type,
                    "author_filter": author,
                    "date_range": date_range,
                    "results": commits,
                    "total_results": len(commits)
                }

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Git command failed: {e}")
                return {"error": f"Git command failed: {e.stderr}"}

        except Exception as e:
            self.logger.error(f"Error in version_control_search: {e}", exc_info=True)
            return {"error": f"Error performing version control search: {str(e)}"}

    def git_history(self, path: str, max_commits: int = 10) -> Dict[str, Any]:
        """Get git history for a file or directory.

        Args:
            path (str): Path to the file or directory.
            max_commits (int, optional): Maximum number of commits to return. Defaults to 10.

        Returns:
            Dict[str, Any]: Dictionary with git history information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform git_history: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting git history for path: {path}")

            resolved_path = self._resolve_path(path)
            if not resolved_path:
                return {"error": f"Could not resolve path: {path}"}

            root_path = self.indexer.root_path
            full_path = os.path.join(root_path, resolved_path)

            if not os.path.exists(full_path):
                return {"error": f"Path does not exist: {resolved_path}"}

            try:
                cmd = [
                    "git",
                    "-C",
                    root_path,
                    "log",
                    "--pretty=format:%H|%an|%ad|%s",
                    "--date=iso",
                    "-n",
                    str(max_commits),
                    "--",
                    resolved_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                if not result.stdout.strip():
                    return {"error": f"No git history found for path: {resolved_path}"}

                commits = []
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue

                    parts = line.split("|", 3)
                    if len(parts) == 4:
                        commit_hash, author, date, message = parts
                        commits.append(
                            {
                                "hash": commit_hash,
                                "author": author,
                                "date": date,
                                "message": message,
                            }
                        )

                return {
                    "path": resolved_path,
                    "commits": commits,
                    "total_commits": len(commits),
                }
            except subprocess.CalledProcessError as e:
                return {"error": f"Git command failed: {e.stderr}"}
            except FileNotFoundError:
                return {
                    "error": "Git command not found. Make sure git is installed and available in PATH."
                }
        except Exception as e:
            self.logger.error(f"Error in git_history: {e}", exc_info=True)
            return {"error": f"Error getting git history: {str(e)}"}

    def file_type_search(self, search_pattern: str, file_extensions: List[str],
                       case_sensitive: bool = False, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for a pattern in specific file types.

        This tool optimizes searches by focusing on specific file types or languages,
        reducing noise in search results.

        Args:
            search_pattern (str): The pattern to search for.
            file_extensions (List[str]): List of file extensions to search in (e.g., [".py", ".js"]).
            case_sensitive (bool, optional): Whether the search is case sensitive. Defaults to False.
            max_results (int, optional): Maximum number of results to return. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of matches with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform file_type_search: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Performing file type search for pattern: {search_pattern}")
            self.logger.info(f"File extensions: {file_extensions}")

            normalized_extensions = []
            for ext in file_extensions:
                if not ext.startswith("."):
                    ext = f".{ext}"
                normalized_extensions.append(ext.lower())

            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(search_pattern, flags)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path

            language_info = self.get_project_languages()
            language_extensions = {}

            if "error" not in language_info and "extensions" in language_info:
                language_extensions = language_info["extensions"]

            extension_to_language = {}
            for lang, exts in language_extensions.items():
                for ext in exts:
                    extension_to_language[ext.lower()] = lang

            common_extensions = {
                ".py": "Python",
                ".js": "JavaScript",
                ".ts": "TypeScript",
                ".html": "HTML",
                ".css": "CSS",
                ".java": "Java",
                ".c": "C",
                ".cpp": "C++",
                ".h": "C/C++ Header",
                ".go": "Go",
                ".rs": "Rust",
                ".rb": "Ruby",
                ".php": "PHP",
                ".sh": "Shell",
                ".bat": "Batch",
                ".ps1": "PowerShell",
                ".json": "JSON",
                ".xml": "XML",
                ".yaml": "YAML",
                ".yml": "YAML",
                ".md": "Markdown",
                ".txt": "Text",
                ".csv": "CSV",
                ".sql": "SQL"
            }

            for ext, lang in common_extensions.items():
                if ext not in extension_to_language:
                    extension_to_language[ext] = lang

            for root, _, files in os.walk(root_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower()

                    if ext not in normalized_extensions:
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, root_path)

                    try:
                        if not self.indexer._is_text_file(file_path):
                            continue

                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    language = extension_to_language.get(ext, "Unknown")

                                    results.append({
                                        "file_path": rel_path,
                                        "line_number": i,
                                        "line_text": line.strip(),
                                        "file_extension": ext,
                                        "language": language
                                    })

                                    if len(results) >= max_results:
                                        break

                        if len(results) >= max_results:
                            break

                    except (IOError, UnicodeDecodeError) as e:
                        self.logger.debug(f"Could not read file {file_path}: {e}")

                if len(results) >= max_results:
                    break

            if results:
                extension_counts = {}
                language_counts = {}
                file_counts = {}

                for item in results:
                    ext = item.get("file_extension", "unknown")
                    lang = item.get("language", "Unknown")
                    file_path = item.get("file_path", "unknown")

                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    file_counts[file_path] = file_counts.get(file_path, 0) + 1

                summary = {
                    "total_matches": len(results),
                    "files_with_matches": len(file_counts),
                    "extensions_searched": normalized_extensions,
                    "extension_matches": extension_counts,
                    "language_matches": language_counts,
                    "pattern": search_pattern,
                    "case_sensitive": case_sensitive
                }

                results.insert(0, {"summary": summary})

            return results

        except Exception as e:
            self.logger.error(f"Error in file_type_search: {e}", exc_info=True)
            return [{"error": f"Error performing file type search: {str(e)}"}]

    def search_imports(self, module_name: str, file_pattern: str = None) -> List[Dict[str, Any]]:
        """Find where specific modules/packages are imported.

        Args:
            module_name (str): The module or package name to search for.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of import statements with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform search_imports: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Searching for imports of module: {module_name}")

            import_patterns = [
                rf"import\s+{re.escape(module_name)}(\s*,|\s+as|\s*$)",
                rf"from\s+{re.escape(module_name)}\s+import",
                rf"import\s+.*,\s*{re.escape(module_name)}(\s*,|\s+as|\s*$)",
            ]

            results = []
            root_path = self.indexer.root_path

            for root, _, files in os.walk(root_path):
                for filename in files:
                    if not filename.endswith(".py"):
                        continue

                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, root_path)

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                for pattern in import_patterns:
                                    if re.search(pattern, line):
                                        results.append(
                                            {
                                                "file_path": rel_path,
                                                "line_number": i,
                                                "import_statement": line.strip(),
                                            }
                                        )
                                        break
                    except (IOError, UnicodeDecodeError) as e:
                        self.logger.debug(f"Could not read file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error in search_imports: {e}", exc_info=True)
            return [{"error": f"Error searching imports: {str(e)}"}]

    def semantic_search(self, query: str, max_results: int = 5, search_mode: str = "comprehensive") -> List[Dict[str, Any]]:
        """Perform a semantic search with enhanced understanding of code concepts.

        This tool goes beyond simple text matching to understand the meaning and intent
        behind the query, making it more effective for conceptual searches.

        Args:
            query (str): The search query in natural language.
            max_results (int, optional): Maximum number of results to return. Defaults to 5.
            search_mode (str, optional): Search mode - "comprehensive" (search all aspects),
                                        "function" (focus on functions), "class" (focus on classes),
                                        "comment" (focus on comments). Defaults to "comprehensive".

        Returns:
            List[Dict[str, Any]]: List of semantically relevant results with file paths and context.
        """
        if not self.indexer or not self.similarity_search:
            self.logger.error("Cannot perform semantic_search: No indexer or embeddings available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Performing semantic search for query: {query} with mode: {search_mode}")

            query_variations = self._generate_semantic_query_variations(query, search_mode)
            self.logger.info(f"Generated {len(query_variations)} semantic query variations")

            results = self.similarity_search.search_multiple(
                query_variations,
                top_k=max_results * 2,
                threshold=0.10
            )

            if not results:
                self.logger.warning(f"No results found for semantic query: {query}")
                results = self.similarity_search.search_multiple(
                    query_variations,
                    top_k=max_results * 3,
                    threshold=0.05
                )

            formatted_results = []
            for result in results:
                file_name = result.get("file", "")
                chunk = result.get("chunk", {})
                score = result.get("score", 0.0)

                context = self._extract_semantic_context(file_name, chunk)

                formatted_result = {
                    "file_path": file_name,
                    "score": score,
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "text": chunk.get("text", ""),
                    "type": chunk.get("type", ""),
                    "context": context,
                    "relevance_explanation": self._explain_result_relevance(query, chunk.get("text", ""))
                }
                formatted_results.append(formatted_result)

            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            formatted_results = formatted_results[:max_results]

            self.logger.info(f"Found {len(formatted_results)} semantic results for query: {query}")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in semantic_search: {e}", exc_info=True)
            return [{"error": f"Error performing semantic search: {str(e)}"}]

    def _generate_semantic_query_variations(self, query: str, search_mode: str) -> List[str]:
        """Generate semantic variations of the query based on programming concepts.

        Args:
            query (str): The original query.
            search_mode (str): The search mode to focus on.

        Returns:
            List[str]: List of query variations.
        """
        variations = [query]

        variations.append(query.lower())
        variations.append(query.upper())
        variations.append(query.title())

        programming_concepts = {
            "list": ["array", "collection", "sequence", "iterable"],
            "array": ["list", "collection", "sequence", "iterable"],
            "dictionary": ["dict", "map", "hash map", "hash table", "associative array"],
            "map": ["dictionary", "dict", "hash map", "hash table"],
            "tree": ["binary tree", "b-tree", "hierarchy", "graph"],
            "graph": ["network", "tree", "nodes and edges"],
            "stack": ["LIFO", "last in first out"],
            "queue": ["FIFO", "first in first out"],

            # Design patterns
            "singleton": ["global instance", "single instance"],
            "factory": ["creator", "builder pattern"],
            "observer": ["pub-sub", "event listener", "callback"],
            "decorator": ["wrapper", "modifier"],

            # Common operations
            "search": ["find", "query", "lookup", "retrieve"],
            "sort": ["order", "arrange", "organize"],
            "filter": ["select", "where", "query"],
            "map function": ["transform", "convert"],
            "reduce": ["aggregate", "accumulate", "fold"],

            # Web concepts
            "api": ["endpoint", "service", "interface", "rest"],
            "authentication": ["auth", "login", "signin", "identity", "security"],
            "authorization": ["permission", "access control", "role"],
            "request": ["http request", "api call", "fetch"],
            "response": ["http response", "api result", "return value"],

            # Database concepts
            "database": ["db", "data store", "persistence"],
            "query": ["sql", "database request", "search"],
            "orm": ["object relational mapping", "database model"],
            "migration": ["schema change", "database update"],

            # Testing concepts
            "test": ["unit test", "integration test", "spec"],
            "mock": ["stub", "fake", "test double"],
            "assertion": ["expect", "verify", "validate"],
        }

        query_lower = query.lower()
        for concept, alternatives in programming_concepts.items():
            if concept.lower() in query_lower:
                for alt in alternatives:
                    new_variation = query.replace(concept, alt)
                    variations.append(new_variation)

        if search_mode == "function":
            variations.append(f"function {query}")
            variations.append(f"method {query}")
            variations.append(f"def {query}")
        elif search_mode == "class":
            variations.append(f"class {query}")
            variations.append(f"object {query}")
            variations.append(f"type {query}")
        elif search_mode == "comment":
            variations.append(f"# {query}")
            variations.append(f"comment {query}")
            variations.append(f"documentation {query}")

        unique_variations = []
        for v in variations:
            if v and v not in unique_variations:
                unique_variations.append(v)

        return unique_variations

    def _extract_semantic_context(self, file_path: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional context for a search result to enhance understanding.

        Args:
            file_path (str): Path to the file.
            chunk (Dict[str, Any]): The chunk information.

        Returns:
            Dict[str, Any]: Additional context information.
        """
        context = {}

        try:
            if file_path:
                file_stats = self.file_stats(file_path)
                if "error" not in file_stats:
                    context["file_info"] = {
                        "name": file_stats.get("file_name", ""),
                        "extension": file_stats.get("extension", ""),
                        "size": file_stats.get("size_human", ""),
                        "lines": file_stats.get("line_count", 0)
                    }

            chunk_text = chunk.get("text", "")
            chunk_type = chunk.get("type", "")

            if "class" in chunk_type.lower():
                class_match = re.search(r"class\s+(\w+)", chunk_text)
                if class_match:
                    context["class_name"] = class_match.group(1)

                method_matches = re.findall(r"def\s+(\w+)\s*\(", chunk_text)
                if method_matches:
                    context["methods"] = method_matches

            elif "function" in chunk_type.lower() or "method" in chunk_type.lower():
                func_match = re.search(r"def\s+(\w+)\s*\((.*?)\)", chunk_text)
                if func_match:
                    context["function_name"] = func_match.group(1)
                    context["parameters"] = func_match.group(2).split(",")

            import_matches = re.findall(r"import\s+(\w+)|from\s+(\w+)", chunk_text)
            if import_matches:
                imports = []
                for match in import_matches:
                    imports.extend([m for m in match if m])
                if imports:
                    context["imports"] = imports

        except Exception as e:
            self.logger.debug(f"Error extracting semantic context: {e}")

        return context

    def _explain_result_relevance(self, query: str, text: str) -> str:
        """Generate a brief explanation of why this result is relevant to the query.

        Args:
            query (str): The search query.
            text (str): The result text.

        Returns:
            str: Explanation of relevance.
        """
        query_terms = set(query.lower().split())
        text_lower = text.lower()

        found_terms = [term for term in query_terms if term in text_lower]

        if found_terms:
            return f"Contains query terms: {', '.join(found_terms)}"
        else:
            return "Semantically related to the query"

    def find_usage(self, symbol: str, file_pattern: str = None) -> List[Dict[str, Any]]:
        """Find where a function, class, or variable is used.

        Args:
            symbol (str): The symbol name to search for.
            file_pattern (str, optional): Optional filter for specific file types. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of usages with file paths and line numbers.
        """
        if not self.indexer:
            self.logger.error("Cannot perform find_usage: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Finding usages of symbol: {symbol}")

            pattern = rf"(^|[^\w]){re.escape(symbol)}(\(|\s|$|\.)"

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return [{"error": f"Invalid regex pattern: {str(e)}"}]

            results = []
            root_path = self.indexer.root_path

            for root, _, files in os.walk(root_path):
                for filename in files:
                    if not filename.endswith(".py"):
                        continue

                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, root_path)

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    results.append(
                                        {
                                            "file_path": rel_path,
                                            "line_number": i,
                                            "line_text": line.strip(),
                                        }
                                    )
                    except (IOError, UnicodeDecodeError) as e:
                        self.logger.debug(f"Could not read file {file_path}: {e}")

            return results
        except Exception as e:
            self.logger.error(f"Error in find_usage: {e}", exc_info=True)
            return [{"error": f"Error finding usages: {str(e)}"}]

    def code_analysis(self, path: str) -> Dict[str, Any]:
        """Analyze code structure, dependencies, and patterns.

        Args:
            path (str): Path to the file to analyze.

        Returns:
            Dict[str, Any]: Dictionary with code analysis information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform code_analysis: No indexer available")
            return [{"error": "No indexed codebase available. Please index a directory first."}]

        try:
            self.logger.info(f"Analyzing code at path: {path}")

            resolved_path = self._resolve_path(path)
            if not resolved_path:
                return {"error": f"Could not resolve path: {path}"}

            root_path = self.indexer.root_path
            full_path = os.path.join(root_path, resolved_path)

            if not os.path.exists(full_path):
                return {"error": f"Path does not exist: {resolved_path}"}

            if not os.path.isfile(full_path) or not full_path.endswith(".py"):
                return {"error": f"Path is not a Python file: {resolved_path}"}

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()

                try:
                    tree = ast.parse(file_content)

                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                imports.append(
                                    {"type": "import", "name": name.name, "alias": name.asname}
                                )
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for name in node.names:
                                imports.append(
                                    {
                                        "type": "from_import",
                                        "module": module,
                                        "name": name.name,
                                        "alias": name.asname,
                                    }
                                )

                    functions = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append(
                                {
                                    "name": node.name,
                                    "line_number": node.lineno,
                                    "args": [arg.arg for arg in node.args.args],
                                    "docstring": ast.get_docstring(node),
                                }
                            )

                    classes = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            methods = []
                            for child_node in ast.iter_child_nodes(node):
                                if isinstance(child_node, ast.FunctionDef):
                                    methods.append(child_node.name)

                            classes.append(
                                {
                                    "name": node.name,
                                    "line_number": node.lineno,
                                    "methods": methods,
                                    "docstring": ast.get_docstring(node),
                                }
                            )

                    lines = file_content.split("\n")
                    code_lines = 0
                    comment_lines = 0
                    blank_lines = 0

                    for line in lines:
                        line = line.strip()
                        if not line:
                            blank_lines += 1
                        elif line.startswith("#"):
                            comment_lines += 1
                        else:
                            code_lines += 1

                    return {
                        "path": resolved_path,
                        "imports": imports,
                        "functions": functions,
                        "classes": classes,
                        "total_lines": len(lines),
                        "code_lines": code_lines,
                        "comment_lines": comment_lines,
                        "blank_lines": blank_lines,
                    }
                except SyntaxError as e:
                    return {"error": f"Syntax error in file: {e}"}
            except (IOError, UnicodeDecodeError) as e:
                return {"error": f"Could not read file: {e}"}
        except Exception as e:
            self.logger.error(f"Error in code_analysis: {e}", exc_info=True)
            return {"error": f"Error analyzing code: {str(e)}"}

    def get_file_description(self, file_path: str) -> str:
        """Get the description of a file from the descriptions directory.

        Args:
            file_path (str): Path to the file (can be imprecise, partial, or full path).

        Returns:
            str: The description of the file, or an error message if not found.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_file_description: No indexer available")
            return "Error: No indexed codebase available. Please index a directory first."

        try:
            self.logger.info(f"Getting file description for: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return f"Error: File not found: {file_path}"

            description_file_name = resolved_path.replace('/', '_').replace('\\', '_') + '.txt'
            descriptions_dir = os.path.join(self.indexer.index_dir, "descriptions")

            if not os.path.exists(descriptions_dir):
                return f"Error: Descriptions directory does not exist: {descriptions_dir}"

            description_file_path = os.path.join(descriptions_dir, description_file_name)

            if not os.path.exists(description_file_path):
                return f"No description found for file: {resolved_path}"

            try:
                with open(description_file_path, 'r', encoding='utf-8') as f:
                    description = f.read()
                return description
            except Exception as e:
                self.logger.error(f"Error reading description file {description_file_path}: {e}")
                return f"Error reading description file: {str(e)}"

        except Exception as e:
            self.logger.error(f"Error in get_file_description: {e}", exc_info=True)
            return f"Error getting file description: {str(e)}"

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get the metadata of a file from the metadata directory.

        Args:
            file_path (str): Path to the file (can be imprecise, partial, or full path).

        Returns:
            Dict[str, Any]: The metadata of the file, or an error message if not found.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_file_metadata: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting file metadata for: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return {"error": f"File not found: {file_path}"}

            metadata_file_name = resolved_path.replace('/', '_').replace('\\', '_') + '.json'
            metadata_dir = os.path.join(self.indexer.index_dir, "metadata")

            if not os.path.exists(metadata_dir):
                return {"error": f"Metadata directory does not exist: {metadata_dir}"}

            metadata_file_path = os.path.join(metadata_dir, metadata_file_name)

            if not os.path.exists(metadata_file_path):
                return {"error": f"No metadata found for file: {resolved_path}"}

            try:
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                self.logger.error(f"Error reading metadata file {metadata_file_path}: {e}")
                return {"error": f"Error reading metadata file: {str(e)}"}

        except Exception as e:
            self.logger.error(f"Error in get_file_metadata: {e}", exc_info=True)
            return {"error": f"Error getting file metadata: {str(e)}"}

    def get_instructions(self, section: str = None) -> Dict[str, Any]:
        """Get custom instructions from the instructions file.

        This tool retrieves custom instructions that the user has defined in the
        .verbalcode_instructions.json file. These instructions can be used to customize
        the behavior of the AI assistant.

        Args:
            section (str, optional): The section of instructions to get. Defaults to None.

        Returns:
            Dict[str, Any]: The loaded instructions, or an empty dict if not found.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_instructions: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting instructions, section: {section}")

            instructions_path = instructions_manager.get_instructions_path()
            if not instructions_path:
                return {
                    "message": "No instructions file found. You can create one with create_instructions_template.",
                    "instructions": {}
                }

            instructions = instructions_manager.get_instructions(section)

            if section:
                return {
                    "section": section,
                    "instructions": instructions,
                    "instructions_path": instructions_path
                }
            else:
                return {
                    "instructions": instructions,
                    "instructions_path": instructions_path,
                    "sections": list(instructions.keys())
                }

        except Exception as e:
            self.logger.error(f"Error in get_instructions: {e}", exc_info=True)
            return {"error": f"Error getting instructions: {str(e)}"}

    def create_instructions_template(self) -> Dict[str, Any]:
        """Create a template instructions file.

        This tool creates a template instructions file at the root of the project.
        The instructions file can be used to customize the behavior of the AI assistant.

        Returns:
            Dict[str, Any]: The template instructions.
        """
        if not self.indexer:
            self.logger.error("Cannot perform create_instructions_template: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info("Creating instructions template")

            instructions_path = instructions_manager.get_instructions_path()
            if instructions_path:
                return {
                    "message": f"Instructions file already exists at {instructions_path}",
                    "instructions_path": instructions_path
                }

            template = instructions_manager.create_template_instructions()

            if not template:
                return {"error": "Failed to create instructions template"}

            return {
                "message": f"Created instructions template at {os.path.join(self.indexer.root_path, '.verbalcode_instructions.json')}",
                "template": template
            }

        except Exception as e:
            self.logger.error(f"Error in create_instructions_template: {e}", exc_info=True)
            return {"error": f"Error creating instructions template: {str(e)}"}

    def add_memory(self, content: str, category: str = None) -> Dict[str, Any]:
        """Add a new memory.

        This tool adds a new memory to the memory system. Memories are used to provide
        context for future interactions with the AI assistant.

        Args:
            content (str): The memory content.
            category (str, optional): The category of the memory. Defaults to None.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        if not self.indexer:
            self.logger.error("Cannot perform add_memory: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Adding memory: {content[:50]}...")

            if not content.strip():
                return {"error": "Cannot add empty memory"}

            success = memory_manager.add_memory(content, category)

            if success:
                return {
                    "message": "Memory added successfully",
                    "content": content,
                    "category": category or "general"
                }
            else:
                return {"error": "Failed to add memory"}

        except Exception as e:
            self.logger.error(f"Error in add_memory: {e}", exc_info=True)
            return {"error": f"Error adding memory: {str(e)}"}

    def get_memories(self, category: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get memories, optionally filtered by category.

        This tool retrieves memories from the memory system. Memories are used to provide
        context for future interactions with the AI assistant.

        Args:
            category (str, optional): The category to filter by. Defaults to None.
            limit (int, optional): The maximum number of memories to return. Defaults to 10.

        Returns:
            Dict[str, Any]: The memories.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_memories: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting memories, category: {category}, limit: {limit}")

            memories = memory_manager.get_memories(category, limit)

            if not memories:
                return {
                    "message": f"No memories found{f' for category {category}' if category else ''}",
                    "memories": []
                }

            formatted_memories = memory_manager.format_memories_for_display(memories)

            return {
                "memories": memories,
                "formatted_memories": formatted_memories,
                "count": len(memories),
                "category": category or "all"
            }

        except Exception as e:
            self.logger.error(f"Error in get_memories: {e}", exc_info=True)
            return {"error": f"Error getting memories: {str(e)}"}

    def search_memories(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search memories using semantic search.

        This tool searches memories in the memory system using semantic search.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            Dict[str, Any]: The search results.
        """
        if not self.indexer:
            self.logger.error("Cannot perform search_memories: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Searching memories for query: {query}")

            results = memory_manager.search_memories(query, limit)

            if not results:
                return {
                    "message": f"No memories found for query: {query}",
                    "results": []
                }

            formatted_results = memory_manager.format_memories_for_display(results)

            return {
                "results": results,
                "formatted_results": formatted_results,
                "count": len(results),
                "query": query
            }

        except Exception as e:
            self.logger.error(f"Error in search_memories: {e}", exc_info=True)
            return {"error": f"Error searching memories: {str(e)}"}

    def run_command(self, command: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        """Execute a system command with configurable timeout.

        This tool executes a system command and returns the output. It includes
        security measures to prevent dangerous commands.

        Args:
            command (str): The command to execute.
            timeout_seconds (int, optional): The timeout in seconds. Defaults to 30.

        Returns:
            Dict[str, Any]: The result of the command execution.
        """
        try:
            self.logger.info(f"Running command: {command}")

            cwd = self.indexer.root_path if self.indexer and self.indexer.root_path else None
            if cwd:
                self.logger.info(f"Using project root as working directory: {cwd}")

            result = terminal_manager.run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)

            return result

        except Exception as e:
            self.logger.error(f"Error in run_command: {e}", exc_info=True)
            return {"error": f"Error executing command: {str(e)}"}

    def read_terminal(self, terminal_id: int, wait: bool = False, max_wait_seconds: int = 60) -> Dict[str, Any]:
        """Read output from a terminal session.

        This tool reads the output from a terminal session. It can optionally
        wait for the command to complete.

        Args:
            terminal_id (int): The terminal ID.
            wait (bool, optional): Whether to wait for the command to complete. Defaults to False.
            max_wait_seconds (int, optional): The maximum time to wait in seconds. Defaults to 60.

        Returns:
            Dict[str, Any]: The terminal output.
        """
        try:
            self.logger.info(f"Reading terminal {terminal_id}")

            result = terminal_manager.read_terminal(terminal_id, wait, max_wait_seconds)

            return result

        except Exception as e:
            self.logger.error(f"Error in read_terminal: {e}", exc_info=True)
            return {"error": f"Error reading terminal: {str(e)}"}

    def kill_terminal(self, terminal_id: int) -> Dict[str, Any]:
        """Terminate a running terminal process.

        This tool kills a terminal process by its ID.

        Args:
            terminal_id (int): The terminal ID.

        Returns:
            Dict[str, Any]: The result of the kill operation.
        """
        try:
            self.logger.info(f"Killing terminal {terminal_id}")

            result = terminal_manager.kill_terminal(terminal_id)

            return result

        except Exception as e:
            self.logger.error(f"Error in kill_terminal: {e}", exc_info=True)
            return {"error": f"Error killing terminal: {str(e)}"}

    def list_terminals(self) -> Dict[str, Any]:
        """List all active terminal sessions.

        This tool lists all active terminal sessions, including their IDs,
        commands, and status.

        Returns:
            Dict[str, Any]: Information about all active terminals.
        """
        try:
            self.logger.info("Listing terminals")

            result = terminal_manager.list_terminals()

            return result

        except Exception as e:
            self.logger.error(f"Error in list_terminals: {e}", exc_info=True)
            return {"error": f"Error listing terminals: {str(e)}"}

    def google_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a Google search and return the results.

        Args:
            query (str): The query to search for.
            num_results (int, optional): The number of results to return. Defaults to 5.

        Returns:
            Dict[str, Any]: Dictionary with search results.
        """
        try:
            self.logger.info(f"Performing Google search for: {query}")

            results = search(term=query, num_results=num_results, advanced=True, unique=True)
            items = [item for item in results]

            formatted_results = []
            for item in items:
                formatted_results.append({
                    "title": item.title,
                    "url": item.url,
                    "description": item.description
                })

            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }

        except Exception as e:
            self.logger.error(f"Error in google_search: {e}", exc_info=True)
            return {"error": f"Error performing Google search: {str(e)}"}

    def ddg_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search using DuckDuckGo.

        Args:
            query (str): Search query
            num_results (int, optional): Number of results to return. Defaults to 5.

        Returns:
            Dict[str, Any]: Dictionary with search results.
        """
        try:
            self.logger.info(f"Performing DuckDuckGo search for: {query}")

            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=num_results)]

            return {
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            self.logger.error(f"Error in ddg_search: {e}", exc_info=True)
            return {"error": f"Error performing DuckDuckGo search: {str(e)}"}

    def bing_news_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search Bing News for recent articles.

        Args:
            query (str): Search query
            num_results (int, optional): Number of results to return. Defaults to 5.

        Returns:
            Dict[str, Any]: Dictionary with news results.
        """
        try:
            self.logger.info(f"Performing Bing News search for: {query}")

            with DDGS() as ddgs:
                results = [r for r in ddgs.news(query, max_results=num_results)]

            return {
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            self.logger.error(f"Error in bing_news_search: {e}", exc_info=True)
            return {"error": f"Error performing Bing news search: {str(e)}"}

    def get_base_knowledge(self, user_location: str = "Unknown", user_time_zone: str = "America/New_York") -> Dict[str, Any]:
        """Get basic knowledge about current date, time, etc.

        Args:
            user_location (str, optional): The user's location. Defaults to "Unknown".
            user_time_zone (str, optional): The user's time zone. Defaults to "America/New_York".

        Returns:
            Dict[str, Any]: Dictionary with basic knowledge.
        """
        try:
            self.logger.info(f"Getting base knowledge for location: {user_location}, timezone: {user_time_zone}")

            try:
                tz = pytz.timezone(user_time_zone)
                now = datetime.now(tz)

                knowledge = {
                    "todays_date": now.date().isoformat(),
                    "current_time": now.strftime("%H:%M:%S"),
                    "user_location": user_location,
                    "user_time_zone": user_time_zone,
                    "day_of_week": now.strftime("%A"),
                    "month": now.strftime("%B"),
                    "year": now.year,
                    "formatted_date": now.strftime("%B %d, %Y"),
                    "formatted_time": now.strftime("%I:%M %p")
                }

            except Exception as e:
                self.logger.warning(f"Error with timezone, using UTC: {e}")
                now = datetime.now()

                knowledge = {
                    "todays_date": now.date().isoformat(),
                    "current_time": now.strftime("%H:%M:%S"),
                    "user_location": user_location,
                    "user_time_zone": "UTC",
                    "day_of_week": now.strftime("%A"),
                    "month": now.strftime("%B"),
                    "year": now.year,
                    "formatted_date": now.strftime("%B %d, %Y"),
                    "formatted_time": now.strftime("%I:%M %p")
                }

            return knowledge

        except Exception as e:
            self.logger.error(f"Error in get_base_knowledge: {e}", exc_info=True)
            return {"error": f"Error getting base knowledge: {str(e)}"}

    def fetch_webpage(self, url: str, limit: int = 2000) -> Dict[str, Any]:
        """Fetch and extract text content from a webpage.

        Args:
            url (str): The URL to fetch content from.
            limit (int, optional): Maximum number of characters to return. Defaults to 2000.

        Returns:
            Dict[str, Any]: Dictionary with extracted text content.
        """
        try:
            self.logger.info(f"Fetching webpage content from: {url}")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            [script.decompose() for script in soup(["script", "style", "meta", "noscript"])]

            text = soup.get_text(separator="\n", strip=True)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            title = soup.title.string if soup.title else "No title found"

            truncated = False
            if len(text) > limit:
                text = text[:limit-3] + "..."
                truncated = True

            return {
                "url": url,
                "title": title,
                "content": text,
                "truncated": truncated,
                "original_length": len(soup.get_text()),
                "returned_length": len(text)
            }

        except Exception as e:
            self.logger.error(f"Error in fetch_webpage: {e}", exc_info=True)
            return {"error": f"Error fetching webpage: {str(e)}"}

    def get_classes(self, file_path: str) -> Dict[str, Any]:
        """Extract all class definitions from a specified file.

        This tool extracts all class definitions from a file, including their
        methods, attributes, base classes, and docstrings.

        Args:
            file_path (str): Path to the file to analyze.

        Returns:
            Dict[str, Any]: Dictionary with class information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_classes: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting classes from file: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return {"error": f"File not found: {file_path}"}

            full_path = os.path.join(self.indexer.root_path, resolved_path)
            if not os.path.isfile(full_path):
                return {"error": f"Not a file: {resolved_path}"}

            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                return {"error": f"Error reading file: {str(e)}"}

            classes = []

            if ext == ".py":
                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            bases = []
                            for base in node.bases:
                                if isinstance(base, ast.Name):
                                    bases.append(base.id)
                                elif isinstance(base, ast.Attribute):
                                    bases.append(ast.unparse(base))
                                else:
                                    bases.append(ast.unparse(base))

                            docstring = ast.get_docstring(node)

                            decorators = []
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Name):
                                    decorators.append(f"@{decorator.id}")
                                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                                    decorators.append(f"@{decorator.func.id}")
                                elif isinstance(decorator, ast.Attribute):
                                    decorators.append(f"@{ast.unparse(decorator)}")
                                else:
                                    decorators.append(f"@{ast.unparse(decorator)}")

                            methods = []
                            attributes = []

                            for child in node.body:
                                if isinstance(child, ast.FunctionDef):
                                    args_list = []
                                    for arg in child.args.args:
                                        args_list.append(arg.arg)

                                    defaults = child.args.defaults
                                    if defaults:
                                        for i in range(len(defaults)):
                                            arg_index = len(args_list) - len(defaults) + i
                                            if arg_index < len(args_list):
                                                default_value = ast.unparse(defaults[i])
                                                args_list[arg_index] = f"{args_list[arg_index]}={default_value}"

                                    if child.args.vararg:
                                        args_list.append(f"*{child.args.vararg.arg}")
                                    if child.args.kwarg:
                                        args_list.append(f"**{child.args.kwarg.arg}")

                                    signature = f"{child.name}({', '.join(args_list)})"

                                    method_docstring = ast.get_docstring(child)

                                    method_decorators = []
                                    for decorator in child.decorator_list:
                                        if isinstance(decorator, ast.Name):
                                            method_decorators.append(f"@{decorator.id}")
                                        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                                            method_decorators.append(f"@{decorator.func.id}")
                                        elif isinstance(decorator, ast.Attribute):
                                            method_decorators.append(f"@{ast.unparse(decorator)}")
                                        else:
                                            method_decorators.append(f"@{ast.unparse(decorator)}")

                                    methods.append({
                                        "name": child.name,
                                        "signature": signature,
                                        "line_number": child.lineno,
                                        "end_line_number": child.end_lineno,
                                        "docstring": method_docstring,
                                        "decorators": method_decorators
                                    })
                                elif isinstance(child, ast.Assign):
                                    for target in child.targets:
                                        if isinstance(target, ast.Name):
                                            attributes.append({
                                                "name": target.id,
                                                "line_number": child.lineno,
                                                "value": ast.unparse(child.value)
                                            })

                            classes.append({
                                "name": node.name,
                                "bases": bases,
                                "line_number": node.lineno,
                                "end_line_number": node.end_lineno,
                                "docstring": docstring,
                                "decorators": decorators,
                                "methods": methods,
                                "attributes": attributes
                            })
                except SyntaxError as e:
                    return {"error": f"Error parsing Python file: {str(e)}"}

            # JavaScript/TypeScript files
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:

                # Class declarations
                class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{"
                for match in re.finditer(class_pattern, content):
                    name, base = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Find the end of the class (simplified)
                    class_start = match.end()
                    brace_count = 1
                    end_pos = class_start

                    for i in range(class_start, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i
                                break

                    class_body = content[class_start:end_pos]
                    end_line_number = line_number + class_body.count('\n') + 1

                    # Extract docstring (JSDoc)
                    docstring = None
                    jsdoc_pattern = r"/\*\*([\s\S]*?)\*/"
                    class_start_pos = match.start()
                    class_line = content[:class_start_pos].rstrip()
                    last_newline = class_line.rfind('\n')

                    if last_newline != -1:
                        preceding_text = content[last_newline:class_start_pos]
                        jsdoc_match = re.search(jsdoc_pattern, preceding_text)
                        if jsdoc_match:
                            docstring = jsdoc_match.group(1).strip()

                    # Extract methods (simplified)
                    methods = []
                    attributes = []

                    # Method pattern
                    method_pattern = r"(?:async\s+)?(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\(([^)]*)\)"
                    for method_match in re.finditer(method_pattern, class_body):
                        method_name, params = method_match.groups()
                        method_line_number = line_number + class_body[:method_match.start()].count('\n')

                        # Find the end of the method (simplified)
                        method_start = method_match.end()
                        if method_start < len(class_body) and class_body[method_start:].lstrip().startswith('{'):
                            method_brace_count = 0
                            found_opening = False
                            method_end_pos = method_start

                            for i in range(method_start, len(class_body)):
                                if class_body[i] == '{':
                                    found_opening = True
                                    method_brace_count += 1
                                elif class_body[i] == '}':
                                    method_brace_count -= 1
                                    if found_opening and method_brace_count == 0:
                                        method_end_pos = i
                                        break

                            method_body = class_body[method_start:method_end_pos+1]
                            method_end_line_number = method_line_number + method_body.count('\n')

                            methods.append({
                                "name": method_name,
                                "signature": f"{method_name}({params})",
                                "line_number": method_line_number,
                                "end_line_number": method_end_line_number,
                                "docstring": None,
                                "decorators": []
                            })

                    # Attribute pattern (class fields)
                    attr_pattern = r"(\w+)\s*=\s*([^;]+);"
                    for attr_match in re.finditer(attr_pattern, class_body):
                        attr_name, attr_value = attr_match.groups()
                        attr_line_number = line_number + class_body[:attr_match.start()].count('\n')

                        attributes.append({
                            "name": attr_name,
                            "line_number": attr_line_number,
                            "value": attr_value.strip()
                        })

                    classes.append({
                        "name": name,
                        "bases": [base] if base else [],
                        "line_number": line_number,
                        "end_line_number": end_line_number,
                        "docstring": docstring,
                        "decorators": [],
                        "methods": methods,
                        "attributes": attributes
                    })

            # Java/C# files
            elif ext in [".java", ".cs"]:

                # Class declarations
                class_pattern = r"(public|private|protected|internal)?\s+(?:abstract\s+|static\s+|sealed\s+|partial\s+)*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{"
                for match in re.finditer(class_pattern, content):
                    modifier, name, base, interfaces = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Find the end of the class (simplified)
                    class_start = match.end()
                    brace_count = 1
                    end_pos = class_start

                    for i in range(class_start, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i
                                break

                    class_body = content[class_start:end_pos]
                    end_line_number = line_number + class_body.count('\n') + 1

                    # Extract docstring (JavaDoc/XML Doc)
                    docstring = None
                    doc_pattern = r"/\*\*([\s\S]*?)\*/|///(.+)"
                    class_start_pos = match.start()
                    class_line = content[:class_start_pos].rstrip()
                    last_newline = class_line.rfind('\n')

                    if last_newline != -1:
                        preceding_text = content[last_newline:class_start_pos]
                        doc_match = re.search(doc_pattern, preceding_text)
                        if doc_match:
                            docstring = (doc_match.group(1) or doc_match.group(2)).strip()

                    # Extract methods (simplified)
                    methods = []
                    attributes = []

                    # Method pattern
                    method_pattern = r"(public|private|protected|internal)?\s+(?:static\s+|virtual\s+|abstract\s+|override\s+|async\s+)*(?:[\w<>[\],\s]+)\s+(\w+)\s*\(([^)]*)\)"
                    for method_match in re.finditer(method_pattern, class_body):
                        method_modifier, method_name, params = method_match.groups()
                        method_line_number = line_number + class_body[:method_match.start()].count('\n')

                        # Find the end of the method (simplified)
                        method_start = method_match.end()
                        if method_start < len(class_body) and class_body[method_start:].lstrip().startswith('{'):
                            method_brace_count = 0
                            found_opening = False
                            method_end_pos = method_start

                            for i in range(method_start, len(class_body)):
                                if class_body[i] == '{':
                                    found_opening = True
                                    method_brace_count += 1
                                elif class_body[i] == '}':
                                    method_brace_count -= 1
                                    if found_opening and method_brace_count == 0:
                                        method_end_pos = i
                                        break

                            method_body = class_body[method_start:method_end_pos+1]
                            method_end_line_number = method_line_number + method_body.count('\n')

                            methods.append({
                                "name": method_name,
                                "signature": f"{method_modifier or ''} {method_name}({params})".strip(),
                                "line_number": method_line_number,
                                "end_line_number": method_end_line_number,
                                "docstring": None,
                                "decorators": []
                            })

                    # Attribute pattern (fields)
                    attr_pattern = r"(public|private|protected|internal)?\s+(?:static\s+|readonly\s+|const\s+)*(?:[\w<>[\],\s]+)\s+(\w+)\s*=\s*([^;]+);"
                    for attr_match in re.finditer(attr_pattern, class_body):
                        attr_modifier, attr_name, attr_value = attr_match.groups()
                        attr_line_number = line_number + class_body[:attr_match.start()].count('\n')

                        attributes.append({
                            "name": attr_name,
                            "line_number": attr_line_number,
                            "value": attr_value.strip()
                        })

                    # Build the inheritance list
                    inheritance = []
                    if base:
                        inheritance.append(base)
                    if interfaces:
                        inheritance.extend([i.strip() for i in interfaces.split(',')])

                    classes.append({
                        "name": name,
                        "bases": inheritance,
                        "line_number": line_number,
                        "end_line_number": end_line_number,
                        "docstring": docstring,
                        "decorators": [],
                        "methods": methods,
                        "attributes": attributes
                    })

            # Sort classes by line number
            classes.sort(key=lambda x: x["line_number"])

            return {
                "file_path": resolved_path,
                "language": ext[1:] if ext.startswith('.') else ext,  # Remove leading dot
                "classes": classes,
                "count": len(classes)
            }

        except Exception as e:
            self.logger.error(f"Error in get_classes: {e}", exc_info=True)
            return {"error": f"Error getting classes: {str(e)}"}

    def get_variables(self, file_path: str) -> Dict[str, Any]:
        """Extract global and class-level variables from a specified file.

        This tool extracts all variable definitions from a file, including their
        types (if available), values, and line numbers.

        Args:
            file_path (str): Path to the file to analyze.

        Returns:
            Dict[str, Any]: Dictionary with variable information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_variables: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting variables from file: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return {"error": f"File not found: {file_path}"}

            full_path = os.path.join(self.indexer.root_path, resolved_path)
            if not os.path.isfile(full_path):
                return {"error": f"Not a file: {resolved_path}"}

            # Get file extension to determine language
            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            # Read file content
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                return {"error": f"Error reading file: {str(e)}"}

            variables = []

            # Python files
            if ext == ".py":
                try:
                    tree = ast.parse(content)

                    # Get global variables
                    for node in tree.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    var_type = None

                                    # Try to infer type from value
                                    if isinstance(node.value, ast.Constant):
                                        var_type = type(node.value.value).__name__ if node.value.value is not None else None
                                    elif isinstance(node.value, ast.List):
                                        var_type = "list"
                                    elif isinstance(node.value, ast.Dict):
                                        var_type = "dict"
                                    elif isinstance(node.value, ast.Set):
                                        var_type = "set"
                                    elif isinstance(node.value, ast.Tuple):
                                        var_type = "tuple"
                                    elif isinstance(node.value, ast.Call):
                                        if isinstance(node.value.func, ast.Name):
                                            var_type = node.value.func.id

                                    variables.append({
                                        "name": target.id,
                                        "type": var_type,
                                        "value": ast.unparse(node.value),
                                        "line_number": node.lineno,
                                        "scope": "global"
                                    })

                    # Get class variables (attributes)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name

                            for child in node.body:
                                if isinstance(child, ast.Assign):
                                    for target in child.targets:
                                        if isinstance(target, ast.Name):
                                            var_type = None

                                            # Try to infer type from value
                                            if isinstance(child.value, ast.Constant):
                                                var_type = type(child.value.value).__name__ if child.value.value is not None else None
                                            elif isinstance(child.value, ast.List):
                                                var_type = "list"
                                            elif isinstance(child.value, ast.Dict):
                                                var_type = "dict"
                                            elif isinstance(child.value, ast.Set):
                                                var_type = "set"
                                            elif isinstance(child.value, ast.Tuple):
                                                var_type = "tuple"
                                            elif isinstance(child.value, ast.Call):
                                                if isinstance(child.value.func, ast.Name):
                                                    var_type = child.value.func.id

                                            variables.append({
                                                "name": target.id,
                                                "type": var_type,
                                                "value": ast.unparse(child.value),
                                                "line_number": child.lineno,
                                                "scope": f"class:{class_name}"
                                            })

                    # Get type annotations
                    for node in ast.walk(tree):
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                            var_type = ast.unparse(node.annotation)

                            # Find the variable in the list and update its type
                            for var in variables:
                                if var["name"] == node.target.id and var["line_number"] == node.lineno:
                                    var["type"] = var_type
                                    break
                            else:
                                # If not found, add it
                                value = ast.unparse(node.value) if node.value else None
                                variables.append({
                                    "name": node.target.id,
                                    "type": var_type,
                                    "value": value,
                                    "line_number": node.lineno,
                                    "scope": "global"  # Simplified - might be in a class
                                })
                except SyntaxError as e:
                    return {"error": f"Error parsing Python file: {str(e)}"}

            # JavaScript/TypeScript files
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:

                # Global variables (var, let, const)
                var_pattern = r"(var|let|const)\s+(\w+)(?::\s*([\w<>[\]|,\s]+))?\s*=\s*([^;]+);"
                for match in re.finditer(var_pattern, content):
                    var_type, name, type_annotation, value = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    variables.append({
                        "name": name,
                        "type": type_annotation,
                        "value": value.strip(),
                        "line_number": line_number,
                        "scope": "global"
                    })

                # Class fields
                class_pattern = r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{"
                for class_match in re.finditer(class_pattern, content):
                    class_name = class_match.group(1)
                    class_start = class_match.end()

                    # Find the end of the class
                    brace_count = 1
                    class_end = class_start
                    for i in range(class_start, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                class_end = i
                                break

                    class_body = content[class_start:class_end]

                    # Find class fields
                    field_pattern = r"(\w+)(?::\s*([\w<>[\]|,\s]+))?\s*=\s*([^;]+);"
                    for field_match in re.finditer(field_pattern, class_body):
                        field_name, field_type, field_value = field_match.groups()
                        field_line_number = line_number + class_body[:field_match.start()].count('\n')

                        variables.append({
                            "name": field_name,
                            "type": field_type,
                            "value": field_value.strip(),
                            "line_number": field_line_number,
                            "scope": f"class:{class_name}"
                        })

            # Java/C#/C++ files
            elif ext in [".java", ".cs", ".cpp", ".c", ".h", ".hpp"]:

                # Global variables
                var_pattern = r"(public|private|protected|internal|static|const|final)?\s+([\w<>[\],\s]+)\s+(\w+)\s*=\s*([^;]+);"
                for match in re.finditer(var_pattern, content):
                    modifiers, var_type, name, value = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Check if we're inside a class
                    scope = "global"

                    # Simple check - if there's a class declaration before this variable
                    class_pattern = r"class\s+(\w+)"
                    for class_match in re.finditer(class_pattern, content[:match.start()]):
                        class_name = class_match.group(1)

                        # Check if we're still in the class scope
                        class_start = class_match.end()
                        brace_count = 0
                        in_class = False

                        # Find the opening brace
                        for i in range(class_start, match.start()):
                            if content[i] == '{':
                                brace_count += 1
                                in_class = True
                                break

                        if in_class:
                            # Count braces to see if we're still in the class
                            for i in range(class_start, match.start()):
                                if content[i] == '{':
                                    brace_count += 1
                                elif content[i] == '}':
                                    brace_count -= 1

                            if brace_count > 0:
                                scope = f"class:{class_name}"

                    variables.append({
                        "name": name,
                        "type": var_type.strip() if var_type else None,
                        "value": value.strip(),
                        "line_number": line_number,
                        "scope": scope,
                        "modifiers": modifiers.split() if modifiers else []
                    })

            # Sort variables by line number
            variables.sort(key=lambda x: x["line_number"])

            return {
                "file_path": resolved_path,
                "language": ext[1:] if ext.startswith('.') else ext,  # Remove leading dot
                "variables": variables,
                "count": len(variables)
            }

        except Exception as e:
            self.logger.error(f"Error in get_variables: {e}", exc_info=True)
            return {"error": f"Error getting variables: {str(e)}"}

    def get_imports(self, file_path: str) -> Dict[str, Any]:
        """Extract all import statements from a specified file.

        This tool extracts all import statements from a file, including the
        imported modules, aliases, and line numbers.

        Args:
            file_path (str): Path to the file to analyze.

        Returns:
            Dict[str, Any]: Dictionary with import information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_imports: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting imports from file: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return {"error": f"File not found: {file_path}"}

            full_path = os.path.join(self.indexer.root_path, resolved_path)
            if not os.path.isfile(full_path):
                return {"error": f"Not a file: {resolved_path}"}

            # Get file extension to determine language
            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            # Read file content
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                return {"error": f"Error reading file: {str(e)}"}

            imports = []

            # Python files
            if ext == ".py":
                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                imports.append({
                                    "type": "import",
                                    "module": name.name,
                                    "alias": name.asname,
                                    "line_number": node.lineno,
                                    "statement": f"import {name.name}" + (f" as {name.asname}" if name.asname else "")
                                })
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for name in node.names:
                                imports.append({
                                    "type": "from_import",
                                    "module": module,
                                    "name": name.name,
                                    "alias": name.asname,
                                    "line_number": node.lineno,
                                    "statement": f"from {module} import {name.name}" + (f" as {name.asname}" if name.asname else "")
                                })
                except SyntaxError as e:
                    return {"error": f"Error parsing Python file: {str(e)}"}

            # JavaScript/TypeScript files
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:

                # ES6 imports
                import_pattern = r"import\s+(?:{([^}]+)}\s+from\s+)?(?:([^;]+)\s+from\s+)?['\"]([^'\"]+)['\"]"
                for match in re.finditer(import_pattern, content):
                    named_imports, default_import, module = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    if default_import:
                        default_import = default_import.strip()
                        if default_import:
                            imports.append({
                                "type": "default_import",
                                "module": module,
                                "name": default_import,
                                "line_number": line_number,
                                "statement": f"import {default_import} from '{module}'"
                            })

                    if named_imports:
                        for named_import in named_imports.split(','):
                            named_import = named_import.strip()
                            if not named_import:
                                continue

                            alias = None
                            if ' as ' in named_import:
                                name, alias = named_import.split(' as ')
                                name = name.strip()
                                alias = alias.strip()
                            else:
                                name = named_import

                            imports.append({
                                "type": "named_import",
                                "module": module,
                                "name": name,
                                "alias": alias,
                                "line_number": line_number,
                                "statement": f"import {{ {name}" + (f" as {alias}" if alias else "") + f" }} from '{module}'"
                            })

                # CommonJS require
                require_pattern = r"(?:const|let|var)\s+(?:{([^}]+)}\s*=\s*)?(\w+)\s*=\s*require\s*\(['\"]([^'\"]+)['\"]\)"
                for match in re.finditer(require_pattern, content):
                    named_imports, default_import, module = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    if default_import:
                        imports.append({
                            "type": "require",
                            "module": module,
                            "name": default_import,
                            "line_number": line_number,
                            "statement": f"const {default_import} = require('{module}')"
                        })

                    if named_imports:
                        for named_import in named_imports.split(','):
                            named_import = named_import.strip()
                            if not named_import:
                                continue

                            alias = None
                            if ':' in named_import:
                                name, alias = named_import.split(':')
                                name = name.strip()
                                alias = alias.strip()
                            else:
                                name = named_import

                            imports.append({
                                "type": "destructured_require",
                                "module": module,
                                "name": name,
                                "alias": alias,
                                "line_number": line_number,
                                "statement": f"const {{ {name}" + (f": {alias}" if alias else "") + f" }} = require('{module}')"
                            })

            # Java files
            elif ext == ".java":
                # Use regex for import detection
                import_pattern = r"import\s+(?:static\s+)?([^;]+);"
                for match in re.finditer(import_pattern, content):
                    module = match.group(1).strip()
                    line_number = content[:match.start()].count('\n') + 1

                    is_static = "static" in content[match.start():match.end()]

                    imports.append({
                        "type": "import" if not is_static else "static_import",
                        "module": module,
                        "line_number": line_number,
                        "statement": f"import {module};"
                    })

            # C# files
            elif ext == ".cs":
                # Use regex for using statements
                using_pattern = r"using\s+(?:static\s+)?([^;]+);"
                for match in re.finditer(using_pattern, content):
                    namespace = match.group(1).strip()
                    line_number = content[:match.start()].count('\n') + 1

                    is_static = "static" in content[match.start():match.end()]
                    alias = None

                    if "=" in namespace:
                        parts = namespace.split('=')
                        if len(parts) == 2:
                            alias = parts[0].strip()
                            namespace = parts[1].strip()

                    imports.append({
                        "type": "using" if not is_static else "static_using",
                        "namespace": namespace,
                        "alias": alias,
                        "line_number": line_number,
                        "statement": f"using {namespace};"
                    })

            # C/C++ files
            elif ext in [".c", ".cpp", ".h", ".hpp"]:
                # Use regex for include statements
                include_pattern = r"#include\s+[<\"]([^>\"]+)[>\"]"
                for match in re.finditer(include_pattern, content):
                    header = match.group(1).strip()
                    line_number = content[:match.start()].count('\n') + 1
                    is_system = "<" in content[match.start():match.end()]

                    imports.append({
                        "type": "system_include" if is_system else "local_include",
                        "header": header,
                        "line_number": line_number,
                        "statement": (
                            f"#include <{header}>"
                            if is_system
                            else f"#include \"{header}\""
                        )
                    })

            # Sort imports by line number
            imports.sort(key=lambda x: x["line_number"])

            return {
                "file_path": resolved_path,
                "language": ext[1:] if ext.startswith('.') else ext,  # Remove leading dot
                "imports": imports,
                "count": len(imports)
            }

        except Exception as e:
            self.logger.error(f"Error in get_imports: {e}", exc_info=True)
            return {"error": f"Error getting imports: {str(e)}"}

    def get_functions(self, file_path: str) -> Dict[str, Any]:
        """Extract all function names from a specified file.

        This tool extracts all function definitions from a file, including their
        signatures, line numbers, and docstrings.

        Args:
            file_path (str): Path to the file to analyze.

        Returns:
            Dict[str, Any]: Dictionary with function information.
        """
        if not self.indexer:
            self.logger.error("Cannot perform get_functions: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Getting functions from file: {file_path}")

            resolved_path = self.find_closest_file_match(file_path)
            if not resolved_path:
                return {"error": f"File not found: {file_path}"}

            full_path = os.path.join(self.indexer.root_path, resolved_path)
            if not os.path.isfile(full_path):
                return {"error": f"Not a file: {resolved_path}"}

            # Get file extension to determine language
            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            # Read file content
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                return {"error": f"Error reading file: {str(e)}"}

            functions = []

            # Python files
            if ext == ".py":
                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Get function signature
                            args_list = []
                            for arg in node.args.args:
                                args_list.append(arg.arg)

                            # Handle default values
                            defaults = node.args.defaults
                            if defaults:
                                for i in range(len(defaults)):
                                    arg_index = len(args_list) - len(defaults) + i
                                    if arg_index < len(args_list):
                                        default_value = ast.unparse(defaults[i])
                                        args_list[arg_index] = f"{args_list[arg_index]}={default_value}"

                            # Handle *args and **kwargs
                            if node.args.vararg:
                                args_list.append(f"*{node.args.vararg.arg}")
                            if node.args.kwarg:
                                args_list.append(f"**{node.args.kwarg.arg}")

                            signature = f"{node.name}({', '.join(args_list)})"

                            # Get docstring if available
                            docstring = ast.get_docstring(node)

                            # Get decorators
                            decorators = []
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Name):
                                    decorators.append(f"@{decorator.id}")
                                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                                    decorators.append(f"@{decorator.func.id}")
                                elif isinstance(decorator, ast.Attribute):
                                    decorators.append(f"@{ast.unparse(decorator)}")
                                else:
                                    decorators.append(f"@{ast.unparse(decorator)}")

                            functions.append({
                                "name": node.name,
                                "signature": signature,
                                "line_number": node.lineno,
                                "end_line_number": node.end_lineno,
                                "docstring": docstring,
                                "decorators": decorators
                            })
                except SyntaxError as e:
                    return {"error": f"Error parsing Python file: {str(e)}"}

            # JavaScript/TypeScript files
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:

                # Regular function declarations
                func_pattern = r"(async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
                for match in re.finditer(func_pattern, content):
                    is_async, name, params = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Find the end of the function (simplified)
                    function_start = match.end()
                    brace_count = 0
                    found_opening = False
                    end_pos = function_start

                    for i in range(function_start, len(content)):
                        if content[i] == '{':
                            found_opening = True
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if found_opening and brace_count == 0:
                                end_pos = i
                                break

                    function_body = content[function_start:end_pos+1]
                    end_line_number = line_number + function_body.count('\n')

                    # Extract docstring (JSDoc)
                    docstring = None
                    jsdoc_pattern = r"/\*\*([\s\S]*?)\*/"
                    function_start_pos = match.start()
                    function_line = content[:function_start_pos].rstrip()
                    last_newline = function_line.rfind('\n')

                    if last_newline != -1:
                        preceding_text = content[last_newline:function_start_pos]
                        jsdoc_match = re.search(jsdoc_pattern, preceding_text)
                        if jsdoc_match:
                            docstring = jsdoc_match.group(1).strip()

                    signature = f"{'async ' if is_async else ''}function {name}({params})"

                    functions.append({
                        "name": name,
                        "signature": signature,
                        "line_number": line_number,
                        "end_line_number": end_line_number,
                        "docstring": docstring,
                        "decorators": []
                    })

                # Arrow functions with explicit names (const/let/var)
                arrow_pattern = r"(const|let|var)\s+(\w+)\s*=\s*(async\s*)?\(([^)]*)\)\s*=>"
                for match in re.finditer(arrow_pattern, content):
                    var_type, name, is_async, params = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Find the end of the function (simplified)
                    function_start = match.end()

                    # Check if it's a block body or expression body
                    if content[function_start:].lstrip().startswith('{'):
                        # Block body
                        brace_count = 0
                        found_opening = False
                        end_pos = function_start

                        for i in range(function_start, len(content)):
                            if content[i] == '{':
                                found_opening = True
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if found_opening and brace_count == 0:
                                    end_pos = i
                                    break
                    else:
                        # Expression body
                        end_pos = function_start
                        for i in range(function_start, len(content)):
                            if content[i] == ';' or content[i] == '\n':
                                end_pos = i
                                break

                    function_body = content[function_start:end_pos+1]
                    end_line_number = line_number + function_body.count('\n')

                    # Extract docstring (JSDoc)
                    docstring = None
                    jsdoc_pattern = r"/\*\*([\s\S]*?)\*/"
                    function_start_pos = match.start()
                    function_line = content[:function_start_pos].rstrip()
                    last_newline = function_line.rfind('\n')

                    if last_newline != -1:
                        preceding_text = content[last_newline:function_start_pos]
                        jsdoc_match = re.search(jsdoc_pattern, preceding_text)
                        if jsdoc_match:
                            docstring = jsdoc_match.group(1).strip()

                    signature = f"{var_type} {name} = {'async ' if is_async else ''}({params}) =>"

                    functions.append({
                        "name": name,
                        "signature": signature,
                        "line_number": line_number,
                        "end_line_number": end_line_number,
                        "docstring": docstring,
                        "decorators": []
                    })

            # Java/C#/C++ files
            elif ext in [".java", ".cs", ".cpp", ".c", ".h", ".hpp"]:

                # Function pattern for Java/C#/C++
                func_pattern = r"(public|private|protected|internal|static|virtual|override|async)?\s*([\w<>[\],\s]+)\s+(\w+)\s*\(([^)]*)\)"
                for match in re.finditer(func_pattern, content):
                    modifiers, return_type, name, params = match.groups()
                    line_number = content[:match.start()].count('\n') + 1

                    # Find the end of the function (simplified)
                    function_start = match.end()

                    # Check if it's a function declaration or definition
                    if content[function_start:].lstrip().startswith('{'):
                        # Function definition
                        brace_count = 0
                        found_opening = False
                        end_pos = function_start

                        for i in range(function_start, len(content)):
                            if content[i] == '{':
                                found_opening = True
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if found_opening and brace_count == 0:
                                    end_pos = i
                                    break

                        function_body = content[function_start:end_pos+1]
                        end_line_number = line_number + function_body.count('\n')
                    else:
                        # Function declaration
                        end_pos = function_start
                        for i in range(function_start, len(content)):
                            if content[i] == ';':
                                end_pos = i
                                break

                        function_body = content[function_start:end_pos+1]
                        end_line_number = line_number + function_body.count('\n')

                    # Extract docstring (JavaDoc/XML Doc/Doxygen)
                    docstring = None
                    doc_pattern = r"/\*\*([\s\S]*?)\*/|///(.+)"
                    function_start_pos = match.start()
                    function_line = content[:function_start_pos].rstrip()
                    last_newline = function_line.rfind('\n')

                    if last_newline != -1:
                        preceding_text = content[last_newline:function_start_pos]
                        doc_match = re.search(doc_pattern, preceding_text)
                        if doc_match:
                            docstring = (doc_match.group(1) or doc_match.group(2)).strip()

                    signature = f"{modifiers or ''} {return_type} {name}({params})"

                    functions.append({
                        "name": name,
                        "signature": signature.strip(),
                        "line_number": line_number,
                        "end_line_number": end_line_number,
                        "docstring": docstring,
                        "decorators": []
                    })

            # Sort functions by line number
            functions.sort(key=lambda x: x["line_number"])

            return {
                "file_path": resolved_path,
                "language": ext[1:] if ext.startswith('.') else ext,
                "functions": functions,
                "count": len(functions)
            }

        except Exception as e:
            self.logger.error(f"Error in get_functions: {e}", exc_info=True)
            return {"error": f"Error getting functions: {str(e)}"}

    def explain_code(
        self, path: str, line_start: int = None, line_end: int = None
    ) -> Dict[str, Any]:
        """Generate an explanation of a code snippet.

        Args:
            path (str): Path to the file.
            line_start (int, optional): Optional starting line number. Defaults to None.
            line_end (int, optional): Optional ending line number. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary with code explanation.
        """
        if not self.indexer:
            self.logger.error("Cannot perform explain_code: No indexer available")
            return {"error": "No indexed codebase available. Please index a directory first."}

        try:
            self.logger.info(f"Explaining code at path: {path}")

            file_content = self.read_file(path, line_start, line_end)

            if "error" in file_content:
                return {"error": file_content["error"]}

            code_snippet = file_content.get("content", "")
            file_path = file_content.get("file_path", path)

            analysis = {}

            if file_path.endswith(".py"):
                try:
                    tree = ast.parse(code_snippet)

                    functions = []
                    classes = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            classes.append(node.name)

                    analysis["functions"] = functions
                    analysis["classes"] = classes

                    lines = code_snippet.split("\n")
                    code_lines = 0
                    comment_lines = 0
                    blank_lines = 0

                    for line in lines:
                        line = line.strip()
                        if not line:
                            blank_lines += 1
                        elif line.startswith("#"):
                            comment_lines += 1
                        else:
                            code_lines += 1

                    analysis["total_lines"] = len(lines)
                    analysis["code_lines"] = code_lines
                    analysis["comment_lines"] = comment_lines
                    analysis["blank_lines"] = blank_lines
                except SyntaxError:
                    analysis["error"] = "Could not parse code due to syntax errors"

            return {
                "path": file_path,
                "code": code_snippet,
                "analysis": analysis,
                "line_start": line_start,
                "line_end": line_end,
            }
        except Exception as e:
            self.logger.error(f"Error in explain_code: {e}", exc_info=True)
            return {"error": f"Error explaining code: {str(e)}"}