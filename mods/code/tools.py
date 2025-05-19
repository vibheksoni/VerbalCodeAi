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
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .directory import DirectoryEntry, DirectoryParser, EntryType
from .embed import SimilaritySearch

logger = logging.getLogger("VerbalCodeAI.Tools")


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

    def ask_buddy(self, question: str) -> Dict[str, Any]:
        """Ask the buddy AI model for opinions or suggestions.

        Args:
            question (str): The question or request to ask the buddy AI.

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

            from .. import llms

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that provides opinions and suggestions to another AI agent. Be concise and direct in your responses."},
                {"role": "user", "content": question},
            ]

            response = llms.generate_response(
                messages=messages,
                temperature=0.7,
                parse_thinking=False,
                provider=provider,
                api_key=api_key,
                model=model,
            )

            return {"response": response, "provider": provider, "model": model}

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