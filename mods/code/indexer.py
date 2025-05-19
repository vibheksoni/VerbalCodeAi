"""File indexer module for analyzing and storing code file information.

This module provides functionality to:
1. Scan directories for code files
2. Generate file descriptions using AI
3. Create and store embeddings
4. Save file metadata and content information
"""

import ast
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..llms import generate_description, generate_embed
from .directory import (
    DirectoryEntry,
    DirectoryParser,
    EntryType,
    HASH_ALGORITHM,
    HASH_BUFFER_SIZE,
)
from .embed import CodeEmbedding, SimilaritySearch

logger = logging.getLogger("VerbalCodeAI.Indexer")
logger.info("[INDEXER] LOGGER WORKING")


class DirectIndexerLogger:
    """Simple direct file logger for the indexer that doesn't rely on logging framework."""

    def __init__(self):
        """Set up the logger with a timestamped file in the logs directory."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = logs_dir / f"indexer_direct_{timestamp}.log"

        try:
            with open(self.filename, "w") as f:
                f.write(f"=== DirectIndexerLogger started at {datetime.datetime.now()} ===\n")
        except Exception as e:
            logger.error(f"ERROR setting up DirectIndexerLogger: {e}")

    def log(self, message: str) -> None:
        """Log a message with timestamp directly to file.

        Args:
            message (str): The message to log.
        """
        try:
            with open(self.filename, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"{timestamp} - {message}\n")
                f.flush()
        except Exception as e:
            logger.error(f"ERROR in DirectIndexerLogger: {e}")


direct_logger = DirectIndexerLogger()


@dataclass
class FileSignature:
    """Represents a function or class signature from a file."""

    name: str
    """The name of the function or class."""
    type: str
    """The type of the signature ('function' or 'class')."""
    signature: str
    """The signature string."""
    line_number: int
    """The line number where the signature is defined."""


@dataclass
class FileMetadata:
    """Represents metadata and analysis results for a single file."""

    name: str
    """The name of the file."""
    path: str
    """The path to the file."""
    hash: str
    """The hash of the file content."""
    size: int
    """The size of the file in bytes."""
    extension: str
    """The file extension."""
    modified_time: float
    """The last modified timestamp of the file."""
    description: str
    """A description of the file's purpose and content."""
    signatures: List[FileSignature]
    """A list of function and class signatures found in the file."""
    chunks: List[Dict[str, Any]]
    """A list of code chunks extracted from the file."""
    embeddings: List[List[float]]
    """A list of embeddings for each code chunk."""


class FileIndexer:
    """A class for indexing code files in a project directory.

    This class handles:
    - Directory traversal with exclusion patterns
    - File content analysis
    - AI-powered description generation
    - Embedding creation and storage
    - Metadata extraction and storage
    """

    DEFAULT_EXCLUDED_EXTENSIONS = {
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
        ".pyc",
        ".pyo",
        ".pyd",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".npz",
        ".npy",
        ".pkl",
        ".pickle",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
    }

    KNOWN_SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".java",
        ".rb",
        ".go",
        ".rs",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".lua",
        ".r",
        ".dart",
        ".hs",
        ".ml",
        ".ex",
        ".exs",
        ".erl",
        ".hrl",
        ".clj",
        ".cljs",
        ".groovy",
        ".pl",
        ".pm",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".psm1",
        ".elm",
        ".f",
        ".f90",
        ".f95",
        ".jl",
        ".m",
        ".mm",
        ".nim",
        ".v",
        ".vhd",
        ".vhdl",
        ".zig",
        ".d",
        ".fs",
        ".fsx",
        ".fsi",
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".adoc",
        ".asciidoc",
        ".wiki",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".conf",
        ".config",
        ".properties",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        ".gitattributes",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".svg",
        ".xml",
        ".csv",
        ".tsv",
        ".sql",
        ".graphql",
        ".gql",
        ".makefile",
        ".dockerfile",
        ".cmake",
        ".gradle",
        ".bazel",
        ".log",
        ".diff",
        ".patch",
        ".template",
        ".j2",
        ".tpl",
        ".proto",
        ".plist",
        ".manifest",
        ".lock",
        ".ipynb",
    }

    def __init__(
        self,
        root_path: str,
        index_dir: str = ".index",
        excluded_extensions: Optional[Set[str]] = None,
        gitignore_path: Optional[str] = None,
        chunk_size: int = 1000,
    ):
        """Initialize the FileIndexer.

        Args:
            root_path (str): Root directory to index.
            index_dir (str): Directory to store index files (relative to root_path). Defaults to ".index".
            excluded_extensions (Optional[Set[str]]): Additional file extensions to exclude. Defaults to None.
            gitignore_path (Optional[str]): Path to .gitignore file for additional exclusions. Defaults to None.
            chunk_size (int): Size of chunks for large files. Defaults to 1000.
        """
        try:
            direct_logger.log(f"FileIndexer.__init__ ENTRY: root_path={root_path}")
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            with open(logs_dir / "indexer_init_debug.log", "a+") as f:
                f.write(f"FileIndexer __init__ called for root_path: {root_path} at {time.time()}\n")

            direct_logger.log(f"Processing arguments: index_dir={index_dir}")

            self.root_path: str = os.path.abspath(root_path)
            direct_logger.log(f"Absolute root_path: {self.root_path}")

            self.index_dir: str = os.path.join(self.root_path, index_dir)
            direct_logger.log(f"Computed index_dir: {self.index_dir}")

            self.excluded_extensions: Set[str] = self.DEFAULT_EXCLUDED_EXTENSIONS.union(
                excluded_extensions or set()
            )
            direct_logger.log(f"Set up excluded_extensions with {len(self.excluded_extensions)} extensions")

            direct_logger.log("Setting up gitignore path...")
            if gitignore_path:
                self.gitignore_path: Optional[str] = (
                    gitignore_path
                    if os.path.isabs(gitignore_path)
                    else os.path.join(self.root_path, gitignore_path)
                )
                if not os.path.exists(self.gitignore_path):
                    direct_logger.log(f"Warning: Specified gitignore file not found at {self.gitignore_path}")
                    print(
                        f"Warning: Specified gitignore file not found at {self.gitignore_path}"
                    )
                    self.gitignore_path = None
                else:
                    direct_logger.log(f"Using specified gitignore file: {self.gitignore_path}")
            else:
                default_gitignore: str = os.path.join(self.root_path, ".gitignore")
                if os.path.exists(default_gitignore):
                    self.gitignore_path: str = default_gitignore
                    direct_logger.log(f"Using default gitignore file: {self.gitignore_path}")
                    print(f"Using default gitignore file at {self.gitignore_path}")
                else:
                    self.gitignore_path: None = None
                    direct_logger.log("No gitignore file found")

            self.chunk_size: int = chunk_size
            direct_logger.log(f"Set chunk_size: {self.chunk_size}")

            direct_logger.log("Creating CodeEmbedding instance")
            self.code_embedder: CodeEmbedding = CodeEmbedding()
            direct_logger.log("CodeEmbedding instance created successfully")

            self.metadata_cache: Dict[str, Any] = {}
            direct_logger.log("Initialized empty metadata_cache")

            self.similarity_search: Optional[SimilaritySearch] = None

            direct_logger.log("Calling _create_index_structure()")
            self._create_index_structure()
            direct_logger.log("_create_index_structure() completed")

            direct_logger.log("Calling _load_metadata_cache()")
            self._load_metadata_cache()
            direct_logger.log("_load_metadata_cache() completed")

            direct_logger.log("Initializing SimilaritySearch")
            self._initialize_similarity_search()
            direct_logger.log("SimilaritySearch initialization completed")

            direct_logger.log("FileIndexer.__init__ completed successfully")
        except Exception as e:
            error_details = traceback.format_exc()
            direct_logger.log(f"ERROR in FileIndexer.__init__: {str(e)}")
            direct_logger.log(f"Traceback: {error_details}")
            raise

    def _create_index_structure(self) -> None:
        """Create the directory structure for storing index files."""
        direct_logger.log(f"_create_index_structure() ENTRY: index_dir={self.index_dir}")

        logger.debug(f"CHECKPOINT: [DIR.1] Creating index directory structure at {self.index_dir}")
        direct_logger.log(f"CHECKPOINT: [DIR.1] Creating index directory structure at {self.index_dir}")

        parent_dir = os.path.dirname(self.index_dir)
        logger.debug(f"CHECKPOINT: [DIR.2] Checking write permissions for parent directory: {parent_dir}")
        direct_logger.log(f"CHECKPOINT: [DIR.2] Checking write permissions for parent directory: {parent_dir}")

        if not os.access(parent_dir, os.W_OK):
            error_msg = f"No write permission to create index in {parent_dir}"
            logger.error(f"CHECKPOINT: [DIR.3] {error_msg}")
            direct_logger.log(f"CHECKPOINT: [DIR.3] {error_msg}")
            raise PermissionError(error_msg)
        else:
            logger.debug(f"CHECKPOINT: [DIR.4] Parent directory is writable: {parent_dir}")
            direct_logger.log(f"CHECKPOINT: [DIR.4] Parent directory is writable: {parent_dir}")

        index_subdirs = ["", "metadata", "embeddings", "descriptions"]
        logger.debug(f"CHECKPOINT: [DIR.5] Will create the following subdirectories: {index_subdirs}")
        direct_logger.log(f"CHECKPOINT: [DIR.5] Will create the following subdirectories: {index_subdirs}")

        for subdir in index_subdirs:
            dir_path = os.path.join(self.index_dir, subdir) if subdir else self.index_dir
            logger.debug(f"CHECKPOINT: [DIR.6] Creating directory: {dir_path}")
            direct_logger.log(f"CHECKPOINT: [DIR.6] Creating directory: {dir_path}")

            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"CHECKPOINT: [DIR.7] Created directory: {dir_path}")
                direct_logger.log(f"CHECKPOINT: [DIR.7] Created directory: {dir_path}")

                if not os.path.exists(dir_path):
                    error_msg = f"Failed to create directory {dir_path} - directory doesn't exist after creation"
                    logger.error(f"CHECKPOINT: [DIR.8] {error_msg}")
                    direct_logger.log(f"CHECKPOINT: [DIR.8] {error_msg}")
                    raise IOError(error_msg)
                else:
                    logger.debug(f"CHECKPOINT: [DIR.9] Verified directory exists: {dir_path}")
                    direct_logger.log(f"CHECKPOINT: [DIR.9] Verified directory exists: {dir_path}")

                test_file_path = os.path.join(dir_path, ".write_test")
                logger.debug(f"CHECKPOINT: [DIR.10] Testing write permissions with test file: {test_file_path}")
                direct_logger.log(f"CHECKPOINT: [DIR.10] Testing write permissions with test file: {test_file_path}")
                try:
                    with open(test_file_path, "w") as f:
                        f.write("test")
                    os.remove(test_file_path)
                    logger.debug(f"CHECKPOINT: [DIR.11] Successfully wrote and removed test file in {dir_path}")
                    direct_logger.log(f"CHECKPOINT: [DIR.11] Successfully wrote and removed test file in {dir_path}")
                except Exception as write_test_error:
                    error_msg = f"Directory {dir_path} created but is not writable: {str(write_test_error)}"
                    logger.error(f"CHECKPOINT: [DIR.12] {error_msg}")
                    direct_logger.log(f"CHECKPOINT: [DIR.12] {error_msg}")
                    raise PermissionError(error_msg)

            except Exception as e:
                logger.error(f"CHECKPOINT: [DIR.13] Failed to create directory {dir_path}: {str(e)}", exc_info=True)
                direct_logger.log(f"CHECKPOINT: [DIR.13] Failed to create directory {dir_path}: {str(e)}")
                direct_logger.log(f"Traceback: {traceback.format_exc()}")
                raise

        logger.info(f"CHECKPOINT: [DIR.14] Index directory structure created successfully at {self.index_dir}")
        direct_logger.log(f"CHECKPOINT: [DIR.14] Index directory structure created successfully at {self.index_dir}")

    def _is_text_file(self, file_path: str) -> bool:
        """Check if a file is a text file by attempting to read its first few bytes.

        Args:
            file_path (str): Path to the file to check.

        Returns:
            bool: True if the file appears to be text.
        """
        logger.debug(f"ISTEXT: Checking file: {file_path}")
        file_ext_lower_dot = f".{os.path.splitext(file_path)[1].lstrip('.').lower()}"

        try:
            with open(file_path, "rb") as f:
                chunk: bytes = f.read(1024)
                if not chunk:
                    logger.debug(f"ISTEXT: Empty file, considered text: {file_path}")
                    return True

                if any(
                    chunk.startswith(sig)
                    for sig in [
                        b"PK\x03\x04",
                        b"\x89PNG\r\n",
                        b"GIF87a",
                        b"GIF89a",
                        b"\xFF\xD8\xFF",
                    ]
                ):
                    logger.debug(f"ISTEXT: Binary signature found: {file_path}")
                    return False

                if b"\x00" in chunk:
                    if file_ext_lower_dot in FileIndexer.KNOWN_SOURCE_EXTENSIONS:
                        try:
                            chunk.decode("utf-8")
                            null_byte_char = b"\\x00"
                            logger.debug(
                                f"ISTEXT: Known source {file_path} contains nulls (count: {chunk.count(null_byte_char)}) but decodes as UTF-8. Considered text."
                            )
                            return True
                        except UnicodeDecodeError:
                            null_byte_char = b"\\x00"
                            logger.debug(
                                f"ISTEXT: Known source {file_path} contains nulls (count: {chunk.count(null_byte_char)}) and fails UTF-8 decode. Considered binary."
                            )
                            return False
                    else:
                        logger.debug(f"ISTEXT: Unknown type {file_path} contains nulls. Considered binary.")
                        return False

                try:
                    chunk.decode("utf-8")
                    logger.debug(f"ISTEXT: Decodes as UTF-8: {file_path}")
                    return True
                except UnicodeDecodeError:
                    try:
                        chunk.decode("ascii")
                        logger.debug(f"ISTEXT: Decodes as ASCII: {file_path}")
                        return True
                    except UnicodeDecodeError:
                        logger.debug(
                            f"ISTEXT: Failed to decode as UTF-8/ASCII and no nulls (or handled): {file_path}"
                        )
                        return False
        except Exception as e:
            logger.error(f"ISTEXT: Error checking file {file_path}: {str(e)}", exc_info=True)
            return False

    def _should_index_file(self, entry: DirectoryEntry) -> bool:
        """Determine if a file should be indexed based on its extension and content.

        Args:
            entry (DirectoryEntry): DirectoryEntry object to check.

        Returns:
            bool: True if the file should be indexed.
        """
        if not entry.is_file():
            return False

        index_dir_abs: str = os.path.abspath(self.index_dir)
        if entry.path.startswith(index_dir_abs):
            return False

        file_ext_lower_dot = f".{entry.extension.lower()}" if entry.extension else ""
        if file_ext_lower_dot in self.excluded_extensions:
            return False

        is_text = self._is_text_file(entry.path)
        if not is_text:
            logger.info(
                f"SHOULD_INDEX: File {entry.path} (ext: {file_ext_lower_dot}) determined as non-text by _is_text_file, won't index."
            )
        return is_text

    def _extract_signatures(self, file_path: str) -> List[FileSignature]:
        """Extract function and class signatures from a file.

        Uses tree-sitter for supported languages and falls back to regex for others.

        Args:
            file_path (str): Path to the file to analyze.

        Returns:
            List[FileSignature]: List of extracted signatures.
        """
        signatures: List[FileSignature] = []

        try:
            ext: str = os.path.splitext(file_path)[1].lower()
            if ext in self.code_embedder.chunker.SUPPORTED_LANGUAGES:
                chunks: List[Dict[str, Any]] = self.code_embedder.chunker.chunk_file(file_path)
                for chunk in chunks:
                    signatures.append(
                        FileSignature(
                            name=os.path.basename(chunk["text"].split("\n")[0].strip()),
                            type=chunk["type"].replace("_definition", ""),
                            signature=chunk["text"].split("\n")[0].strip(),
                            line_number=chunk["start_line"],
                        )
                    )
            elif ext == ".py":
                with open(file_path, "r", encoding="utf-8") as f:
                    content: str = f.read()

                tree: ast.AST = ast.parse(content)

                class_nodes: Set[ast.ClassDef] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_nodes.add(node)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        is_method: bool = False
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef) and node in parent.body:
                                is_method = True
                                break

                        if not is_method:
                            signatures.append(
                                FileSignature(
                                    name=node.name,
                                    type="function",
                                    signature=self._get_function_signature(node),
                                    line_number=node.lineno,
                                )
                            )
                    elif isinstance(node, ast.ClassDef):
                        signatures.append(
                            FileSignature(
                                name=node.name,
                                type="class",
                                signature=f"class {node.name}",
                                line_number=node.lineno,
                            )
                        )
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                patterns = [
                    r"(?:function|const)\s+(\w+)\s*\([^)]*\)",
                    r"(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\)",
                    r"def\s+(\w+)",
                    r"func\s+(\w+)\s*\([^)]*\)",
                ]

                line_number: int = 1
                for line in content.split("\n"):
                    for pattern in patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            signatures.append(
                                FileSignature(
                                    name=match.group(1),
                                    type="function",
                                    signature=line.strip(),
                                    line_number=line_number,
                                )
                            )
                    line_number += 1

        except Exception as e:
            logger.error(f"Error extracting signatures from {file_path}: {e}")

        return signatures

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get a string representation of a function's signature.

        Args:
            node (ast.FunctionDef): AST node for the function.

        Returns:
            str: Function signature.
        """
        args: List[str] = []

        for arg in node.args.args:
            arg_str: str = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_str += f": {arg.annotation.id}"
                elif isinstance(arg.annotation, ast.Constant):
                    arg_str += f": {arg.annotation.value}"
            args.append(arg_str)

        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_str += f": {arg.annotation.id}"
            args.append(arg_str)

        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        returns: str = ""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = f" -> {node.returns.id}"
            elif isinstance(node.returns, ast.Constant):
                returns = f" -> {node.returns.value}"

        return f"def {node.name}({', '.join(args)}){returns}"

    def _generate_description(
        self, file_path: str, signatures: List[FileSignature]
    ) -> str:
        """Generate a description of the file using AI by sampling from multiple parts.

        Args:
            file_path (str): Path to the file to describe.
            signatures (List[FileSignature]): List of extracted signatures from the file.

        Returns:
            str: Generated description.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content: str = f.read()

            file_size: int = len(content)
            sample_size: int = min(1000, file_size // 3)

            start_sample: str = content[:sample_size]
            mid_point: int = file_size // 2
            mid_sample: str = content[mid_point - sample_size // 2 : mid_point + sample_size // 2]
            end_sample: str = content[-sample_size :]

            prompt: str = f"""Please provide a concise description (max 5 lines) of this code file. Include:
1. The file's main purpose
2. Key functionality
3. Important classes/functions
4. Code structure/patterns used
5. Any notable dependencies or requirements

File: {os.path.basename(file_path)}
Signatures found:
{chr(10).join(f'- {sig.type}: {sig.signature}' for sig in signatures)}

Code samples from different parts of the file:

START OF FILE:
{start_sample}

MIDDLE OF FILE:
{mid_sample}

END OF FILE:
{end_sample}
"""

            return generate_description(prompt)

        except Exception as e:
            return f"Error generating description: {str(e)}"

    def _save_file_metadata(self, metadata: FileMetadata) -> None:
        """Save file metadata to the index directory.

        Args:
            metadata (FileMetadata): FileMetadata object to save.
        """
        try:
            logger.debug(f"CHECKPOINT: [SAVE.1] Starting to save metadata for file: {metadata.path}")

            if not os.path.exists(self.index_dir):
                logger.warning(f"CHECKPOINT: [SAVE.2] Index directory does not exist, creating it: {self.index_dir}")
                self._create_index_structure()
                logger.debug(f"CHECKPOINT: [SAVE.3] Index directory structure created")
            else:
                logger.debug(f"CHECKPOINT: [SAVE.4] Index directory exists: {self.index_dir}")

            rel_path = os.path.relpath(metadata.path, self.root_path)
            safe_path = re.sub(r"[^\w\-_\.]", "_", rel_path)
            logger.debug(f"CHECKPOINT: [SAVE.5] Generated safe path: {safe_path}")

            try:
                metadata_dir = os.path.join(self.index_dir, "metadata")
                logger.debug(f"CHECKPOINT: [SAVE.6] Metadata directory path: {metadata_dir}")

                if not os.path.exists(metadata_dir):
                    logger.warning(f"CHECKPOINT: [SAVE.7] Metadata directory does not exist, creating it")
                    os.makedirs(metadata_dir, exist_ok=True)
                    logger.debug(f"CHECKPOINT: [SAVE.8] Metadata directory created")
                else:
                    logger.debug(f"CHECKPOINT: [SAVE.9] Metadata directory exists")

                metadata_path = os.path.join(metadata_dir, f"{safe_path}.json")
                logger.debug(f"CHECKPOINT: [SAVE.10] Saving metadata JSON to: {metadata_path}")

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "name": metadata.name,
                            "path": metadata.path,
                            "hash": metadata.hash,
                            "size": metadata.size,
                            "extension": metadata.extension,
                            "modified_time": metadata.modified_time,
                            "description": metadata.description,
                            "signatures": [vars(sig) for sig in metadata.signatures],
                        },
                        f,
                        indent=2,
                    )
                logger.debug(f"CHECKPOINT: [SAVE.11] Saved metadata JSON to {metadata_path}")

                if os.path.exists(metadata_path):
                    file_size = os.path.getsize(metadata_path)
                    logger.debug(f"CHECKPOINT: [SAVE.12] Verified metadata file exists, size: {file_size} bytes")
                else:
                    logger.error(f"CHECKPOINT: [SAVE.13] Metadata file was not created: {metadata_path}")

            except Exception as e:
                logger.error(
                    f"CHECKPOINT: [SAVE.14] Failed to save metadata JSON for {metadata.path}: {str(e)}",
                    exc_info=True,
                )
                raise

            try:
                descriptions_dir = os.path.join(self.index_dir, "descriptions")
                logger.debug(f"CHECKPOINT: [SAVE.15] Descriptions directory path: {descriptions_dir}")

                if not os.path.exists(descriptions_dir):
                    logger.warning(f"CHECKPOINT: [SAVE.16] Descriptions directory does not exist, creating it")
                    os.makedirs(descriptions_dir, exist_ok=True)
                    logger.debug(f"CHECKPOINT: [SAVE.17] Descriptions directory created")
                else:
                    logger.debug(f"CHECKPOINT: [SAVE.18] Descriptions directory exists")

                description_path = os.path.join(descriptions_dir, f"{safe_path}.txt")
                logger.debug(f"CHECKPOINT: [SAVE.19] Saving description to: {description_path}")

                with open(description_path, "w", encoding="utf-8") as f:
                    f.write(metadata.description)
                logger.debug(f"CHECKPOINT: [SAVE.20] Saved description to {description_path}")

                if os.path.exists(description_path):
                    file_size = os.path.getsize(description_path)
                    logger.debug(f"CHECKPOINT: [SAVE.21] Verified description file exists, size: {file_size} bytes")
                else:
                    logger.error(f"CHECKPOINT: [SAVE.22] Description file was not created: {description_path}")

            except Exception as e:
                logger.error(
                    f"CHECKPOINT: [SAVE.23] Failed to save description for {metadata.path}: {str(e)}",
                    exc_info=True,
                )

            try:
                embeddings_dir = os.path.join(self.index_dir, "embeddings")
                logger.debug(f"CHECKPOINT: [SAVE.24] Embeddings directory path: {embeddings_dir}")

                if not os.path.exists(embeddings_dir):
                    logger.warning(f"CHECKPOINT: [SAVE.25] Embeddings directory does not exist, creating it")
                    os.makedirs(embeddings_dir, exist_ok=True)
                    logger.debug(f"CHECKPOINT: [SAVE.26] Embeddings directory created")
                else:
                    logger.debug(f"CHECKPOINT: [SAVE.27] Embeddings directory exists")

                embedding_path = os.path.join(embeddings_dir, f"{safe_path}.json")
                logger.debug(f"CHECKPOINT: [SAVE.28] Saving embeddings to: {embedding_path}")

                with open(embedding_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "path": metadata.path,
                            "chunks": metadata.chunks,
                            "embeddings": metadata.embeddings,
                        },
                        f,
                        indent=2,
                    )
                logger.debug(f"CHECKPOINT: [SAVE.29] Saved embeddings to {embedding_path}")

                if os.path.exists(embedding_path):
                    file_size = os.path.getsize(embedding_path)
                    logger.debug(f"CHECKPOINT: [SAVE.30] Verified embeddings file exists, size: {file_size} bytes")
                else:
                    logger.error(f"CHECKPOINT: [SAVE.31] Embeddings file was not created: {embedding_path}")

            except Exception as e:
                logger.error(
                    f"CHECKPOINT: [SAVE.32] Failed to save embeddings for {metadata.path}: {str(e)}",
                    exc_info=True,
                )
                raise

            logger.debug(f"CHECKPOINT: [SAVE.33] Successfully saved all metadata for {metadata.path}")

        except Exception as e:
            logger.error(
                f"CHECKPOINT: [SAVE.34] Failed to save file metadata for {metadata.path}: {str(e)}",
                exc_info=True,
            )
            raise

    def _initialize_similarity_search(self) -> None:
        """Initialize the SimilaritySearch instance.

        This creates a SimilaritySearch object that can be used by both the agent mode
        and chat features to ensure they're using the same embeddings.
        """
        embeddings_dir = os.path.join(self.index_dir, "embeddings")
        logger.debug(f"Initializing SimilaritySearch with embeddings directory: {embeddings_dir}")

        if os.path.exists(embeddings_dir):
            try:
                self.similarity_search = SimilaritySearch(embeddings_dir=embeddings_dir)
                logger.info(f"SimilaritySearch initialized successfully with {len(self.similarity_search.embeddings)} embedding files")
            except Exception as e:
                logger.error(f"Error initializing SimilaritySearch: {e}", exc_info=True)
                self.similarity_search = None
        else:
            logger.warning(f"Embeddings directory does not exist: {embeddings_dir}")
            self.similarity_search = None

    def _load_metadata_cache(self) -> None:
        """Load all existing metadata into cache."""
        metadata_dir: str = os.path.join(self.index_dir, "metadata")
        logger.debug(f"Loading metadata cache from {metadata_dir}")

        if not os.path.exists(metadata_dir):
            logger.warning(f"Metadata directory does not exist: {metadata_dir}")
            return

        try:
            metadata_files: List[str] = os.listdir(metadata_dir)
            logger.debug(f"Found {len(metadata_files)} files in metadata directory")
        except Exception as e:
            logger.error(f"Error listing metadata directory {metadata_dir}: {str(e)}", exc_info=True)
            return

        loaded_count: int = 0
        error_count: int = 0

        for metadata_file in metadata_files:
            if metadata_file.endswith(".json"):
                file_path: str = os.path.join(metadata_dir, metadata_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data: Dict[str, Any] = json.load(f)
                        self.metadata_cache[data["path"]] = {
                            "hash": data["hash"],
                            "modified_time": data["modified_time"],
                        }
                        loaded_count += 1
                        if loaded_count % 100 == 0:
                            logger.debug(f"Loaded {loaded_count} metadata files so far")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error loading metadata cache from {file_path}: {str(e)}", exc_info=True)

        logger.info(f"Loaded {loaded_count} metadata files into cache (with {error_count} errors)")

    def _should_update_file(self, entry: DirectoryEntry) -> bool:
        """Check if a file needs to be updated in the index.

        Args:
            entry (DirectoryEntry): DirectoryEntry for the file to check.

        Returns:
            bool: True if the file needs to be indexed/updated.
        """
        if not self._should_index_file(entry):
            return False

        cached_data: Optional[Dict[str, Any]] = self.metadata_cache.get(entry.path)
        if not cached_data:
            return True

        if cached_data["modified_time"] != entry.modified_time:
            return True

        if cached_data["hash"] != entry.file_hash:
            return True

        return False

    def index_directory(
        self, cancel_check_callback: Optional[Callable[[], bool]] = None
    ) -> List[FileMetadata]:
        """Index all files in the root directory using parallel processing.

        Only updates files that have changed since last indexing.

        Args:
            cancel_check_callback (Optional[Callable[[], bool]], optional): Optional callable that returns True if indexing should be cancelled. Defaults to None.

        Returns:
            List[FileMetadata]: List of metadata for all indexed files.
        """
        logger.info(f"CHECKPOINT: [1] Starting indexing for directory: {self.root_path}")
        direct_logger.log(f"CHECKPOINT: [1] Starting indexing for directory: {self.root_path}")

        indexed_files: List[FileMetadata] = []

        try:
            logger.debug(f"CHECKPOINT: [1.1] Checking if index directory exists: {self.index_dir}")
            direct_logger.log(f"CHECKPOINT: [1.1] Checking if index directory exists: {self.index_dir}")
            if not os.path.exists(self.index_dir):
                logger.warning(f"CHECKPOINT: [1.2] Index directory does not exist, attempting to create it: {self.index_dir}")
                direct_logger.log(f"CHECKPOINT: [1.2] Index directory does not exist, attempting to create it: {self.index_dir}")
                self._create_index_structure()
                logger.info(f"CHECKPOINT: [1.3] Index directory structure created successfully")
                direct_logger.log(f"CHECKPOINT: [1.3] Index directory structure created successfully")
            elif not os.access(self.index_dir, os.W_OK):
                error_msg: str = f"Index directory exists but is not writable: {self.index_dir}"
                logger.error(f"CHECKPOINT: [1.4] {error_msg}")
                direct_logger.log(f"CHECKPOINT: [1.4] {error_msg}")
                raise PermissionError(error_msg)
            else:
                logger.debug(f"CHECKPOINT: [1.5] Index directory exists and is writable: {self.index_dir}")
                direct_logger.log(f"CHECKPOINT: [1.5] Index directory exists and is writable: {self.index_dir}")
        except Exception as e:
            logger.error(f"CHECKPOINT: [1.6] Failed to verify/create index directory: {str(e)}", exc_info=True)
            direct_logger.log(f"CHECKPOINT: [1.6] Failed to verify/create index directory: {str(e)}")
            direct_logger.log(f"CHECKPOINT: Returning empty indexed_files list due to error")
            return indexed_files

        try:
            logger.debug(f"CHECKPOINT: [2] Creating DirectoryParser for {self.root_path}")
            direct_logger.log(f"CHECKPOINT: [2] Creating DirectoryParser for {self.root_path}")
            parser: DirectoryParser = DirectoryParser(
                self.root_path, gitignore_path=self.gitignore_path
            )
            logger.debug(f"CHECKPOINT: [2.1] DirectoryParser created successfully")
            direct_logger.log(f"CHECKPOINT: [2.1] DirectoryParser created successfully")
        except Exception as e:
            logger.error(f"CHECKPOINT: [2.2] Failed to create DirectoryParser: {str(e)}", exc_info=True)
            direct_logger.log(f"CHECKPOINT: [2.2] Failed to create DirectoryParser: {str(e)}")
            direct_logger.log(f"CHECKPOINT: Returning empty indexed_files list due to error")
            return indexed_files

        try:
            logger.debug(f"CHECKPOINT: [3] Starting directory tree parsing")
            direct_logger.log(f"CHECKPOINT: [3] Starting directory tree parsing")
            root_entry: DirectoryEntry = parser.parse()
            logger.debug(f"CHECKPOINT: [3.1] Directory tree parsing complete")
            direct_logger.log(f"CHECKPOINT: [3.1] Directory tree parsing complete")
        except Exception as e:
            logger.error(f"CHECKPOINT: [3.2] Failed to parse directory tree: {str(e)}", exc_info=True)
            direct_logger.log(f"CHECKPOINT: [3.2] Failed to parse directory tree: {str(e)}")
            direct_logger.log(f"CHECKPOINT: Returning empty indexed_files list due to error")
            return indexed_files

        files_to_index: List[DirectoryEntry] = []
        try:
            logger.debug(f"CHECKPOINT: [4] Starting file collection process")
            direct_logger.log(f"CHECKPOINT: [4] Starting file collection process")

            def collect_files(entry: DirectoryEntry):
                try:
                    if self._should_update_file(entry):
                        files_to_index.append(entry)
                        if len(files_to_index) % 100 == 0:
                            logger.debug(f"CHECKPOINT: [4.1] Collected {len(files_to_index)} files so far")
                            direct_logger.log(f"CHECKPOINT: [4.1] Collected {len(files_to_index)} files so far")
                except Exception as e:
                    logger.error(f"CHECKPOINT: [4.2] Error checking if file should be updated {entry.path}: {str(e)}", exc_info=True)
                    direct_logger.log(f"CHECKPOINT: [4.2] Error checking if file should be updated {entry.path}: {str(e)}")

                for child in entry.children:
                    collect_files(child)

            logger.debug(f"CHECKPOINT: [4.3] Starting recursive file collection")
            direct_logger.log(f"CHECKPOINT: [4.3] Starting recursive file collection")
            collect_files(root_entry)
            logger.info(f"CHECKPOINT: [4.4] Found {len(files_to_index)} files to index")
            direct_logger.log(f"CHECKPOINT: [4.4] Found {len(files_to_index)} files to index")
        except Exception as e:
            logger.error(f"CHECKPOINT: [4.5] Failed to collect files to index: {str(e)}", exc_info=True)
            direct_logger.log(f"CHECKPOINT: [4.5] Failed to collect files to index: {str(e)}")
            direct_logger.log(f"CHECKPOINT: Returning empty indexed_files list due to error")
            return indexed_files

        if not files_to_index:
            logger.info("CHECKPOINT: [4.6] No files need updating in the index.")
            direct_logger.log("CHECKPOINT: [4.6] No files need updating in the index.")
            return indexed_files

        if files_to_index:
            sample_files: List[DirectoryEntry] = files_to_index[:5]
            logger.debug(f"CHECKPOINT: [4.7] Sample files to index: {', '.join(os.path.basename(f.path) for f in sample_files)}")
            direct_logger.log(f"CHECKPOINT: [4.7] Sample files to index: {', '.join(os.path.basename(f.path) for f in sample_files)}")

        indexed_files = []
        failed_files = []

        try:
            logger.debug(f"CHECKPOINT: [5] Creating thread pool with {os.cpu_count()} workers")
            direct_logger.log(f"CHECKPOINT: [5] Creating thread pool with {os.cpu_count()} workers")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                logger.debug(f"CHECKPOINT: [5.1] Submitting {len(files_to_index)} files for processing")
                direct_logger.log(f"CHECKPOINT: [5.1] Submitting {len(files_to_index)} files for processing")
                future_to_file = {
                    executor.submit(self._process_single_file, entry): entry
                    for entry in files_to_index
                }
                logger.debug(f"CHECKPOINT: [5.2] All files submitted to thread pool")
                direct_logger.log(f"CHECKPOINT: [5.2] All files submitted to thread pool")

                completed_count: int = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == len(files_to_index):
                        logger.debug(f"CHECKPOINT: [5.3] Processed {completed_count}/{len(files_to_index)} files")
                        direct_logger.log(f"CHECKPOINT: [5.3] Processed {completed_count}/{len(files_to_index)} files")

                    if cancel_check_callback and cancel_check_callback():
                        logger.info("CHECKPOINT: [5.4] Indexing cancelled by user")
                        direct_logger.log("CHECKPOINT: [5.4] Indexing cancelled by user")
                        executor.shutdown(wait=False, cancel_futures=True)
                        return indexed_files

                    entry: DirectoryEntry = future_to_file[future]
                    try:
                        metadata: Optional[FileMetadata] = future.result()
                        if metadata:
                            indexed_files.append(metadata)
                            self.metadata_cache[entry.path] = {
                                "hash": entry.file_hash,
                                "modified_time": entry.modified_time,
                            }
                            if len(indexed_files) % 50 == 0:
                                logger.debug(f"CHECKPOINT: [5.5] Successfully indexed {len(indexed_files)} files so far")
                                direct_logger.log(f"CHECKPOINT: [5.5] Successfully indexed {len(indexed_files)} files so far")
                        else:
                            failed_files.append(entry.path)
                            logger.warning(f"CHECKPOINT: [5.6] Failed to index {os.path.basename(entry.path)}")
                            direct_logger.log(f"CHECKPOINT: [5.6] Failed to index {os.path.basename(entry.path)}")
                    except Exception as e:
                        failed_files.append(entry.path)
                        logger.error(f"CHECKPOINT: [5.7] Error processing file {entry.path}: {str(e)}", exc_info=True)
                        direct_logger.log(f"CHECKPOINT: [5.7] Error processing file {entry.path}: {str(e)}")
        except Exception as e:
            logger.error(f"CHECKPOINT: [5.8] Error during parallel file processing: {str(e)}", exc_info=True)
            direct_logger.log(f"CHECKPOINT: [5.8] Error during parallel file processing: {str(e)}")

        logger.info(f"CHECKPOINT: [6] Indexing process completed, generating summary")
        direct_logger.log(f"CHECKPOINT: [6] Indexing process completed, generating summary")
        if failed_files:
            logger.warning(f"CHECKPOINT: [6.1] Failed to index {len(failed_files)} files")
            direct_logger.log(f"CHECKPOINT: [6.1] Failed to index {len(failed_files)} files")
            if len(failed_files) <= 5:
                logger.warning(f"CHECKPOINT: [6.2] Failed files: {', '.join(os.path.basename(f) for f in failed_files)}")
                direct_logger.log(f"CHECKPOINT: [6.2] Failed files: {', '.join(os.path.basename(f) for f in failed_files)}")
            else:
                logger.warning(f"CHECKPOINT: [6.3] Sample of failed files: {', '.join(os.path.basename(f) for f in failed_files[:5])}")
                direct_logger.log(f"CHECKPOINT: [6.3] Sample of failed files: {', '.join(os.path.basename(f) for f in failed_files[:5])}")

        index_dir_exists: bool = os.path.exists(self.index_dir)
        logger.info(f"CHECKPOINT: [6.4] Index directory exists: {index_dir_exists}")
        direct_logger.log(f"CHECKPOINT: [6.4] Index directory exists: {index_dir_exists}")

        if index_dir_exists:
            metadata_dir: str = os.path.join(self.index_dir, "metadata")
            metadata_dir_exists: bool = os.path.exists(metadata_dir)
            logger.info(f"CHECKPOINT: [6.5] Metadata directory exists: {metadata_dir_exists}")
            direct_logger.log(f"CHECKPOINT: [6.5] Metadata directory exists: {metadata_dir_exists}")

            if metadata_dir_exists:
                try:
                    metadata_files: List[str] = os.listdir(metadata_dir)
                    logger.info(f"CHECKPOINT: [6.6] Metadata directory contains {len(metadata_files)} files")
                    direct_logger.log(f"CHECKPOINT: [6.6] Metadata directory contains {len(metadata_files)} files")
                except Exception as e:
                    logger.error(f"CHECKPOINT: [6.7] Error listing metadata directory: {str(e)}", exc_info=True)
                    direct_logger.log(f"CHECKPOINT: [6.7] Error listing metadata directory: {str(e)}")

        logger.info(f"CHECKPOINT: [7] Indexing complete. Successfully indexed {len(indexed_files)} files")
        direct_logger.log(f"CHECKPOINT: [7] Indexing complete. Successfully indexed {len(indexed_files)} files")

        return indexed_files

    def _process_single_file(self, entry: DirectoryEntry) -> Optional[FileMetadata]:
        """Process a single file for indexing.

        Args:
            entry (DirectoryEntry): DirectoryEntry for the file to process.

        Returns:
            Optional[FileMetadata]: Metadata for the processed file, or None if processing failed.
        """
        try:
            logger.debug(f"CHECKPOINT: [FILE.1] Starting to process file: {entry.path}")

            if not os.path.exists(entry.path):
                logger.error(f"CHECKPOINT: [FILE.2] File no longer exists: {entry.path}")
                return None

            if not os.access(entry.path, os.R_OK):
                logger.error(f"CHECKPOINT: [FILE.3] File is not readable: {entry.path}")
                return None

            logger.debug(f"CHECKPOINT: [FILE.4] File exists and is readable: {entry.path}")

            try:
                logger.debug(f"CHECKPOINT: [FILE.5] Extracting signatures from {entry.path}")
                signatures: List[FileSignature] = self._extract_signatures(entry.path)
                logger.debug(f"CHECKPOINT: [FILE.6] Found {len(signatures)} signatures in {entry.path}")
            except Exception as e:
                logger.error(f"CHECKPOINT: [FILE.7] Error extracting signatures from {entry.path}: {str(e)}", exc_info=True)
                signatures = []
                logger.debug(f"CHECKPOINT: [FILE.8] Continuing with empty signatures list")

            try:
                logger.debug(f"CHECKPOINT: [FILE.9] Generating description for {entry.path}")
                description: str = self._generate_description(entry.path, signatures)
                logger.debug(f"CHECKPOINT: [FILE.10] Description generated successfully for {entry.path}")
            except Exception as e:
                logger.error(f"CHECKPOINT: [FILE.11] Error generating description for {entry.path}: {str(e)}", exc_info=True)
                description = f"File: {os.path.basename(entry.path)}"
                logger.debug(f"CHECKPOINT: [FILE.12] Using fallback description: {description}")

            try:
                logger.debug(f"CHECKPOINT: [FILE.13] Processing file with code_embedder: {entry.path}")
                chunks: List[Dict[str, Any]] = []
                embeddings: Optional[List[List[float]]] = None

                try:
                    logger.debug(f"CHECKPOINT: [FILE.14] Calling code_embedder.process_file for {entry.path}")
                    chunks, embeddings = self.code_embedder.process_file(entry.path)
                    logger.debug(f"CHECKPOINT: [FILE.15] Generated {len(chunks)} chunks and embeddings for {entry.path}")
                except ValueError as e:
                    if "Unsupported language for file" in str(e):
                        logger.info(f"CHECKPOINT: [FILE.16] Using generic text processing for unsupported file type: {entry.path}")
                        try:
                            logger.debug(f"CHECKPOINT: [FILE.17] Reading file content for generic processing: {entry.path}")
                            with open(entry.path, "r", encoding="utf-8", errors="replace") as f:
                                content: str = f.read()
                            logger.debug(f"CHECKPOINT: [FILE.18] Successfully read file content, length: {len(content)} chars")

                            chunks = [
                                {
                                    "text": content,
                                    "type": "generic_text",
                                    "start_line": 1,
                                    "end_line": content.count("\n") + 1,
                                }
                            ]
                            logger.debug(f"CHECKPOINT: [FILE.19] Created generic text chunk")

                            logger.debug(f"CHECKPOINT: [FILE.20] Embedding generic text chunk")
                            embeddings = self.code_embedder.embed_chunks(chunks)
                            logger.debug(f"CHECKPOINT: [FILE.21] Generic text embedding generated successfully")
                        except Exception as inner_e:
                            logger.error(f"CHECKPOINT: [FILE.22] Error processing generic text for {entry.path}: {str(inner_e)}", exc_info=True)
                            raise
                    else:
                        logger.error(f"CHECKPOINT: [FILE.23] Error processing file: {str(e)}")
                        raise
            except Exception as e:
                logger.error(f"CHECKPOINT: [FILE.24] Failed to process file content for {entry.path}: {str(e)}", exc_info=True)
                return None

            try:
                logger.debug(f"CHECKPOINT: [FILE.25] Creating FileMetadata object for {entry.path}")
                metadata: FileMetadata = FileMetadata(
                    name=entry.name,
                    path=entry.path,
                    hash=entry.file_hash,
                    size=entry.size,
                    extension=entry.extension,
                    modified_time=entry.modified_time,
                    description=description,
                    signatures=signatures,
                    chunks=chunks,
                    embeddings=embeddings.tolist(),
                )
                logger.debug(f"CHECKPOINT: [FILE.26] FileMetadata object created successfully")

                logger.debug(f"CHECKPOINT: [FILE.27] Saving file metadata for {entry.path}")
                self._save_file_metadata(metadata)
                logger.debug(f"CHECKPOINT: [FILE.28] File metadata saved successfully for {entry.path}")
                return metadata
            except Exception as e:
                logger.error(f"CHECKPOINT: [FILE.29] Error saving metadata for {entry.path}: {str(e)}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"CHECKPOINT: [FILE.30] Unexpected error processing file {entry.path}: {str(e)}", exc_info=True)
            return None

    def load_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Load metadata for a specific file.

        Args:
            file_path (str): Path to the file.

        Returns:
            Optional[FileMetadata]: Loaded metadata or None if not found.
        """
        index_dir_abs: str = os.path.abspath(self.index_dir)
        if file_path.startswith(index_dir_abs):
            return None

        rel_path: str = os.path.relpath(file_path, self.root_path)

        if rel_path.startswith(".index"):
            return None

        safe_path: str = re.sub(r"[^\w\-_\.]", "_", rel_path)
        metadata_path: str = os.path.join(
            self.index_dir, "metadata", f"{safe_path}.json"
        )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            embedding_path: str = os.path.join(
                self.index_dir, "embeddings", f"{safe_path}.json"
            )
            with open(embedding_path, "r", encoding="utf-8") as f:
                embedding_data: Dict[str, Any] = json.load(f)

            description: Optional[str] = None

            if "description" in data:
                description = data["description"]
            else:
                try:
                    description_path: str = os.path.join(
                        self.index_dir, "descriptions", f"{safe_path}.txt"
                    )
                    if os.path.exists(description_path):
                        with open(
                            description_path, "r", encoding="utf-8"
                        ) as f:
                            description = f.read().strip()
                except Exception:
                    pass

                if not description:
                    description = f"File: {os.path.basename(file_path)}"

                data["description"] = description
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass

            return FileMetadata(
                name=data["name"],
                path=data["path"],
                hash=data["hash"],
                size=data["size"],
                extension=data.get("extension", ""),
                modified_time=data.get("modified_time", 0.0),
                description=description,
                signatures=[
                    FileSignature(**sig) for sig in data.get("signatures", [])
                ],
                chunks=embedding_data.get("chunks", []),
                embeddings=embedding_data.get("embeddings", []),
            )
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error loading metadata for {file_path}: {e}")
            return None

    def force_reindex_all(self, cancel_check_callback: Optional[Callable[[], bool]] = None) -> List[FileMetadata]:
        """Force reindex all files in the directory, regardless of their current state.

        Args:
            cancel_check_callback (Optional[Callable[[], bool]], optional): Optional callable that returns True if indexing should be cancelled. Defaults to None.

        Returns:
            List[FileMetadata]: List of metadata for all indexed files.
        """
        self.metadata_cache.clear()
        return self.index_directory(cancel_check_callback)

    def reindex_file(self, file_path: str) -> Optional[FileMetadata]:
        """Reindex a specific file, regardless of its current state.

        Args:
            file_path (str): Path to the file to reindex.

        Returns:
            Optional[FileMetadata]: Updated metadata for the file, or None if processing failed.
        """
        try:
            file_stat: os.stat_result = os.stat(file_path)
            entry: DirectoryEntry = DirectoryEntry(
                name=os.path.basename(file_path),
                path=file_path,
                parent=os.path.dirname(file_path),
                entry_type=EntryType.FILE,
                size=file_stat.st_size,
                extension=os.path.splitext(file_path)[1].lstrip("."),
                modified_time=file_stat.st_mtime,
            )

            hasher = hashlib.new(HASH_ALGORITHM)
            with open(file_path, "rb") as f:
                buffer = f.read(HASH_BUFFER_SIZE)
                while buffer:
                    hasher.update(buffer)
                    buffer = f.read(HASH_BUFFER_SIZE)
            entry.file_hash = hasher.hexdigest()

            return self._process_single_file(entry)
        except Exception as e:
            logger.error(f"Error reindexing file {file_path}: {e}")
            return None

    def update_outdated(self) -> List[FileMetadata]:
        """Update only files that are outdated (have mismatched hashes or modified times).

        This is an alias for index_directory() which already implements this logic.

        Returns:
            List[FileMetadata]: List of metadata for updated files.
        """
        return self.index_directory()

    def get_indexed_files(self) -> List[str]:
        """Get a list of all currently indexed files.

        Returns:
            List[str]: List of absolute paths to all indexed files (excluding .index files).
        """
        index_dir_abs: str = os.path.abspath(self.index_dir)

        valid_files: List[str] = []
        for file_path in self.metadata_cache.keys():
            if not file_path.startswith(index_dir_abs) and not os.path.relpath(
                file_path, self.root_path
            ).startswith(".index"):
                valid_files.append(file_path)

        return valid_files

    def get_outdated_files(self) -> List[str]:
        """Get a list of files that need to be updated.

        Returns:
            List[str]: List of absolute paths to files that need updating.
        """
        logger.debug(f"Getting outdated files for {self.root_path}")
        outdated: List[str] = []

        try:
            if not os.path.exists(self.index_dir):
                logger.warning(f"Index directory does not exist, will need to create it: {self.index_dir}")
                return self._get_all_indexable_files()

            try:
                logger.debug(f"Creating DirectoryParser")
                parser: DirectoryParser = DirectoryParser(
                    self.root_path, gitignore_path=self.gitignore_path
                )
            except Exception as e:
                logger.error(f"Failed to create DirectoryParser: {str(e)}", exc_info=True)
                raise

            try:
                logger.debug(f"Parsing directory tree")
                root_entry: DirectoryEntry = parser.parse()
                logger.debug(f"Directory tree parsing complete")
            except Exception as e:
                logger.error(f"Failed to parse directory tree: {str(e)}", exc_info=True)
                raise

            def check_entry(entry: DirectoryEntry):
                try:
                    should_update: bool = self._should_update_file(entry)
                    if should_update:
                        outdated.append(entry.path)
                        if len(outdated) % 100 == 0:
                            logger.debug(f"Found {len(outdated)} outdated files so far")
                except Exception as e:
                    logger.error(f"Error checking if file should be updated {entry.path}: {str(e)}", exc_info=True)
                    outdated.append(entry.path)

                for child in entry.children:
                    check_entry(child)

            logger.debug(f"Checking for outdated files")
            check_entry(root_entry)
            logger.info(f"Found {len(outdated)} outdated files")

            if outdated:
                sample_files: List[str] = outdated[:5]
                logger.debug(f"Sample outdated files: {', '.join(os.path.basename(f) for f in sample_files)}")

        except Exception as e:
            logger.error(f"Error getting outdated files: {str(e)}", exc_info=True)
            return []

        return outdated

    def _get_all_indexable_files(self) -> List[str]:
        """Get a list of all files that should be indexed.

        This is used when no index exists yet.

        Returns:
            List[str]: List of absolute paths to all files that should be indexed.
        """
        logger.debug(f"Getting all indexable files for {self.root_path}")
        indexable_files: List[str] = []

        try:
            parser: DirectoryParser = DirectoryParser(
                self.root_path, gitignore_path=self.gitignore_path
            )

            root_entry: DirectoryEntry = parser.parse()

            def collect_files(entry: DirectoryEntry):
                try:
                    if self._should_index_file(entry):
                        indexable_files.append(entry.path)
                        if len(indexable_files) % 100 == 0:
                            logger.debug(f"Found {len(indexable_files)} indexable files so far")
                except Exception as e:
                    logger.error(f"Error checking if file should be indexed {entry.path}: {str(e)}", exc_info=True)

                for child in entry.children:
                    collect_files(child)

            collect_files(root_entry)
            logger.info(f"Found {len(indexable_files)} indexable files")

            if indexable_files:
                sample_files: List[str] = indexable_files[:5]
                logger.debug(f"Sample indexable files: {', '.join(os.path.basename(f) for f in sample_files)}")

        except Exception as e:
            logger.error(f"Error getting all indexable files: {str(e)}", exc_info=True)

        return indexable_files

    def get_sample_files(self, count: int = 5) -> List[str]:
        """Get a sample of indexed files.

        Args:
            count (int): Number of files to return.

        Returns:
            List[str]: List of file paths.
        """
        logger.debug(f"Getting sample of {count} indexed files")
        metadata_dir: str = os.path.join(self.index_dir, "metadata")
        if not os.path.exists(metadata_dir):
            logger.warning(f"Metadata directory does not exist: {metadata_dir}")
            return []

        files: List[str] = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        sample: List[str] = files[:count] if len(files) > count else files

        result: List[str] = []
        for file in sample:
            try:
                with open(os.path.join(metadata_dir, file), 'r', encoding='utf-8') as f:
                    data: Dict[str, Any] = json.load(f)
                    result.append(data.get('path', file))
            except Exception as e:
                logger.error(f"Error reading metadata file {file}: {str(e)}", exc_info=True)
                result.append(file)

        logger.debug(f"Returning {len(result)} sample files")
        return result

    def cleanup_index_files(self) -> int:
        """Remove any index files from the index (files within the .index directory or starting with .index).

        This is useful after upgrading to a new version that properly excludes index files.

        Returns:
            int: Number of files removed from the index.
        """
        index_dir_abs: str = os.path.abspath(self.index_dir)
        indexed_files: List[str] = list(self.metadata_cache.keys())
        removed_count: int = 0

        for file_path in indexed_files:
            if file_path.startswith(index_dir_abs):
                self.metadata_cache.pop(file_path, None)
                removed_count += 1

                try:
                    rel_path: str = os.path.relpath(file_path, self.root_path)
                    safe_path: str = re.sub(r"[^\w\-_\.]", "_", rel_path)

                    metadata_path: str = os.path.join(
                        self.index_dir, "metadata", f"{safe_path}.json"
                    )
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)

                    embedding_path: str = os.path.join(
                        self.index_dir, "embeddings", f"{safe_path}.json"
                    )
                    if os.path.exists(embedding_path):
                        os.remove(embedding_path)

                    description_path: str = os.path.join(
                        self.index_dir, "descriptions", f"{safe_path}.txt"
                    )
                    if os.path.exists(description_path):
                        os.remove(description_path)
                except Exception as e:
                    logger.error(f"Error removing index files for {file_path}: {e}")

            elif os.path.relpath(file_path, self.root_path).startswith(".index"):
                self.metadata_cache.pop(file_path, None)
                removed_count += 1

                try:
                    rel_path = os.path.relpath(file_path, self.root_path)
                    safe_path = re.sub(r"[^\w\-_\.]", "_", rel_path)

                    metadata_path = os.path.join(
                        self.index_dir, "metadata", f"{safe_path}.json"
                    )
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)

                    embedding_path = os.path.join(
                        self.index_dir, "embeddings", f"{safe_path}.json"
                    )
                    if os.path.exists(embedding_path):
                        os.remove(embedding_path)

                    description_path = os.path.join(
                        self.index_dir, "descriptions", f"{safe_path}.txt"
                    )
                    if os.path.exists(description_path):
                        os.remove(description_path)
                except Exception as e:
                    logger.error(f"Error removing index files for {file_path}: {e}")

        return removed_count

    def is_index_complete(self) -> Dict[str, Any]:
        """Check if the index is complete for the current directory.

        This method verifies:
        1. If an index exists
        2. If all files in the directory are indexed
        3. If all indexed files are up-to-date (no hash mismatches)

        Returns:
            Dict[str, Any]: Dictionary containing:
            - 'complete' (bool): True if index is complete, False otherwise
            - 'reason' (str): Reason why index is not complete (if applicable)
            - 'outdated_count' (int): Number of outdated files (if applicable)
            - 'missing_count' (int): Number of files missing from index (if applicable)
            - 'ignored_count' (int): Number of files ignored due to gitignore/exclusions
        """
        logger.debug(f"Checking if index is complete for {self.root_path}")
        result: Dict[str, Any] = {
            "complete": False,
            "reason": "",
            "outdated_count": 0,
            "missing_count": 0,
            "ignored_count": 0,
        }

        if not os.path.exists(self.index_dir):
            logger.debug(f"Index directory does not exist: {self.index_dir}")
            result["reason"] = "Index directory does not exist"
            return result

        metadata_dir = os.path.join(self.index_dir, "metadata")
        embeddings_dir = os.path.join(self.index_dir, "embeddings")
        descriptions_dir = os.path.join(self.index_dir, "descriptions")

        logger.debug(f"Checking index directory structure:")
        logger.debug(f"- metadata_dir exists: {os.path.exists(metadata_dir)}")
        logger.debug(f"- embeddings_dir exists: {os.path.exists(embeddings_dir)}")
        logger.debug(f"- descriptions_dir exists: {os.path.exists(descriptions_dir)}")

        if not all(
            os.path.exists(d) for d in [metadata_dir, embeddings_dir, descriptions_dir]
        ):
            result["reason"] = "Index directory structure is incomplete"
            return result

        logger.debug(f"Creating DirectoryParser for {self.root_path}")
        parser = DirectoryParser(
            self.root_path, gitignore_path=self.gitignore_path
        )
        logger.debug(f"Parsing directory tree")
        root_entry = parser.parse()
        logger.debug(f"Directory tree parsing complete")

        current_files = []
        missing_files = []
        ignored_files = []

        def collect_files(entry: DirectoryEntry):
            if parser._is_ignored(entry.path, entry.is_folder()):
                ignored_files.append(entry.path)
                return

            if self._should_index_file(entry):
                current_files.append(entry)
                norm_path = os.path.normpath(entry.path).lower()
                found = False
                for cache_path in self.metadata_cache.keys():
                    if os.path.normpath(cache_path).lower() == norm_path:
                        found = True
                        break
                if not found:
                    missing_files.append(entry.path)
            for child in entry.children:
                collect_files(child)

        logger.debug(f"Collecting files to index")
        collect_files(root_entry)
        logger.debug(f"File collection complete")

        result["missing_count"] = len(missing_files)
        result["ignored_count"] = len(ignored_files)

        logger.info(
            f"Found {len(current_files)} indexable files, {len(ignored_files)} ignored files, {len(missing_files)} missing from index"
        )

        if missing_files:
            sample_files = missing_files[:5]
            logger.debug(f"Sample missing files: {', '.join(os.path.basename(f) for f in sample_files)}")
            result['reason'] = f"{len(missing_files)} files are not indexed"
            return result

        logger.debug(f"Checking for outdated files")
        outdated_files = []
        for entry in current_files:
            norm_path = os.path.normpath(entry.path).lower()
            matched_cache_path = None

            for cache_path in self.metadata_cache.keys():
                if os.path.normpath(cache_path).lower() == norm_path:
                    matched_cache_path = cache_path
                    break

            if not matched_cache_path:
                logger.debug(f"No matched cache path for {entry.path}")
                continue

            cached_data = self.metadata_cache[matched_cache_path]

            time_diff = abs(cached_data['modified_time'] - entry.modified_time)
            time_mismatch = time_diff > 1.0

            if time_mismatch or cached_data['hash'] != entry.file_hash:
                outdated_files.append(entry.path)

                if time_mismatch:
                    logger.debug(f"Modified time mismatch for {os.path.basename(entry.path)}: "
                          f"cached={cached_data['modified_time']}, current={entry.modified_time}")
                if cached_data['hash'] != entry.file_hash:
                    logger.debug(f"Hash mismatch for {os.path.basename(entry.path)}: "
                          f"cached={cached_data['hash']}, current={entry.file_hash}")

        result['outdated_count'] = len(outdated_files)
        if outdated_files:
            logger.info(f"Found {len(outdated_files)} outdated files")
            sample_files = outdated_files[:5]
            logger.debug(f"Sample outdated files: {', '.join(os.path.basename(f) for f in sample_files)}")
            result['reason'] = f"{len(outdated_files)} files are outdated"
            return result

        logger.info(f"Index is complete!")
        result['complete'] = True
        return result