import concurrent.futures
import fnmatch
import hashlib
import logging
import os
import psutil
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv

logger = logging.getLogger("VerbalCodeAI.Code.DirectoryParser")
logger.info("[DIRECTORY PARSER] LOGGER WORKING")

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)

HASH_ALGORITHM: str = "md5"
HASH_BUFFER_SIZE: int = 1024 * 1024
MAX_HASH_SIZE: int = 100 * 1024 * 1024

PERFORMANCE_MODE = os.getenv("PERFORMANCE_MODE", "MEDIUM").upper()
MAX_THREADS_ENV = os.getenv("MAX_THREADS")

CPU_COUNT = os.cpu_count() or 4
TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024 * 1024 * 1024)


def calculate_max_workers() -> int:
    """Calculate the maximum number of worker threads based on performance mode."""
    if MAX_THREADS_ENV and MAX_THREADS_ENV.isdigit():
        return int(MAX_THREADS_ENV)

    if PERFORMANCE_MODE == "LOW":
        return max(2, CPU_COUNT // 2)
    elif PERFORMANCE_MODE == "MEDIUM":
        return min(16, CPU_COUNT)
    elif PERFORMANCE_MODE == "MAX":
        return min(32, CPU_COUNT * 2)
    else:
        return min(16, CPU_COUNT)


def calculate_memory_workers() -> int:
    """Calculate the number of memory-intensive worker threads based on performance mode."""
    if PERFORMANCE_MODE == "LOW":
        return max(1, CPU_COUNT // 4)
    elif PERFORMANCE_MODE == "MEDIUM":
        return max(2, CPU_COUNT // 2)
    elif PERFORMANCE_MODE == "MAX":
        return max(2, CPU_COUNT - 1)
    else:
        return max(2, CPU_COUNT // 2)


MAX_WORKERS: int = calculate_max_workers()
MEMORY_WORKERS: int = calculate_memory_workers()

if PERFORMANCE_MODE == "LOW":
    LRU_CACHE_SIZE: int = 5000
    HASH_CACHE_SIZE: int = 500
elif PERFORMANCE_MODE == "MAX":
    LRU_CACHE_SIZE: int = 20000
    HASH_CACHE_SIZE: int = 2000
else:
    LRU_CACHE_SIZE: int = 10000
    HASH_CACHE_SIZE: int = 1000

logger.info(f"Performance mode: {PERFORMANCE_MODE}")
logger.info(f"Max workers: {MAX_WORKERS}")
logger.info(f"Memory workers: {MEMORY_WORKERS}")
logger.info(f"LRU cache size: {LRU_CACHE_SIZE}")
logger.info(f"Hash cache size: {HASH_CACHE_SIZE}")

DEFAULT_IGNORED_NAMES: Set[str] = {
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "env/",
    "venv/",
    "ENV/",
    ".idea",
    ".vs",
    ".vscode",
    ".DS_Store",
    "*.swp",
    "*.swo",
    "*~",
    "node_modules",
    "bower_components",
    "jspm_packages",
    "build/",
    "dist/",
    "*.egg-info/",
    ".eggs/",
    ".index",
    ".cache",
    ".pytest_cache",
    ".coverage",
    "htmlcov/",
    "*.log",
    "*.tmp",
    "temp/",
    "tmp/",
    "temp_*",
    ".history",
    "*.bak",
}


class EntryType(Enum):
    """Enumeration for entry types in a directory."""

    FILE: auto = auto()
    FOLDER: auto = auto()


@dataclass
class DirectoryEntry:
    """
    Represents a file or folder in a directory tree.

    Attributes:
        name (str): Name of the file or folder.
        path (str): Absolute path to the entry.
        parent (str): Absolute path to the parent directory.
        entry_type (EntryType): EntryType.FILE or EntryType.FOLDER.
        size (int): Size in bytes (files only; 0 for folders).
        extension (str): File extension (files only; empty for folders).
        file_hash (str): File hash (files only; empty for folders or if skipped).
        modified_time (float): Last modified timestamp.
        children (List['DirectoryEntry']): List of DirectoryEntry objects (folders only).

    Performance:
        - Uses dataclass for efficient object creation and memory usage.
        - File hashing is performed as part of the parsing and can be parallelized.
    """

    name: str
    path: str
    parent: str
    entry_type: EntryType
    size: int = 0
    extension: str = ""
    file_hash: str = ""
    modified_time: float = 0.0
    children: List["DirectoryEntry"] = field(default_factory=list)

    def is_file(self) -> bool:
        """Returns True if this entry is a file."""
        return self.entry_type == EntryType.FILE

    def is_folder(self) -> bool:
        """Returns True if this entry is a folder."""
        return self.entry_type == EntryType.FOLDER

    def __repr__(self) -> str:
        """Provides a string representation of the DirectoryEntry."""
        return (
            f"<DirectoryEntry name={self.name!r} type={self.entry_type.name} "
            f"size={self.size} path={self.path!r}>"
        )


def _glob_to_regex_segment(segment: str) -> str:
    """
    Converts a single glob path segment (without '**') to a regex segment.

    Args:
        segment (str): The glob path segment to convert.

    Returns:
        str: The converted regex segment.
    """
    return re.escape(segment).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")


class GitIgnorePattern:
    """
    Represents a single .gitignore pattern and provides matching capabilities.

    Attributes:
        original_pattern (str): The raw pattern string from the .gitignore file.
        base_dir (str): The absolute path to the directory containing the .gitignore file.
        is_negation (bool): True if the pattern is a negation (starts with '!').
        is_directory_only (bool): True if the pattern specifically targets directories (ends with '/').
        raw_pattern (str): The core pattern string after stripping '!', leading/trailing '/'.
        regex (re.Pattern): The compiled regular expression for this pattern.
        match_all_files (bool): True if the pattern is '**', which matches all files and directories.
        contains_slash (bool): True if the original pattern (after stripping '!') contained a '/'.
        is_anchored_to_base (bool): True if the original pattern started with '/'.

    Implementation Notes:
    - Handles `!` for negation.
    - Handles patterns ending with `/` for directories.
    - Handles patterns starting with `/` for anchoring to `base_dir`.
    - Converts `*`, `?`, and `**` glob-like syntax into regex.
        - `*`: Matches any sequence of characters except `/`.
        - `?`: Matches any single character except `/`.
        - `**`: Matches any sequence of characters including `/`, used to span directories.
          - `**/foo`: Matches `foo` in any directory.
          - `foo/**`: Matches everything under `foo`.
          - `foo/**/bar`: Matches `bar` under `foo` with any intermediate directories.
    - Patterns without `/` can match at any directory level.
    - Patterns with `/` are matched relative to `base_dir`.

    Performance:
    - Regex is pre-compiled during initialization for efficient matching.
    """

    def __init__(self, pattern_str: str, base_dir: str) -> None:
        """
        Initializes a GitIgnorePattern object.

        Args:
            pattern_str (str): The pattern string from the .gitignore file.
            base_dir (str): The base directory for the .gitignore file.
        """
        self.original_pattern: str = pattern_str
        self.base_dir: str = os.path.normpath(base_dir)

        self.is_negation: bool = pattern_str.startswith("!")
        if self.is_negation:
            pattern_str = pattern_str[1:]

        self.is_directory_only: bool = pattern_str.endswith("/")
        if self.is_directory_only:
            pattern_str = pattern_str[:-1]

        self.is_anchored_to_base: bool = pattern_str.startswith("/")
        if self.is_anchored_to_base:
            pattern_str = pattern_str[1:]

        self.raw_pattern: str = pattern_str
        self.contains_slash: bool = "/" in self.raw_pattern

        if self.raw_pattern == "**":
            self.regex = re.compile(".*")
            self.match_all_files = True
            return
        else:
            self.match_all_files = False

        regex_expr_str: str
        if "**" in self.raw_pattern:
            if (
                self.raw_pattern.startswith("**/")
                and self.raw_pattern.endswith("/**")
                and len(self.raw_pattern) >= 5
            ):
                middle_part = self.raw_pattern[3:-3]
                regex_expr_str = f"(?:.*/)?{_glob_to_regex_segment(middle_part)}(?:/.*)?"
            elif self.raw_pattern.startswith("**/"):
                actual_pattern_part = self.raw_pattern[3:]
                regex_expr_str = f"(?:.*/)?{_glob_to_regex_segment(actual_pattern_part)}"
            elif self.raw_pattern.endswith("/**"):
                actual_pattern_part = self.raw_pattern[:-3]
                regex_expr_str = f"{_glob_to_regex_segment(actual_pattern_part)}(?:/.*)?"
            else:
                sub_patterns = self.raw_pattern.split("**")
                regex_parts_list = [_glob_to_regex_segment(p) for p in sub_patterns]
                regex_expr_str = "(.*)".join(regex_parts_list)
        else:
            regex_expr_str = _glob_to_regex_segment(self.raw_pattern)

        if self.is_anchored_to_base or self.contains_slash:
            final_regex_str = f"^{regex_expr_str}"
            if self.is_directory_only:
                final_regex_str += "(?:$|/.*)"
            else:
                final_regex_str += "$"
            self.regex = re.compile(final_regex_str)
        else:
            if self.is_directory_only:
                self.regex = re.compile(regex_expr_str + "$")
            else:
                self.regex = re.compile(regex_expr_str + "$")

    def matches(self, abs_path: str, is_dir: bool) -> bool:
        """
        Checks if the given absolute path matches this gitignore pattern.

        Args:
            abs_path (str): The absolute path of the file or directory to check.
            is_dir (bool): True if the abs_path refers to a directory, False otherwise.

        Returns:
            bool: True if the path matches the pattern, False otherwise.

        Performance:
            O(1) for regex matching after path normalization. Path normalization may take
            time proportional to path length.
        """
        if self.match_all_files:
            return not (self.is_directory_only and not is_dir)

        if self.is_directory_only and not is_dir:
            return False

        norm_abs_path = os.path.normpath(abs_path)
        try:
            rel_path = os.path.relpath(norm_abs_path, self.base_dir)
        except ValueError:
            return False

        rel_path = rel_path.replace(os.sep, "/")

        if rel_path.startswith("../") or rel_path == "..":
            return False

        if self.is_anchored_to_base or self.contains_slash:
            return bool(self.regex.fullmatch(rel_path))
        else:
            if self.regex.fullmatch(os.path.basename(rel_path)):
                return True

            current_path_segment = rel_path
            while True:
                if self.regex.fullmatch(os.path.basename(current_path_segment)):
                    return True

                parent_path_segment = os.path.dirname(current_path_segment)
                if parent_path_segment == current_path_segment or not parent_path_segment:
                    break
                current_path_segment = parent_path_segment

            return False


class DirectoryParser:
    """
    Parses a directory tree, builds a representation, optionally applying .gitignore rules
    and calculating file hashes.

    Args:
        directory_path (str): Path to the root directory to parse.
        gitignore_path (Optional[str]): Optional path to a .gitignore file.
        parallel (bool): Whether to use parallel processing for directory traversal and hashing.
        hash_files (bool): Whether to calculate file hashes.
        extra_exclude_patterns (Optional[List[str]]): A list of additional filename patterns to ignore.

    Raises:
        FileNotFoundError: If the `directory_path` does not exist.
        ValueError: If `directory_path` is not a directory.

    Performance:
        - Uses `os.scandir()` for efficient directory listing.
        - Optionally uses a `ThreadPoolExecutor` for parallel processing of subdirectories and file hashing.
        - Caches results of .gitignore checks to avoid redundant computations.
        - File hashing uses buffered reading and skips very large files.
    """

    def __init__(
        self,
        directory_path: str,
        gitignore_path: Optional[str] = None,
        parallel: bool = True,
        hash_files: bool = True,
        extra_exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes a DirectoryParser object.

        Args:
            directory_path (str): Path to the root directory to parse.
            gitignore_path (str): Optional path to a .gitignore file.
            parallel (bool): Whether to use parallel processing for directory traversal and hashing.
            hash_files (bool): Whether to calculate file hashes.
            extra_exclude_patterns (List[str]): A list of additional filename patterns to ignore.
        """
        self.root_directory_path: str = os.path.abspath(directory_path)
        if not os.path.exists(self.root_directory_path):
            raise FileNotFoundError(f"Directory not found: {self.root_directory_path}")
        if not os.path.isdir(self.root_directory_path):
            raise ValueError(f"Not a directory: {self.root_directory_path}")

        self.gitignore_rules_path: Optional[str] = (
            os.path.abspath(gitignore_path) if gitignore_path else None
        )
        self.gitignore_patterns: List[GitIgnorePattern] = []
        self.logger: logging.Logger = logger
        self.use_parallel_processing: bool = parallel
        self.calculate_hashes: bool = hash_files
        self.extra_exclude_patterns: List[str] = (
            extra_exclude_patterns if extra_exclude_patterns else []
        )

        self._ignored_paths_cache: Dict[str, bool] = {}

        self._load_gitignore_rules()

    def _load_gitignore_rules(self) -> None:
        """
        Loads and compiles .gitignore patterns from the specified file.
        Patterns are relative to the directory containing the .gitignore file.

        Performance: O(N*M) where N is lines in .gitignore and M is avg pattern complexity for regex compilation.
        """
        if not self.gitignore_rules_path or not os.path.isfile(
            self.gitignore_rules_path
        ):
            self.logger.info(
                f"No .gitignore file specified or found at: {self.gitignore_rules_path}"
            )
            return

        gitignore_base_dir = os.path.dirname(self.gitignore_rules_path)
        try:
            with open(self.gitignore_rules_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    pattern_str = line.strip()
                    if not pattern_str or pattern_str.startswith("#"):
                        continue
                    try:
                        self.gitignore_patterns.append(
                            GitIgnorePattern(pattern_str, gitignore_base_dir)
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Error compiling gitignore pattern '{pattern_str}' from "
                            f"{self.gitignore_rules_path}:{line_num} - {e}"
                        )
            self.logger.info(
                f"Loaded {len(self.gitignore_patterns)} patterns from {self.gitignore_rules_path}"
            )
        except IOError as e:
            self.logger.error(
                f"Failed to read .gitignore file at {self.gitignore_rules_path}: {e}"
            )

    def _is_ignored(self, abs_path: str, is_dir: bool) -> bool:
        """
        Checks if a given path should be ignored based on .gitignore patterns
        and common ignored names.

        Args:
            abs_path (str): The absolute path to check.
            is_dir (bool): True if the path is a directory, False otherwise.

        Returns:
            bool: True if the path should be ignored, False otherwise.

        Performance:
            O(P) where P is the number of .gitignore patterns (worst case).
            Uses a cache, so O(1) on average for repeated checks of the same path.
        """
        if abs_path in self._ignored_paths_cache:
            return self._ignored_paths_cache[abs_path]

        path_basename = os.path.basename(abs_path)
        if path_basename in DEFAULT_IGNORED_NAMES:
            self._ignored_paths_cache[abs_path] = True
            return True

        for pattern in self.extra_exclude_patterns:
            if fnmatch.fnmatch(path_basename, pattern):
                self._ignored_paths_cache[abs_path] = True
                return True

        ignored_status = False
        if not self.gitignore_patterns:
            self._ignored_paths_cache[abs_path] = False
            return False

        for pattern_obj in self.gitignore_patterns:
            if pattern_obj.matches(abs_path, is_dir):
                ignored_status = not pattern_obj.is_negation

        self._ignored_paths_cache[abs_path] = ignored_status
        return ignored_status

    @lru_cache(maxsize=HASH_CACHE_SIZE)
    def _calculate_file_hash(self, file_path: str, modified_time: float = 0.0) -> str:
        """
        Calculates the hash of a file's content with caching.

        Args:
            file_path (str): Absolute path to the file.
            modified_time (float): Last modified timestamp for cache invalidation.

        Returns:
            str: Hexadecimal string representation of the hash, or a size-based pseudo-hash
                 for very large files, or an empty string if hashing is disabled or fails.

        Performance:
            O(S/B) where S is file size and B is buffer size for I/O. Reads file in chunks.
            Skips hashing for files > MAX_HASH_SIZE.
            Uses LRU cache to avoid re-hashing unchanged files.
        """
        if not self.calculate_hashes:
            return ""

        try:
            stat_info = os.stat(file_path)
            file_size = stat_info.st_size

            if file_size > MAX_HASH_SIZE:
                self.logger.debug(
                    f"Skipping hash for large file: {file_path} ({file_size} bytes)"
                )
                return f"size:{file_size}"

            if file_size == 0:
                return "empty"

            if file_size > HASH_BUFFER_SIZE * 10:
                return self._calculate_hash_mmap(file_path, file_size)

            hasher = hashlib.new(HASH_ALGORITHM)
            with open(file_path, "rb") as f:
                buffer = f.read(HASH_BUFFER_SIZE)
                while buffer:
                    hasher.update(buffer)
                    buffer = f.read(HASH_BUFFER_SIZE)
            return hasher.hexdigest()

        except FileNotFoundError:
            self.logger.warning(
                f"File not found during hash calculation: {file_path}"
            )
        except PermissionError:
            self.logger.warning(
                f"Permission denied during hash calculation for: {file_path}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash for {file_path}: {e}")
        return ""

    def _calculate_hash_mmap(self, file_path: str, file_size: int) -> str:
        """
        Calculate file hash using memory-mapped files for better performance with large files.

        Args:
            file_path (str): Path to the file.
            file_size (int): Size of the file in bytes.

        Returns:
            str: Hexadecimal hash string.
        """
        try:
            import mmap

            hasher = hashlib.new(HASH_ALGORITHM)

            with open(file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    chunk_size = min(HASH_BUFFER_SIZE * 4, file_size)
                    for i in range(0, file_size, chunk_size):
                        end = min(i + chunk_size, file_size)
                        hasher.update(mm[i:end])

            return hasher.hexdigest()

        except (ImportError, PermissionError, OSError) as e:
            self.logger.debug(
                f"Memory mapping failed for {file_path}, falling back to standard method: {e}"
            )
            hasher = hashlib.new(HASH_ALGORITHM)
            with open(file_path, "rb") as f:
                buffer = f.read(HASH_BUFFER_SIZE)
                while buffer:
                    hasher.update(buffer)
                    buffer = f.read(HASH_BUFFER_SIZE)
            return hasher.hexdigest()

    def _parse_single_entry(
        self, dir_entry: os.DirEntry, current_parent_path: str
    ) -> Optional[DirectoryEntry]:
        """
        Parses a single os.DirEntry object (file or subdirectory).
        This is the target for parallel execution.

        Args:
            dir_entry (os.DirEntry): The directory entry to parse.
            current_parent_path (str): The path of the parent directory.

        Returns:
            Optional[DirectoryEntry]: A DirectoryEntry object representing the parsed entry, or None if the entry should be ignored.
        """
        entry_abs_path = dir_entry.path
        entry_name = dir_entry.name

        try:
            is_dir = dir_entry.is_dir()
        except OSError as e:
            self.logger.warning(f"Cannot determine type of {entry_abs_path}: {e}")
            return None

        if self._is_ignored(entry_abs_path, is_dir):
            self.logger.debug(f"Ignoring path: {entry_abs_path}")
            return None

        entry_type = EntryType.FOLDER if is_dir else EntryType.FILE
        file_size = 0
        extension = ""
        file_hash_val = ""
        children_entries: List[DirectoryEntry] = []

        modified_time = 0.0
        try:
            stat_info = dir_entry.stat()
            modified_time = stat_info.st_mtime
        except OSError as e:
            self.logger.warning(f"Cannot stat {entry_abs_path}: {e}")

        if is_dir:
            try:
                children_entries = self._parse_directory_contents(entry_abs_path)
            except PermissionError as e:
                self.logger.warning(
                    f"Permission denied listing directory {entry_abs_path}: {e}"
                )
            except OSError as e:
                self.logger.warning(
                    f"Could not list directory {entry_abs_path}: {e}"
                )

        else:
            try:
                file_size = (
                    stat_info.st_size if stat_info else dir_entry.stat().st_size
                )

                if "." in entry_name:
                    ext_part = entry_name.rsplit(".", 1)
                    if len(ext_part) > 1 and ext_part[1]:
                        extension = ext_part[1]

                file_hash_val = self._calculate_file_hash(
                    entry_abs_path, modified_time
                )
            except FileNotFoundError:
                self.logger.warning(
                    f"File not found while processing: {entry_abs_path}"
                )
                return None
            except PermissionError:
                self.logger.warning(
                    f"Permission denied while processing file: {entry_abs_path}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to get file info for {entry_abs_path}: {e}"
                )

        return DirectoryEntry(
            name=entry_name,
            path=entry_abs_path,
            parent=current_parent_path,
            entry_type=entry_type,
            size=file_size,
            extension=extension,
            file_hash=file_hash_val,
            modified_time=modified_time,
            children=children_entries,
        )

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _is_ignored_cached(self, abs_path: str, is_dir: bool) -> bool:
        """
        Cached version of _is_ignored for better performance.

        Args:
            abs_path (str): The absolute path to check.
            is_dir (bool): True if the path is a directory, False otherwise.

        Returns:
            bool: True if the path should be ignored, False otherwise.
        """
        return self._is_ignored(abs_path, is_dir)

    def _parse_directory_contents(self, dir_path: str) -> List[DirectoryEntry]:
        """
        Helper to parse contents of a given directory path with optimized batch processing.

        This implementation:
        1. Scans the entire directory at once
        2. Filters ignored entries in a batch
        3. Processes files and directories separately for better parallelism
        4. Uses adaptive thread pool sizes based on workload type

        Args:
            dir_path (str): The directory path to parse.

        Returns:
            List[DirectoryEntry]: A list of DirectoryEntry objects representing the contents of the directory.
        """
        children_entries: List[DirectoryEntry] = []

        try:
            scanned_entries = list(os.scandir(dir_path))
        except PermissionError as e:
            self.logger.warning(
                f"Permission denied listing directory {dir_path}: {e}"
            )
            return children_entries
        except OSError as e:
            self.logger.warning(
                f"Could not list directory {dir_path}: {e}"
            )
            return children_entries

        if not scanned_entries:
            return children_entries

        filtered_entries = []
        for entry in scanned_entries:
            try:
                is_dir = entry.is_dir()
                if not self._is_ignored_cached(entry.path, is_dir):
                    filtered_entries.append((entry, is_dir))
            except OSError:
                continue

        if not filtered_entries:
            return children_entries

        files = [
            (entry, dir_path) for entry, is_dir in filtered_entries if not is_dir
        ]
        directories = [
            (entry, dir_path) for entry, is_dir in filtered_entries if is_dir
        ]

        if self.use_parallel_processing:
            if files and len(files) > 5:
                file_workers = min(MAX_WORKERS, len(files) + 1)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=file_workers
                ) as executor:
                    file_futures = {
                        executor.submit(
                            self._parse_single_entry, entry, parent
                        ): (entry, parent)
                        for entry, parent in files
                    }
                    for future in concurrent.futures.as_completed(file_futures):
                        try:
                            result = future.result()
                            if result:
                                children_entries.append(result)
                        except Exception as e:
                            entry_info = file_futures[future]
                            self.logger.error(
                                f"Error processing file {entry_info[0].path}: {e}"
                            )
            else:
                for entry, parent in files:
                    try:
                        result = self._parse_single_entry(entry, parent)
                        if result:
                            children_entries.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing file {entry.path}: {e}"
                        )

            if directories and len(directories) > 2:
                dir_workers = min(MEMORY_WORKERS, len(directories) + 1)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=dir_workers
                ) as executor:
                    dir_futures = {
                        executor.submit(
                            self._parse_single_entry, entry, parent
                        ): (entry, parent)
                        for entry, parent in directories
                    }
                    for future in concurrent.futures.as_completed(dir_futures):
                        try:
                            result = future.result()
                            if result:
                                children_entries.append(result)
                        except Exception as e:
                            entry_info = dir_futures[future]
                            self.logger.error(
                                f"Error processing directory {entry_info[0].path}: {e}"
                            )
            else:
                for entry, parent in directories:
                    try:
                        result = self._parse_single_entry(entry, parent)
                        if result:
                            children_entries.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing directory {entry.path}: {e}"
                        )
        else:
            for entry, is_dir in filtered_entries:
                try:
                    result = self._parse_single_entry(entry, dir_path)
                    if result:
                        children_entries.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Error processing entry {entry.path}: {e}"
                    )

        return children_entries

    def parse(self) -> DirectoryEntry:
        """
        Parses the root directory and its entire tree with performance benchmarking.

        Returns:
            DirectoryEntry: A DirectoryEntry object representing the root of the parsed directory tree.

        Performance:
            Traversal is O(N) where N is total number of files/dirs.
            Each entry involves stat calls, .gitignore checks (cached), and potentially hashing.
            Parallelism can speed up on multi-core systems, especially with I/O bound tasks.
        """
        self.logger.info(
            f"Starting directory tree parsing for: {self.root_directory_path}"
        )

        metrics = {
            "start_time": time.monotonic(),
            "file_count": 0,
            "dir_count": 0,
            "total_size": 0,
            "ignored_count": 0,
            "hashed_count": 0,
            "memory_before": psutil.Process().memory_info().rss / (1024 * 1024),
        }

        root_name = os.path.basename(self.root_directory_path)
        root_parent = os.path.dirname(self.root_directory_path)

        root_children = self._parse_directory_contents(self.root_directory_path)

        root_stat = os.stat(self.root_directory_path)

        root_entry = DirectoryEntry(
            name=root_name,
            path=self.root_directory_path,
            parent=root_parent,
            entry_type=EntryType.FOLDER,
            size=0,
            modified_time=root_stat.st_mtime,
            children=root_children,
        )

        def collect_metrics(entry: DirectoryEntry) -> None:
            if entry.is_folder():
                metrics["dir_count"] += 1
                for child in entry.children:
                    collect_metrics(child)
            else:
                metrics["file_count"] += 1
                metrics["total_size"] += entry.size
                if (
                    entry.file_hash
                    and entry.file_hash != "empty"
                    and not entry.file_hash.startswith("size:")
                ):
                    metrics["hashed_count"] += 1

        collect_metrics(root_entry)

        metrics["end_time"] = time.monotonic()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["memory_after"] = (
            psutil.Process().memory_info().rss / (1024 * 1024)
        )
        metrics["memory_used"] = metrics["memory_after"] - metrics["memory_before"]
        metrics["ignored_count"] = len(self._ignored_paths_cache) - (
            metrics["file_count"] + metrics["dir_count"]
        )

        self.logger.info(f"Directory parsing complete in {metrics['total_time']:.3f}s")
        self.logger.info(
            f"Files: {metrics['file_count']}, Directories: {metrics['dir_count']}, "
            f"Ignored: {metrics['ignored_count']}"
        )
        self.logger.info(
            f"Total size: {metrics['total_size'] / (1024 * 1024):.2f} MB, "
            f"Hashed files: {metrics['hashed_count']}"
        )
        self.logger.info(
            f"Memory used: {metrics['memory_used']:.2f} MB, "
            f"Cache entries: {len(self._ignored_paths_cache)}"
        )

        if self.use_parallel_processing:
            self.logger.info(f"Parallel processing enabled with max workers: {MAX_WORKERS}")

        return root_entry

    def to_dict(self, entry: DirectoryEntry) -> Dict[str, Any]:
        """Converts a DirectoryEntry object (and its children, recursively) to a dictionary.

        Args:
            entry (DirectoryEntry): The DirectoryEntry to convert.

        Returns:
            Dict[str, Any]: A dictionary representation of the DirectoryEntry.

        Performance: O(N) where N is total entries in the sub-tree rooted at `entry`.
        """
        result: Dict[str, Any] = {
            "name": entry.name,
            "path": entry.path,
            "parent": entry.parent,
            "type": entry.entry_type.name,
            "size": entry.size,
            "extension": entry.extension,
            "file_hash": entry.file_hash,
            "modified_time": entry.modified_time,
        }
        if entry.is_folder():
            result["children"] = [self.to_dict(child) for child in entry.children]
        return result

    def get_tree_string(
        self, entry: Optional[DirectoryEntry] = None, indent_level: int = 0
    ) -> str:
        """Returns the directory tree structure as a string, including file hashes.

        Args:
            entry (Optional[DirectoryEntry]): The DirectoryEntry to start printing from. If None, parsing is triggered.
            indent_level (int): The current indentation level (used for recursion).

        Returns:
            str: A string representation of the directory tree.
        """
        if entry is None:
            raise ValueError(
                "An entry must be provided to get_tree_string. Call parser.parse() first."
            )

        tree_string = ""
        indent_str: str = "    " * indent_level
        prefix: str = "+-- " if indent_level > 0 else ""

        entry_display_name = f"{entry.name}/" if entry.is_folder() else entry.name
        hash_display = (
            f" [hash: {entry.file_hash}]" if entry.is_file() and entry.file_hash else ""
        )
        size_display = f" ({entry.size} bytes)" if entry.is_file() else ""
        tree_string += f"{indent_str}{prefix}{entry_display_name}{size_display}{hash_display}\n"

        sorted_children = sorted(entry.children, key=lambda e: (e.is_file(), e.name.lower()))

        for child in sorted_children:
            tree_string += self.get_tree_string(child, indent_level + 1)

        return tree_string

    def print_tree(
        self, entry: Optional[DirectoryEntry] = None, indent_level: int = 0
    ) -> None:
        """Prints the directory tree structure to standard output, including file hashes.

        Args:
            entry (Optional[DirectoryEntry]): The DirectoryEntry to start printing from. If None, parsing is triggered.
            indent_level (int): The current indentation level (used for recursion).
        """
        print(self.get_tree_string(entry, indent_level))


if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = DirectoryParser(
        directory_path=r"D:\StockAssist",
        gitignore_path=r"D:\StockAssist\.gitignore",
        parallel=True,
        hash_files=True,
        extra_exclude_patterns=["*.log", "temp_*"],
    )

    start_time = time.time()
    root = parser.parse()
    print(f"Total parsing time: {time.time() - start_time:.2f} seconds")

    parser.print_tree(root)
