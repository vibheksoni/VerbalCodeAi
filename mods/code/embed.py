import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tree_sitter_language_pack import get_parser

from ..llms import generate_embed

logger = logging.getLogger("VerbalCodeAI.CodeEmbed")

class CodeChunker:
    """
    A class to chunk code files using tree-sitter for intelligent code splitting.
    """

    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'c_sharp',
        '.java': 'java',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.lua': 'lua',
        '.r': 'r',
        '.dart': 'dart',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',
        '.hrl': 'erlang',
        '.clj': 'clojure',
        '.cljs': 'clojure',
        '.groovy': 'groovy',
        '.pl': 'perl',
        '.pm': 'perl',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.psm1': 'powershell',
        '.elm': 'elm',
        '.f': 'fortran',
        '.f90': 'fortran',
        '.f95': 'fortran',
        '.jl': 'julia',
        '.m': 'objective-c',
        '.mm': 'objective-c',
        '.nim': 'nim',
        '.v': 'verilog',
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        '.zig': 'zig',
        '.d': 'd',
        '.fs': 'fsharp',
        '.fsx': 'fsharp',
        '.fsi': 'fsharp',
    }

    TEXT_FILE_EXTENSIONS = {
        '.md', '.markdown', '.txt', '.rst', '.adoc', '.asciidoc', '.wiki',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.config', '.properties',
        '.env', '.gitignore', '.dockerignore', '.editorconfig', '.gitattributes',
        '.html', '.htm', '.css', '.scss', '.less', '.svg', '.jsx', '.tsx',
        '.sh', '.bash', '.zsh', '.bat', '.cmd', '.ps1', '.psm1',
        '.xml', '.csv', '.tsv', '.sql', '.graphql', '.gql',
        '.makefile', '.dockerfile', '.cmake', '.gradle', '.bazel',
        '.log', '.diff', '.patch', '.template', '.j2', '.tpl',
        '.proto', '.plist', '.manifest', '.lock', '.ipynb'
    }

    def __init__(self):
        """Initialize CodeChunker with parsers."""
        self.parsers: Dict[str, Any] = {}
        self._init_parsers()

    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        initialized_parsers: set[str] = set()
        failed_parsers: set[str] = set()

        for lang in self.SUPPORTED_LANGUAGES.values():
            if lang in initialized_parsers or lang in failed_parsers:
                continue

            try:
                self.parsers[lang] = get_parser(lang)
                initialized_parsers.add(lang)
            except Exception as e:
                if lang not in failed_parsers:
                    logger.warning(f"Failed to initialize parser for {lang}: {e}")
                    failed_parsers.add(lang)

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            Optional[str]: The detected language or None if not supported.
        """
        ext: str = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_LANGUAGES.get(ext)

    def _extract_node_text(self, node: Any, source_bytes: bytes) -> str:
        """Extract text from a tree-sitter node.

        Args:
            node (Any): The tree-sitter node.
            source_bytes (bytes): The source code in bytes.

        Returns:
            str: The extracted text.
        """
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

    def is_text_file(self, file_path: str) -> bool:
        """Check if the file is a text file we can process even if not a supported language.

        Args:
            file_path (str): Path to the file.

        Returns:
            bool: True if it's a recognized text file format.
        """
        ext: str = os.path.splitext(file_path)[1].lower()
        return ext in self.TEXT_FILE_EXTENSIONS

    def chunk_file(self, file_path: str, min_chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Chunk a code file into semantically meaningful parts using tree-sitter.

        Args:
            file_path (str): Path to the code file.
            min_chunk_size (int): Minimum size of a chunk in characters.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing chunk info (text, type, start_line, end_line).

        Raises:
            ValueError: If the language is not supported and not a recognized text file.
        """
        language: Optional[str] = self._detect_language(file_path)

        if not language or language not in self.parsers:
            if self.is_text_file(file_path):
                return self._chunk_generic_text_file(file_path, min_chunk_size)
            else:
                raise ValueError(f"Unsupported language for file: {file_path}")

        with open(file_path, 'rb') as f:
            source_bytes: bytes = f.read()

        parser = self.parsers[language]
        tree = parser.parse(source_bytes)
        root_node = tree.root_node

        chunks: List[Dict[str, Any]] = []

        chunks = self._extract_semantic_chunks(root_node, source_bytes, min_chunk_size)

        if not chunks:
            chunks = self._chunk_code_fallback(file_path, source_bytes, root_node, min_chunk_size)

        return chunks

    def _extract_semantic_chunks(self, root_node: Any, source_bytes: bytes, min_chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Extract semantically meaningful chunks from code using adaptive strategies.

        This method works for any programming language by:
        1. Identifying common semantic patterns (functions, classes, etc.)
        2. Using heuristics to detect meaningful code structures
        3. Adapting to different language syntaxes automatically

        Args:
            root_node (Any): The tree-sitter root node.
            source_bytes (bytes): The source code in bytes.
            min_chunk_size (int): Minimum size of a chunk in characters.

        Returns:
            List[Dict[str, Any]]: List of semantically meaningful chunks.
        """
        chunks: List[Dict[str, Any]] = []

        semantic_chunks = self._find_semantic_nodes(root_node, source_bytes)

        code_min_chunk_size = max(10, min_chunk_size // 3)

        for node_info in semantic_chunks:
            node, confidence = node_info['node'], node_info['confidence']
            chunk_text = self._extract_node_text(node, source_bytes)

            if len(chunk_text) >= code_min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'type': node.type,
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'confidence': confidence
                })

        chunks.sort(key=lambda x: (-x['confidence'], x['start_line']))

        return chunks

    def _find_semantic_nodes(self, root_node: Any, source_bytes: bytes) -> List[Dict[str, Any]]:
        """Find nodes that are likely to be semantically meaningful (functions, classes, etc.).

        Uses multiple heuristics to identify important code structures across languages.

        Args:
            root_node (Any): The tree-sitter root node.
            source_bytes (bytes): The source code in bytes.

        Returns:
            List[Dict[str, Any]]: List of node info with confidence scores.
        """
        semantic_nodes = []

        known_semantic_types = {
            # Functions
            'function_definition': 1.0,     # Python
            'function_item': 1.0,           # Rust
            'function_declaration': 1.0,    # C/C++
            'function': 1.0,                # JavaScript
            'arrow_function': 1.0,          # JavaScript
            'method_definition': 1.0,       # Python, JavaScript
            'function_expression': 1.0,     # JavaScript

            # Classes and types
            'class_definition': 1.0,        # Python
            'class_declaration': 1.0,       # C++, Java, JavaScript
            'struct_item': 1.0,             # Rust
            'enum_item': 1.0,               # Rust
            'trait_item': 1.0,              # Rust
            'impl_item': 1.0,               # Rust
            'interface_declaration': 1.0,   # TypeScript, Java
            'type_alias': 1.0,              # TypeScript, Rust

            # Other important constructs
            'module_declaration': 0.9,      # TypeScript
            'namespace_declaration': 0.9,   # C++
            'package_declaration': 0.8,     # Java, Go
        }

        def traverse_node(node):
            if node.type in known_semantic_types:
                semantic_nodes.append({
                    'node': node,
                    'confidence': known_semantic_types[node.type],
                    'reason': f'known_type:{node.type}'
                })
            else:
                confidence = self._calculate_semantic_confidence(node, source_bytes)
                if confidence > 0.5:
                    semantic_nodes.append({
                        'node': node,
                        'confidence': confidence,
                        'reason': 'heuristic_analysis'
                    })

            if node.child_count > 0 and node.child_count < 50:
                for child in node.children:
                    traverse_node(child)

        traverse_node(root_node)

        semantic_nodes = self._remove_overlapping_nodes(semantic_nodes)

        return semantic_nodes

    def _calculate_semantic_confidence(self, node: Any, source_bytes: bytes) -> float:
        """Calculate confidence that a node represents a meaningful semantic unit.

        Uses various heuristics to determine if an unknown node type is likely
        to be a function, class, or other important code structure.

        Args:
            node (Any): The tree-sitter node to analyze.
            source_bytes (bytes): The source code in bytes.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        confidence = 0.0
        node_text = self._extract_node_text(node, source_bytes).lower()

        semantic_keywords = ['function', 'class', 'method', 'def', 'fn', 'struct', 'enum', 'trait', 'impl', 'interface', 'type']
        for keyword in semantic_keywords:
            if keyword in node.type.lower():
                confidence += 0.4
                break

        definition_patterns = [
            'def ', 'function ', 'class ', 'fn ', 'struct ', 'enum ', 'trait ', 'impl ',
            'interface ', 'type ', 'const ', 'let ', 'var ', 'public ', 'private ',
            'static ', 'async ', 'export '
        ]
        for pattern in definition_patterns:
            if node_text.startswith(pattern):
                confidence += 0.3
                break

        node_size = len(node_text)
        if 20 <= node_size <= 2000:
            confidence += 0.2
        elif 10 <= node_size <= 5000:
            confidence += 0.1

        brace_balance = node_text.count('{') - node_text.count('}')
        paren_balance = node_text.count('(') - node_text.count(')')
        if abs(brace_balance) <= 1 and abs(paren_balance) <= 1:
            confidence += 0.2

        code_patterns = ['(', ')', '{', '}', ';', '=', 'return', 'if', 'for', 'while']
        pattern_count = sum(1 for pattern in code_patterns if pattern in node_text)
        if pattern_count >= 3:
            confidence += 0.1

        node_depth = self._calculate_node_depth(node)
        if 1 <= node_depth <= 3:
            confidence += 0.1

        return min(confidence, 1.0)  # Cap at 1.0

    def _calculate_node_depth(self, node: Any) -> int:
        """Calculate the depth of a node in the syntax tree."""
        depth = 0
        current = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def _remove_overlapping_nodes(self, semantic_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping nodes, preferring higher confidence and more specific scope."""
        if not semantic_nodes:
            return []

        semantic_nodes.sort(key=lambda x: (-x['confidence'], x['node'].end_byte - x['node'].start_byte))

        filtered_nodes = []

        for node_info in semantic_nodes:
            node = node_info['node']
            overlaps = False

            for selected_info in filtered_nodes:
                selected_node = selected_info['node']

                if (node.start_byte < selected_node.end_byte and
                    node.end_byte > selected_node.start_byte):
                    overlaps = True
                    break

            if not overlaps:
                filtered_nodes.append(node_info)

        return filtered_nodes

    def _chunk_code_fallback(self, file_path: str, source_bytes: bytes, root_node: Any, min_chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Fallback chunking strategy for code files without functions/classes.

        This method tries different strategies to create meaningful chunks from code files
        that don't have traditional function or class definitions.

        Args:
            file_path (str): Path to the code file.
            source_bytes (bytes): The source code in bytes.
            root_node (Any): The tree-sitter root node.
            min_chunk_size (int): Minimum size of a chunk in characters.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing chunk info.
        """
        chunks: List[Dict[str, Any]] = []

        meaningful_nodes = []
        for child in root_node.children:
            if child.type in (
                'import_statement', 'import_from_statement',         # Python imports
                'assignment', 'expression_statement',                # Assignments and expressions
                'if_statement', 'for_statement', 'while_statement',  # Control flow
                'try_statement', 'with_statement',                   # Exception handling, context managers
                'decorated_definition',                              # Decorated functions/classes
                'global_statement', 'nonlocal_statement',            # Scope declarations
                'assert_statement', 'pass_statement',                # Other statements
                'variable_declaration', 'const_declaration',         # JavaScript/TypeScript
                'struct_item', 'enum_item', 'impl_item',             # Rust
                'package_declaration', 'interface_declaration',      # Java/Go
            ):
                meaningful_nodes.append(child)

        if meaningful_nodes:
            current_chunk_nodes = []
            current_chunk_size = 0

            for node in meaningful_nodes:
                node_text = self._extract_node_text(node, source_bytes)
                node_size = len(node_text)

                if current_chunk_nodes and (current_chunk_size + node_size > 500):
                    chunk_text = self._combine_nodes_text(current_chunk_nodes, source_bytes)
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append({
                            'text': chunk_text,
                            'type': 'code_block',
                            'start_line': current_chunk_nodes[0].start_point[0] + 1,
                            'end_line': current_chunk_nodes[-1].end_point[0] + 1
                        })
                    current_chunk_nodes = []
                    current_chunk_size = 0

                current_chunk_nodes.append(node)
                current_chunk_size += node_size

            if current_chunk_nodes:
                chunk_text = self._combine_nodes_text(current_chunk_nodes, source_bytes)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'type': 'code_block',
                        'start_line': current_chunk_nodes[0].start_point[0] + 1,
                        'end_line': current_chunk_nodes[-1].end_point[0] + 1
                    })

        if not chunks:
            full_text = source_bytes.decode('utf-8', errors='replace')
            if len(full_text) >= min_chunk_size:
                chunks.append({
                    'text': full_text,
                    'type': 'full_file',
                    'start_line': 1,
                    'end_line': full_text.count('\n') + 1
                })

        return chunks

    def _combine_nodes_text(self, nodes: List[Any], source_bytes: bytes) -> str:
        """Combine text from multiple tree-sitter nodes.

        Args:
            nodes (List[Any]): List of tree-sitter nodes.
            source_bytes (bytes): The source code in bytes.

        Returns:
            str: Combined text from all nodes.
        """
        if not nodes:
            return ""

        start_byte = nodes[0].start_byte
        end_byte = nodes[-1].end_byte
        return source_bytes[start_byte:end_byte].decode('utf-8', errors='replace')

    def _chunk_generic_text_file(self, file_path: str, min_chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Process a generic text file by splitting it into manageable chunks.

        Args:
            file_path (str): Path to the text file.
            min_chunk_size (int): Minimum size of a chunk in characters.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing chunk info.
        """
        chunks: List[Dict[str, Any]] = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines: List[str] = f.readlines()

            if not lines:
                return []

            if len(lines) <= 100:
                chunks.append({
                    'text': ''.join(lines),
                    'type': 'generic_text',
                    'start_line': 1,
                    'end_line': len(lines)
                })
                return chunks

            chunk_size: int = 100
            total_lines: int = len(lines)

            for i in range(0, total_lines, chunk_size):
                end_idx: int = min(i + chunk_size, total_lines)
                chunk_text: str = ''.join(lines[i:end_idx])

                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'type': 'generic_text',
                        'start_line': i + 1,
                        'end_line': end_idx
                    })

        except Exception as e:
            logger.error(f"Error chunking generic text file {file_path}: {e}")
            chunks.append({
                'text': f"File: {os.path.basename(file_path)}",
                'type': 'filename_only',
                'start_line': 1,
                'end_line': 1
            })

        return chunks

class CodeEmbedding:
    """
    A class to generate and manage code embeddings with advanced features.

    Features:
    - Efficient embedding generation for code chunks
    - Support for dimensionality reduction
    - Versioned embedding format
    - Optimized storage with compression
    """

    EMBEDDING_VERSION = 1
    DEFAULT_REDUCED_DIMS = 256

    def __init__(self, use_dimensionality_reduction: bool = False, reduced_dims: int = DEFAULT_REDUCED_DIMS):
        """Initialize CodeEmbedding with a CodeChunker.

        Args:
            use_dimensionality_reduction (bool): Whether to use dimensionality reduction.
            reduced_dims (int): Target dimensionality for reduction.
        """
        self.chunker: CodeChunker = CodeChunker()
        self.use_dimensionality_reduction: bool = use_dimensionality_reduction
        self.reduced_dims: int = reduced_dims
        self.pca_model: Any = None
        self.embedding_time: float = 0.0
        self.total_embeddings: int = 0
        self.total_chunks: int = 0

    def _apply_dimensionality_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to embeddings.

        Args:
            embeddings (np.ndarray): Original embeddings.

        Returns:
            np.ndarray: Reduced embeddings.
        """
        if not self.use_dimensionality_reduction:
            return embeddings

        if embeddings.shape[0] < 2:
            return embeddings

        try:
            from sklearn.decomposition import PCA

            if self.pca_model is None or self.pca_model.n_components != min(self.reduced_dims, embeddings.shape[1]):
                n_components = min(self.reduced_dims, embeddings.shape[1])
                self.pca_model = PCA(n_components=n_components)

            reduced_embeddings = self.pca_model.fit_transform(embeddings)
            return reduced_embeddings

        except ImportError:
            logger.warning("scikit-learn not installed. Dimensionality reduction skipped.")
            return embeddings
        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {e}")
            return embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for code chunks with optional dimensionality reduction.

        Args:
            chunks (List[Dict[str, Any]]): List of code chunks.

        Returns:
            np.ndarray: Embeddings for the code chunks.
        """
        start_time = time.time()
        self.total_chunks += len(chunks)
        texts: List[str] = [chunk['text'] for chunk in chunks]
        embeddings: List[List[float]] = generate_embed(texts)
        embeddings_array: np.ndarray = np.array(embeddings)

        if self.use_dimensionality_reduction:
            embeddings_array = self._apply_dimensionality_reduction(embeddings_array)

        self.embedding_time += time.time() - start_time
        self.total_embeddings += len(chunks)

        return embeddings_array

    def process_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process a file to generate chunks and their embeddings.

        Args:
            file_path (str): Path to the file.

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray]: Chunks and their embeddings.
        """
        chunks: List[Dict[str, Any]] = self.chunker.chunk_file(file_path)
        embeddings: np.ndarray = self.embed_chunks(chunks)
        return chunks, embeddings

    def save_embeddings(self, file_path: str, chunks: List[Dict[str, Any]],
                        embeddings: np.ndarray, output_dir: str = "embeddings"):
        """Save embeddings and chunks to disk with versioning and metadata.

        Args:
            file_path (str): Path to the file.
            chunks (List[Dict[str, Any]]): List of code chunks.
            embeddings (np.ndarray): Embeddings for the code chunks.
            output_dir (str): Directory to save the embeddings and chunks.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name: str = os.path.basename(file_path)
        embed_file: str = os.path.join(output_dir, f"{base_name}.npz")
        meta_file: str = os.path.join(output_dir, f"{base_name}.json")

        metadata = {
            "version": self.EMBEDDING_VERSION,
            "timestamp": time.time(),
            "file_path": file_path,
            "embedding_shape": embeddings.shape,
            "embedding_dtype": str(embeddings.dtype),
            "dimensionality_reduced": self.use_dimensionality_reduction,
            "original_dimensions": None if not self.use_dimensionality_reduction else self.pca_model.n_features_in_,
            "reduced_dimensions": None if not self.use_dimensionality_reduction else self.pca_model.n_components_,
            "chunks_count": len(chunks)
        }

        np.savez_compressed(embed_file, embeddings=embeddings, metadata=metadata)

        with open(meta_file, 'w') as f:
            json.dump({
                "metadata": metadata,
                "chunks": chunks
            }, f, indent=2)

    def load_embeddings(self, file_path: str, output_dir: str = "embeddings") -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Load embeddings and chunks for a given file with version checking.

        Args:
            file_path (str): Path to the file.
            output_dir (str): Directory to load the embeddings and chunks from.

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray]: Chunks and embeddings.

        Raises:
            FileNotFoundError: If the embedding files don't exist.
            ValueError: If the embedding version is incompatible.
        """
        base_name: str = os.path.basename(file_path)
        embed_file: str = os.path.join(output_dir, f"{base_name}.npz")
        meta_file: str = os.path.join(output_dir, f"{base_name}.json")

        if not (os.path.exists(embed_file) and os.path.exists(meta_file)):
            raise FileNotFoundError(f"Embeddings or metadata not found for {file_path}")

        npz_data = np.load(embed_file)
        embeddings: np.ndarray = npz_data['embeddings']

        if 'metadata' in npz_data:
            metadata = npz_data['metadata'].item()
            if metadata.get('version', 0) != self.EMBEDDING_VERSION:
                logger.warning(f"Embedding version mismatch. Expected {self.EMBEDDING_VERSION}, got {metadata.get('version', 0)}")

        with open(meta_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'chunks' in data:
            chunks = data['chunks']
            metadata = data.get('metadata', {})
        else:
            chunks = data
            metadata = {}

        if metadata.get('version', 0) != self.EMBEDDING_VERSION:
            logger.warning(f"Metadata version mismatch. Expected {self.EMBEDDING_VERSION}, got {metadata.get('version', 0)}")

        return chunks, embeddings

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for embedding generation.

        Returns:
            Dict[str, Any]: Dictionary with performance metrics.
        """
        avg_embedding_time = self.embedding_time / max(1, self.total_embeddings)

        return {
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "embedding_time": self.embedding_time,
            "avg_embedding_time": avg_embedding_time,
            "dimensionality_reduction": self.use_dimensionality_reduction,
            "reduced_dims": self.reduced_dims if self.use_dimensionality_reduction else None
        }

class SimilaritySearch:
    """
    A class for efficient similarity search in code embeddings.
    Uses optimized vector search algorithms with caching and performance enhancements.
    """

    def __init__(self, embeddings_dir: str = "embeddings", cache_size: int = None):
        """Initialize SimilaritySearch with embeddings directory.

        Args:
            embeddings_dir (str): Directory containing the embeddings.
            cache_size (int): Number of recent queries to cache. If None, uses EMBEDDING_CACHE_SIZE from .env.
        """
        self.embeddings_dir: str = embeddings_dir
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.normalized_embeddings: Dict[str, np.ndarray] = {}

        from os import environ

        if cache_size is None:
            try:
                cache_size = int(environ.get("EMBEDDING_CACHE_SIZE", "100"))
            except (ValueError, TypeError):
                cache_size = 100
                logger.warning(
                    f"Invalid EMBEDDING_CACHE_SIZE in .env, using default: {cache_size}"
                )

        self.cache_size: int = cache_size
        logger.info(f"Initializing SimilaritySearch with cache size: {self.cache_size}")

        try:
            self.default_threshold = float(
                environ.get("EMBEDDING_SIMILARITY_THRESHOLD", "0.15")
            )
        except (ValueError, TypeError):
            self.default_threshold = 0.15
            logger.warning(
                f"Invalid EMBEDDING_SIMILARITY_THRESHOLD in .env, using default: {self.default_threshold}"
            )

        logger.info(f"Using default similarity threshold: {self.default_threshold}")

        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.load_embeddings()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_searches: int = 0
        self.search_time: float = 0.0

    def load_embeddings(self):
        """Load all embeddings from the embeddings directory and pre-normalize them."""
        if not os.path.exists(self.embeddings_dir):
            return

        npz_files_loaded = 0
        for file in os.listdir(self.embeddings_dir):
            if file.endswith(".npz"):
                base_name: str = os.path.splitext(file)[0]
                embed_path: str = os.path.join(self.embeddings_dir, file)
                meta_path: str = os.path.join(self.embeddings_dir, f"{base_name}.json")

                if os.path.exists(meta_path):
                    try:
                        self.embeddings[base_name] = np.load(embed_path)["embeddings"]
                        norms = np.linalg.norm(
                            self.embeddings[base_name], axis=1, keepdims=True
                        )
                        self.normalized_embeddings[base_name] = self.embeddings[
                            base_name
                        ] / (norms + 1e-8)

                        with open(meta_path) as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "chunks" in data:
                                self.chunks[base_name] = data["chunks"]
                            else:
                                self.chunks[base_name] = data
                        npz_files_loaded += 1
                    except Exception as e:
                        logger.warning(f"Error loading NPZ file {embed_path}: {e}")

        if npz_files_loaded == 0:
            logger.info("No NPZ files found. Attempting to load embeddings directly from JSON files.")
            json_files_loaded = 0

            for file in os.listdir(self.embeddings_dir):
                if file.endswith(".json"):
                    base_name: str = os.path.splitext(file)[0]
                    json_path: str = os.path.join(self.embeddings_dir, file)

                    try:
                        with open(json_path) as f:
                            data = json.load(f)

                            if isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list):
                                self.embeddings[base_name] = np.array(data["embeddings"])

                                norms = np.linalg.norm(
                                    self.embeddings[base_name], axis=1, keepdims=True
                                )
                                self.normalized_embeddings[base_name] = self.embeddings[
                                    base_name
                                ] / (norms + 1e-8)

                                if "chunks" in data:
                                    self.chunks[base_name] = data["chunks"]
                                else:
                                    self.chunks[base_name] = data

                                json_files_loaded += 1
                            elif isinstance(data, dict) and "chunks" in data and "path" in data:
                                if "embeddings" in data and isinstance(data["embeddings"], list):
                                    try:
                                        embeddings_data = data["embeddings"]

                                        if len(embeddings_data) > 0:
                                            if isinstance(embeddings_data[0], (list, tuple)) and len(embeddings_data[0]) > 0:
                                                self.embeddings[base_name] = np.array(embeddings_data)
                                            elif isinstance(embeddings_data[0], (int, float)):
                                                self.embeddings[base_name] = np.array([embeddings_data])
                                            else:
                                                logger.warning(f"Unknown embedding format in {json_path}: {type(embeddings_data[0])}")
                                                continue
                                        else:
                                            logger.warning(f"Empty embeddings list in {json_path}")
                                            continue

                                        norms = np.linalg.norm(
                                            self.embeddings[base_name], axis=1, keepdims=True
                                        )
                                        self.normalized_embeddings[base_name] = self.embeddings[
                                            base_name
                                        ] / (norms + 1e-8)

                                        self.chunks[base_name] = data["chunks"]

                                        json_files_loaded += 1
                                    except Exception as e:
                                        logger.warning(f"Error processing embeddings in {json_path}: {e}")
                                        continue
                    except Exception as e:
                        logger.warning(f"Error loading embeddings from JSON file {json_path}: {e}")

            if json_files_loaded > 0:
                logger.info(f"Successfully loaded embeddings from {json_files_loaded} JSON files.")
            else:
                logger.warning("No embeddings could be loaded from either NPZ or JSON files.")

    def search(
        self, query: str, top_k: int = 5, threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar code chunks using optimized cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            threshold (float): The minimum similarity score threshold. If None, uses the default threshold.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing matched chunks and their scores.
        """
        if threshold is None:
            threshold = self.default_threshold
        self.total_searches += 1
        start_time = time.time()
        cache_key = f"{query}:{top_k}:{threshold}"

        if cache_key in self.query_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for query: {query}")
            return self.query_cache[cache_key]

        self.cache_misses += 1
        logger.debug(f"Cache miss for query: {query}, generating embedding")

        try:
            query_emb_result = generate_embed(query)
            if not query_emb_result or len(query_emb_result) == 0:
                logger.warning(f"Failed to generate embedding for query: {query}")
                return []

            query_emb: np.ndarray = np.array(query_emb_result[0])

            if query_emb.size == 0:
                logger.warning(f"Query embedding is empty for query: {query}")
                return []

            if len(query_emb.shape) == 0:
                logger.warning(f"Query embedding has wrong shape: {query_emb.shape}")
                try:
                    query_emb = np.array([float(query_emb)])
                    logger.debug(f"Reshaped scalar embedding to 1D array: {query_emb.shape}")
                except:
                    return []

            query_norm: float = np.linalg.norm(query_emb)

            if query_norm < 1e-10:
                logger.warning(f"Query embedding has near-zero norm: {query_norm}")
                return []

            normalized_query: np.ndarray = query_emb / query_norm
        except Exception as e:
            logger.error(f"Error generating embedding for query '{query}': {e}")
            return []

        all_results: List[Dict[str, Any]] = []

        batch_size = 10
        file_names = list(self.normalized_embeddings.keys())

        for i in range(0, len(file_names), batch_size):
            batch_files = file_names[i : i + batch_size]

            for file_name in batch_files:
                normalized_file_embeddings = self.normalized_embeddings[file_name]

                if normalized_file_embeddings.size == 0:
                    logger.warning(f"File embeddings are empty for file: {file_name}")
                    continue

                if len(normalized_file_embeddings.shape) < 2:
                    normalized_file_embeddings = normalized_file_embeddings.reshape(1, -1)

                if normalized_file_embeddings.shape[1] != normalized_query.shape[0]:
                    logger.warning(f"Incompatible dimensions for matrix multiplication: file embeddings shape {normalized_file_embeddings.shape}, query shape {normalized_query.shape}")
                    continue

                try:
                    similarities: np.ndarray = normalized_file_embeddings @ normalized_query
                except Exception as e:
                    logger.error(f"Error computing similarities for file {file_name}: {e}")
                    continue

                if threshold > 0:
                    mask = similarities >= threshold
                    if not np.any(mask):
                        continue

                    indices = np.where(mask)[0]
                    if len(indices) == 0:
                        continue

                    top_indices = indices[
                        np.argsort(similarities[indices])[
                            -min(top_k, len(indices)) :
                        ][::-1]
                    ]
                else:
                    top_indices: np.ndarray = np.argsort(similarities)[
                        -min(top_k, len(similarities)) :
                    ][::-1]

                for idx in top_indices:
                    score = float(similarities[idx])
                    if score >= threshold:
                        try:
                            chunk_data = self.chunks[file_name]
                            if isinstance(chunk_data, list) and idx < len(chunk_data):
                                chunk = chunk_data[idx]
                            else:
                                chunk = {
                                    "text": f"Chunk from {file_name}",
                                    "start_line": 0,
                                    "end_line": 0,
                                    "type": "unknown",
                                }

                            all_results.append(
                                {"file": file_name, "chunk": chunk, "score": score}
                            )
                        except (IndexError, KeyError) as e:
                            logger.warning(
                                f"Error accessing chunk {idx} for file {file_name}: {e}"
                            )

        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]

        search_duration = time.time() - start_time
        self.search_time += search_duration
        logger.debug(
            f"Search completed in {search_duration:.4f}s with {len(top_results)} results"
        )

        if len(self.query_cache) >= self.cache_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = top_results

        return top_results

    def search_multiple(
        self, queries: List[str], top_k: int = 5, threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search using multiple queries and combine results.

        This is useful when using query optimization to try multiple search terms.

        Args:
            queries (List[str]): List of search queries.
            top_k (int): Number of results to return.
            threshold (float): Minimum similarity score threshold. If None, uses the default threshold.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing matched chunks and their scores.
        """
        if threshold is None:
            threshold = self.default_threshold
        if not queries:
            return []

        if len(queries) == 1:
            return self.search(queries[0], top_k, threshold)

        logger.debug(f"Performing multi-query search with {len(queries)} variations")
        start_time = time.time()

        per_query_top_k = min(top_k * 2, 20)

        all_results: Dict[Tuple[str, int], Dict[str, Any]] = {}

        query_contributions = {}

        for i, query in enumerate(queries):
            try:
                results = self.search(query, per_query_top_k, threshold)

                for result in results:
                    file_name = result["file"]

                    try:
                        chunk = result["chunk"]
                        chunk_idx = self.chunks[file_name].index(chunk)
                    except (ValueError, KeyError, IndexError):
                        start_line = chunk.get("start_line", 0)
                        end_line = chunk.get("end_line", 0)
                        chunk_idx = f"{start_line}-{end_line}"

                    key = (file_name, chunk_idx)

                    if key not in query_contributions:
                        query_contributions[key] = []
                    query_contributions[key].append(i)

                    if (
                        key not in all_results
                        or result["score"] > all_results[key]["score"]
                    ):
                        all_results[key] = result

            except Exception as e:
                logger.error(f"Error processing query variation '{query}': {e}")
                continue

        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        final_results = combined_results[:top_k]

        search_duration = time.time() - start_time
        logger.debug(
            f"Multi-query search completed in {search_duration:.4f}s with {len(final_results)} results"
        )

        return final_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the similarity search.

        Returns:
            Dict[str, Any]: Dictionary with performance metrics.
        """
        avg_search_time = self.search_time / max(1, self.total_searches)
        cache_hit_rate = self.cache_hits / max(1, self.total_searches) * 100

        return {
            "total_searches": self.total_searches,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "avg_search_time": avg_search_time,
            "total_search_time": self.search_time,
            "num_files": len(self.embeddings),
            "total_chunks": sum(len(chunks) for chunks in self.chunks.values()),
        }
