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
        for child in root_node.children:
            if child.type in ('function_definition', 'class_definition', 'method_definition'):
                chunk_text: str = self._extract_node_text(child, source_bytes)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'type': child.type,
                        'start_line': child.start_point[0] + 1,
                        'end_line': child.end_point[0] + 1
                    })

        return chunks

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

    def __init__(self, embeddings_dir: str = "embeddings", cache_size: int = 10):
        """Initialize SimilaritySearch with embeddings directory.

        Args:
            embeddings_dir (str): Directory containing the embeddings.
            cache_size (int): Number of recent queries to cache.
        """
        self.embeddings_dir: str = embeddings_dir
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.normalized_embeddings: Dict[str, np.ndarray] = {}
        self.cache_size: int = cache_size
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

        for file in os.listdir(self.embeddings_dir):
            if file.endswith('.npz'):
                base_name: str = os.path.splitext(file)[0]
                embed_path: str = os.path.join(self.embeddings_dir, file)
                meta_path: str = os.path.join(self.embeddings_dir, f"{base_name}.json")

                if os.path.exists(meta_path):
                    self.embeddings[base_name] = np.load(embed_path)['embeddings']
                    norms = np.linalg.norm(self.embeddings[base_name], axis=1, keepdims=True)
                    self.normalized_embeddings[base_name] = self.embeddings[base_name] / (norms + 1e-8)

                    with open(meta_path) as f:
                        self.chunks[base_name] = json.load(f)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar code chunks using optimized cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            threshold (float): The minimum similarity score threshold.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing matched chunks and their scores.
        """
        self.total_searches += 1
        start_time = time.time()
        cache_key = f"{query}:{top_k}:{threshold}"

        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]

        self.cache_misses += 1
        query_emb: np.ndarray = np.array(generate_embed(query)[0])
        query_norm: float = np.linalg.norm(query_emb)
        normalized_query: np.ndarray = query_emb / (query_norm + 1e-8)
        all_results: List[Dict[str, Any]] = []

        for file_name, normalized_file_embeddings in self.normalized_embeddings.items():
            similarities: np.ndarray = normalized_file_embeddings @ normalized_query

            if threshold > 0:
                mask = similarities >= threshold
                if not np.any(mask):
                    continue

                indices = np.where(mask)[0]
                top_indices = indices[np.argsort(similarities[indices])[-min(top_k, len(indices)):][::-1]]
            else:
                top_indices: np.ndarray = np.argsort(similarities)[-top_k:][::-1]

            for idx in top_indices:
                score = float(similarities[idx])
                if score >= threshold:
                    all_results.append({
                        'file': file_name,
                        'chunk': self.chunks[file_name][idx],
                        'score': score
                    })

        all_results.sort(key=lambda x: x['score'], reverse=True)
        results = all_results[:top_k]

        if len(self.query_cache) >= self.cache_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = results
        self.search_time += time.time() - start_time

        return results

    def search_multiple(self, queries: List[str], top_k: int = 5,
                       threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search using multiple queries and combine results.

        This is useful when using query optimization to try multiple search terms.

        Args:
            queries (List[str]): List of search queries.
            top_k (int): Number of results to return.
            threshold (float): Minimum similarity score threshold.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing matched chunks and their scores.
        """
        if not queries:
            return []

        if len(queries) == 1:
            return self.search(queries[0], top_k, threshold)

        all_results: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for query in queries:
            results = self.search(query, top_k, threshold)

            for result in results:
                file_name = result['file']
                chunk_idx = self.chunks[file_name].index(result['chunk'])
                key = (file_name, chunk_idx)

                if key not in all_results or result['score'] > all_results[key]['score']:
                    all_results[key] = result

        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        return combined_results[:top_k]

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
            "total_chunks": sum(len(chunks) for chunks in self.chunks.values())
        }
