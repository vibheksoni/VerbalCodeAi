"""Storage module for efficient data storage and retrieval.

This module provides optimized storage solutions for code embeddings and metadata,
with support for compression, versioning, and data integrity checks.
"""

import hashlib
import json
import os
import pickle
import shutil
import time
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

STORAGE_VERSION = 1


class StorageError(Exception):
    """Base exception for storage-related errors."""

    pass


class VersionError(StorageError):
    """Exception raised when there's a version mismatch."""

    pass


class IntegrityError(StorageError):
    """Exception raised when data integrity check fails."""

    pass


class StorageManager:
    """Manages efficient storage and retrieval of code embeddings and metadata.

    Features:
    - Binary storage for reduced size and faster access
    - Compression for embeddings to reduce storage requirements
    - Data integrity checks via checksums
    - Versioning for future compatibility
    - Caching for frequently accessed data
    """

    def __init__(self, storage_dir: str = ".index") -> None:
        """Initialize the storage manager.

        Args:
            storage_dir (str): Directory for storing data.
        """
        self.storage_dir = storage_dir
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        self.embeddings_dir = os.path.join(storage_dir, "embeddings")
        self.binary_dir = os.path.join(storage_dir, "binary")

        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.binary_dir, exist_ok=True)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100

    def _compute_checksum(self, data: bytes) -> str:
        """Compute a checksum for data integrity verification.

        Args:
            data (bytes): The data to checksum.

        Returns:
            str: Hexadecimal checksum.
        """
        return hashlib.sha256(data).hexdigest()

    def _compress_data(self, data: bytes, level: int = 6) -> bytes:
        """Compress binary data.

        Args:
            data (bytes): Data to compress.
            level (int): Compression level (0-9, higher = more compression but slower).

        Returns:
            bytes: Compressed data.
        """
        return zlib.compress(data, level)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress binary data.

        Args:
            data (bytes): Compressed data.

        Returns:
            bytes: Decompressed data.
        """
        return zlib.decompress(data)

    def save_embeddings(
        self,
        file_path: str,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        compress: bool = True,
    ) -> str:
        """Save embeddings and chunks to binary storage.

        Args:
            file_path (str): Path to the original file.
            embeddings (np.ndarray): Embeddings array.
            chunks (List[Dict[str, Any]]): List of code chunks.
            compress (bool): Whether to compress the data.

        Returns:
            str: Path to the saved binary file.
        """
        rel_path = os.path.relpath(file_path, os.path.dirname(self.storage_dir))
        safe_path = rel_path.replace(os.sep, "_").replace(".", "_")
        binary_path = os.path.join(self.binary_dir, f"{safe_path}.bin")

        metadata = {
            "version": STORAGE_VERSION,
            "file_path": file_path,
            "timestamp": time.time(),
            "compressed": compress,
            "embedding_shape": embeddings.shape,
            "embedding_dtype": str(embeddings.dtype),
            "chunks_count": len(chunks),
        }

        embeddings_bytes = embeddings.tobytes()
        chunks_bytes = pickle.dumps(chunks)

        if compress:
            embeddings_bytes = self._compress_data(embeddings_bytes)
            chunks_bytes = self._compress_data(chunks_bytes)

        metadata["embeddings_checksum"] = self._compute_checksum(embeddings_bytes)
        metadata["chunks_checksum"] = self._compute_checksum(chunks_bytes)

        with open(binary_path, "wb") as f:
            metadata_bytes = json.dumps(metadata).encode("utf-8")
            f.write(len(metadata_bytes).to_bytes(4, byteorder="little"))
            f.write(metadata_bytes)

            f.write(len(embeddings_bytes).to_bytes(8, byteorder="little"))
            f.write(embeddings_bytes)

            f.write(len(chunks_bytes).to_bytes(8, byteorder="little"))
            f.write(chunks_bytes)

        return binary_path

    def load_embeddings(self, file_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and chunks from binary storage.

        Args:
            file_path (str): Path to the original file.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Embeddings and chunks.

        Raises:
            FileNotFoundError: If the binary file doesn't exist.
            VersionError: If the storage version is incompatible.
            IntegrityError: If data integrity check fails.
        """
        cache_key = f"embeddings:{file_path}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]["embeddings"], self._cache[cache_key]["chunks"]

        self._cache_misses += 1

        rel_path = os.path.relpath(file_path, os.path.dirname(self.storage_dir))
        safe_path = rel_path.replace(os.sep, "_").replace(".", "_")
        binary_path = os.path.join(self.binary_dir, f"{safe_path}.bin")

        if not os.path.exists(binary_path):
            return self._load_legacy_embeddings(file_path)

        with open(binary_path, "rb") as f:
            metadata_size = int.from_bytes(f.read(4), byteorder="little")
            metadata_bytes = f.read(metadata_size)
            metadata = json.loads(metadata_bytes.decode("utf-8"))

            if metadata["version"] != STORAGE_VERSION:
                raise VersionError(
                    f"Storage version mismatch: expected {STORAGE_VERSION}, got {metadata['version']}"
                )

            embeddings_size = int.from_bytes(f.read(8), byteorder="little")
            embeddings_bytes = f.read(embeddings_size)

            chunks_size = int.from_bytes(f.read(8), byteorder="little")
            chunks_bytes = f.read(chunks_size)

            if self._compute_checksum(embeddings_bytes) != metadata["embeddings_checksum"]:
                raise IntegrityError("Embeddings checksum verification failed")

            if self._compute_checksum(chunks_bytes) != metadata["chunks_checksum"]:
                raise IntegrityError("Chunks checksum verification failed")

            if metadata["compressed"]:
                embeddings_bytes = self._decompress_data(embeddings_bytes)
                chunks_bytes = self._decompress_data(chunks_bytes)

            embeddings = np.frombuffer(
                embeddings_bytes, dtype=np.dtype(metadata["embedding_dtype"])
            ).reshape(metadata["embedding_shape"])

            chunks = pickle.loads(chunks_bytes)

            if len(self._cache) >= self._max_cache_size:
                self._cache.pop(next(iter(self._cache)))

            self._cache[cache_key] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "timestamp": time.time(),
            }

            return embeddings, chunks

    def _load_legacy_embeddings(self, file_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings from legacy JSON/NPZ format.

        Args:
            file_path (str): Path to the original file.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Embeddings and chunks.

        Raises:
            FileNotFoundError: If the legacy files don't exist.
        """
        base_name = os.path.basename(file_path)
        embed_file = os.path.join(self.embeddings_dir, f"{base_name}.npz")
        meta_file = os.path.join(self.embeddings_dir, f"{base_name}.json")

        if not (os.path.exists(embed_file) and os.path.exists(meta_file)):
            raise FileNotFoundError(f"Embeddings or metadata not found for {file_path}")

        embeddings = np.load(embed_file)["embeddings"]
        with open(meta_file, "r") as f:
            chunks = json.load(f)

        return embeddings, chunks
