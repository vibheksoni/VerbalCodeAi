"""HTTP API module for VerbalCodeAI.

This module provides a lightweight HTTP API for VerbalCodeAI functionality,
allowing remote access to core features like indexing, agent queries, and status checks.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from mods.code.agent_mode import AgentMode
from mods.code.indexer import FileIndexer
from mods.code.memory import MemoryManager

load_dotenv()

logger = logging.getLogger("VerbalCodeAI.HTTP")

indexer: Optional[FileIndexer] = None
agent_mode: Optional[AgentMode] = None
memory_manager: Optional[MemoryManager] = None
indexing_status: Dict[str, Any] = {
    "in_progress": False,
    "directory": None,
    "start_time": None,
    "complete": False,
    "error": None,
    "progress": 0,
}

background_tasks = set()


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get a boolean value from environment variables.

    Args:
        key (str): The environment variable key.
        default (bool): The default value if the key is not found.

    Returns:
        bool: The boolean value.
    """
    value: str = os.getenv(key, str(default)).upper()
    return value in ("TRUE", "YES", "1", "Y", "T")


async def initialize_directory(request: Request) -> JSONResponse:
    """Initialize a directory for use with VerbalCodeAI.

    Args:
        request (Request): HTTP request with directory_path in JSON body

    Returns:
        JSONResponse: Result of initialization
    """
    global indexer, agent_mode, memory_manager, indexing_status

    try:
        data = await request.json()
        directory_path = data.get("directory_path")

        if not directory_path:
            return JSONResponse({"success": False, "error": "Missing directory_path parameter"}, status_code=400)

        if not os.path.isdir(directory_path):
            return JSONResponse({
                "success": False,
                "error": f"Directory not found: {directory_path}"
            }, status_code=404)

        indexer = FileIndexer(directory_path)
        agent_mode = AgentMode(indexer)
        memory_manager = MemoryManager(root_path=directory_path, indexer=indexer)

        indexing_status = indexer.is_index_complete()
        indexing_status["directory"] = directory_path

        return JSONResponse({
            "success": True,
            "directory": directory_path,
            "index_status": indexing_status
        })
    except Exception as e:
        logger.error(f"Error initializing directory: {e}", exc_info=True)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def ask_agent(request: Request) -> JSONResponse:
    """Ask the agent a question about the codebase.

    Args:
        request (Request): HTTP request with question in JSON body

    Returns:
        JSONResponse: Agent's response
    """
    global agent_mode, indexer

    try:
        data = await request.json()
        question = data.get("question")

        if not question:
            return JSONResponse({"success": False, "error": "Missing question parameter"}, status_code=400)

        if not agent_mode or not indexer:
            return JSONResponse({
                "success": False,
                "error": "No directory initialized. Call initialize_directory first."
            }, status_code=400)

        response = await agent_mode.process_query(question)

        return JSONResponse({
            "success": True,
            "question": question,
            "response": response
        })
    except Exception as e:
        logger.error(f"Error processing agent question: {e}", exc_info=True)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def run_indexing() -> None:
    """Run the indexing process in the background."""
    global indexer, indexing_status

    try:
        if not indexer:
            logger.error("Indexer not initialized")
            indexing_status["error"] = "Indexer not initialized"
            indexing_status["in_progress"] = False
            return

        logger.info(f"Starting indexing for directory: {indexer.root_path}")
        indexer.index_directory()
        indexing_status = indexer.is_index_complete()
        indexing_status["in_progress"] = False

        logger.info(f"Indexing completed for directory: {indexer.root_path}")
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        indexing_status["error"] = str(e)
        indexing_status["in_progress"] = False


async def start_indexing(request: Request) -> JSONResponse:
    """Start indexing a directory.

    Args:
        request (Request): HTTP request with directory_path in JSON body

    Returns:
        JSONResponse: Result of starting indexing
    """
    global indexer, indexing_status

    try:
        data = await request.json()
        directory_path = data.get("directory_path")

        if not directory_path:
            return JSONResponse({"success": False, "error": "Missing directory_path parameter"}, status_code=400)

        if not os.path.isdir(directory_path):
            return JSONResponse({
                "success": False,
                "error": f"Directory not found: {directory_path}"
            }, status_code=404)

        if indexing_status.get("in_progress", False):
            return JSONResponse({
                "success": False,
                "error": "Indexing already in progress",
                "status": indexing_status
            }, status_code=409)

        if not indexer or indexer.root_path != directory_path:
            indexer = FileIndexer(directory_path)

        indexing_status = {
            "in_progress": True,
            "directory": directory_path,
            "start_time": time.time(),
            "complete": False,
            "error": None,
            "progress": 0,
        }

        task = asyncio.create_task(run_indexing())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        return JSONResponse({
            "success": True,
            "status": indexing_status
        })
    except Exception as e:
        logger.error(f"Error starting indexing: {e}", exc_info=True)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def get_indexing_status(request: Request) -> JSONResponse:
    """Get the status of the indexing process.

    Args:
        request (Request): The HTTP request.

    Returns:
        JSONResponse: Current indexing status
    """
    global indexer, indexing_status

    try:
        if indexer:
            current_status = indexer.is_index_complete()
            current_status["in_progress"] = indexing_status.get("in_progress", False)
            current_status["directory"] = indexer.root_path
            indexing_status = current_status

        return JSONResponse({
            "success": True,
            "status": indexing_status
        })
    except Exception as e:
        logger.error(f"Error getting indexing status: {e}", exc_info=True)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def health_check(request: Request) -> JSONResponse:
    """Simple health check endpoint.

    Args:
        request (Request): The HTTP request.

    Returns:
        JSONResponse: Health status
    """
    return JSONResponse({
        "status": "ok",
        "version": "1.0.0",
        "service": "VerbalCodeAI HTTP API"
    })


routes = [
    Route("/api/health", health_check, methods=["GET"]),
    Route("/api/initialize", initialize_directory, methods=["POST"]),
    Route("/api/ask", ask_agent, methods=["POST"]),
    Route("/api/index/start", start_indexing, methods=["POST"]),
    Route("/api/index/status", get_indexing_status, methods=["GET"]),
]


def create_app(allow_all_origins: bool = None) -> Starlette:
    """Create and configure the Starlette application.

    Args:
        allow_all_origins (bool, optional): Whether to allow all origins or only localhost.
                                             If None, reads from environment variable. Defaults to None.

    Returns:
        Starlette: Configured application
    """
    if allow_all_origins is None:
        allow_all_origins = _get_env_bool("HTTP_ALLOW_ALL_ORIGINS", False)

    if allow_all_origins:
        logger.warning("HTTP API server configured to allow connections from any IP address")
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        ]
    else:
        logger.info("HTTP API server configured to allow connections from localhost only")
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["http://localhost", "http://127.0.0.1"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        ]

    app = Starlette(
        debug=True,
        routes=routes,
        middleware=middleware,
    )

    return app
