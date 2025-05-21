#!/usr/bin/env python
"""
VerbalCodeAI - HTTP-based MCP Server

This module provides a Model Context Protocol (MCP) server that connects to the
VerbalCodeAI HTTP API server, allowing Claude and other MCP-compatible AI assistants
to interact with VerbalCodeAI functionality.

Usage:
    python mcp_server_http.py

The server will check if the HTTP API server is running and notify you if it needs to be started.
Configuration is done through environment variables in the .env file:
- MCP_API_URL: The URL of the HTTP API server (default: http://localhost:8000)
- MCP_HTTP_PORT: The port to run the HTTP API server on (default: 8000)
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("VerbalCodeAI.MCP")

mcp = FastMCP(
    "VerbalCodeAI",
    description="VerbalCodeAI MCP Server - Interact with your codebase through Claude",
    dependencies=["requests", "httpx", "python-dotenv"],
)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_HTTP_PORT = 8000

api_url = os.getenv("MCP_API_URL", DEFAULT_API_URL)
http_port = int(os.getenv("MCP_HTTP_PORT", DEFAULT_HTTP_PORT))
http_server_process = None


def save_api_url(url: str) -> None:
    """Save API URL to environment variable and .env file.

    Args:
        url (str): The API URL to save.
    """
    global api_url
    api_url = url

    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()

            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith("MCP_API_URL="):
                    lines[i] = f"MCP_API_URL={url}\n"
                    updated = True
                    break

            if not updated:
                lines.append(f"MCP_API_URL={url}\n")

            with open(env_path, "w") as f:
                f.writelines(lines)

            logger.info(f"Updated MCP_API_URL={url} in .env file")
        else:
            logger.warning(".env file not found, creating it")
            with open(env_path, "w") as f:
                f.write(f"MCP_API_URL={url}\n")
    except Exception as e:
        logger.error(f"Error updating MCP_API_URL in .env file: {e}")


def is_http_server_running() -> bool:
    """Check if the HTTP API server is running.

    Returns:
        bool: True if the server is running, False otherwise.
    """
    try:
        response = requests.get(f"{api_url}/api/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def start_http_server(port: int = None) -> subprocess.Popen:
    """Start the HTTP API server.

    Args:
        port (int, optional): The port to run the server on. If None, uses http_port.

    Returns:
        subprocess.Popen: The server process.
    """
    if port is None:
        port = http_port

    logger.info(f"Starting HTTP API server on port {port}")

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    process = subprocess.Popen(
        [sys.executable, app_path, "--serve", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for _ in range(10):
        if is_http_server_running():
            logger.info("HTTP API server started successfully")
            return process
        time.sleep(1)

    logger.error("Failed to start HTTP API server")
    return process


@mcp.tool()
def set_api_url(url: str) -> str:
    """Set the API URL for the HTTP API server.

    Args:
        url (str): The API URL (e.g., http://localhost:8000).

    Returns:
        str: Success message.
    """
    global api_url
    api_url = url
    save_api_url(url)

    if is_http_server_running():
        return f"API URL set to {url}. HTTP API server is running."
    else:
        return f"API URL set to {url}. HTTP API server is not running. Use start_http_server() to start it."


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check if the HTTP API server is running.

    Returns:
        Dict[str, Any]: Health status.
    """
    if not is_http_server_running():
        return {
            "status": "error",
            "message": f"HTTP API server is not running at {api_url}. Use start_http_server() to start it."
        }

    try:
        response = requests.get(f"{api_url}/api/health")
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to HTTP API server: {str(e)}"
        }


@mcp.tool()
def start_http_server_tool(port: int = None) -> Dict[str, Any]:
    """Start the HTTP API server.

    Args:
        port (int, optional): The port to run the server on. If None, uses the default port.

    Returns:
        Dict[str, Any]: Status message.
    """
    global http_server_process, api_url, http_port

    if is_http_server_running():
        return {
            "status": "ok",
            "message": f"HTTP API server is already running at {api_url}"
        }

    try:
        if port is not None:
            http_port = port

        http_server_process = start_http_server(http_port)

        if is_http_server_running():
            api_url = f"http://localhost:{http_port}"
            save_api_url(api_url)

            return {
                "status": "ok",
                "message": f"HTTP API server started successfully at {api_url}"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to start HTTP API server"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting HTTP API server: {str(e)}"
        }


@mcp.tool()
def initialize_directory(directory_path: str) -> Dict[str, Any]:
    """Initialize a directory for use with VerbalCodeAI.

    Args:
        directory_path (str): Path to the directory to initialize.

    Returns:
        Dict[str, Any]: Result of initialization.
    """
    if not is_http_server_running():
        return {
            "status": "error",
            "message": f"HTTP API server is not running at {api_url}. Use start_http_server() to start it."
        }

    try:
        response = requests.post(
            f"{api_url}/api/initialize", json={"directory_path": directory_path}
        )
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error initializing directory: {str(e)}"
        }


@mcp.tool()
def ask_agent(question: str) -> Dict[str, Any]:
    """Ask the agent a question about the codebase.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Agent's response.
    """
    if not is_http_server_running():
        return {
            "status": "error",
            "message": f"HTTP API server is not running at {api_url}. Use start_http_server() to start it."
        }

    try:
        response = requests.post(f"{api_url}/api/ask", json={"question": question})
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error asking agent: {str(e)}"
        }


@mcp.tool()
def start_indexing(directory_path: str) -> Dict[str, Any]:
    """Start indexing a directory.

    Args:
        directory_path (str): Path to the directory to index.

    Returns:
        Dict[str, Any]: Result of starting indexing.
    """
    if not is_http_server_running():
        return {
            "status": "error",
            "message": f"HTTP API server is not running at {api_url}. Use start_http_server() to start it."
        }

    try:
        response = requests.post(
            f"{api_url}/api/index/start", json={"directory_path": directory_path}
        )
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting indexing: {str(e)}"
        }


@mcp.tool()
def get_indexing_status() -> Dict[str, Any]:
    """Get the status of the indexing process.

    Args:

    Returns:
        Dict[str, Any]: Current indexing status.
    """
    if not is_http_server_running():
        return {
            "status": "error",
            "message": f"HTTP API server is not running at {api_url}. Use start_http_server() to start it."
        }

    try:
        response = requests.get(f"{api_url}/api/index/status")
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting indexing status: {str(e)}"
        }


if __name__ == "__main__":
    logger.info(f"Using API URL: {api_url}")

    if not is_http_server_running():
        logger.warning(f"HTTP API server is not running at {api_url}")
        logger.info(f"Starting HTTP API server on port {http_port}...")
        http_server_process = start_http_server(http_port)

    mcp.run()
