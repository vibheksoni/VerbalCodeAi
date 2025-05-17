"""LLM integration module for VerbalCodeAI.

This module provides a unified interface to different LLM providers,
with support for text generation, embeddings, and streaming responses.

Supported providers:
- Ollama (local models)
- Google AI (cloud-based models)

Features:
- Automatic model pulling for Ollama
- Streaming response support
- Error handling and validation
- Configurable system prompts
- Performance metrics
"""

import json
import logging
import os
import time
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import google.generativeai as genai
import ollama
from dotenv import load_dotenv
from ollama import AsyncClient as OllamaAsyncClient

logger = logging.getLogger("VerbalCodeAI.LLMs")

load_dotenv()

AI_CHAT_PROVIDER: str = os.getenv('AI_CHAT_PROVIDER', 'ollama')
AI_EMBEDDING_PROVIDER: str = os.getenv('AI_EMBEDDING_PROVIDER', 'ollama')
AI_DESCRIPTION_PROVIDER: str = os.getenv('AI_DESCRIPTION_PROVIDER', 'ollama')
AI_CHAT_API_KEY: str = os.getenv('AI_CHAT_API_KEY')
AI_EMBEDDING_API_KEY: str = os.getenv('AI_EMBEDDING_API_KEY')
AI_DESCRIPTION_API_KEY: str = os.getenv('AI_DESCRIPTION_API_KEY')

CHAT_MODEL: str = os.getenv('CHAT_MODEL')
EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL')
DESCRIPTION_MODEL: str = os.getenv('DESCRIPTION_MODEL')

PERFORMANCE_METRICS: Dict[str, Union[int, float, Dict[str, Dict[str, Union[int, float]]]]] = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0.0,
    "errors": 0,
    "provider_stats": {
        "ollama": {"requests": 0, "time": 0.0},
        "google": {"requests": 0, "time": 0.0}
    }
}

if (AI_CHAT_PROVIDER == 'google' or
        AI_EMBEDDING_PROVIDER == 'google' or
        AI_DESCRIPTION_PROVIDER == 'google'):

    api_key: Optional[str] = None
    if AI_CHAT_PROVIDER == 'google' and AI_CHAT_API_KEY and AI_CHAT_API_KEY.lower() != 'none':
        api_key = AI_CHAT_API_KEY
    elif AI_EMBEDDING_PROVIDER == 'google' and AI_EMBEDDING_API_KEY and AI_EMBEDDING_API_KEY.lower() != 'none':
        api_key = AI_EMBEDDING_API_KEY
    elif AI_DESCRIPTION_PROVIDER == 'google' and AI_DESCRIPTION_API_KEY and AI_DESCRIPTION_API_KEY.lower() != 'none':
        api_key = AI_DESCRIPTION_API_KEY

    if api_key:
        genai.configure(api_key=api_key)
PROMPT_TEMPLATES = {
    "code_description": """Analyze the following code and provide a concise description of its purpose and functionality.
Focus on the main functionality, key components, and how they interact.

CODE:
{code}

Respond with a clear, concise description (2-3 sentences) that explains what this code does.
""",

    "code_summary": """Summarize the following code file:```
{code}```

Provide a comprehensive summary that includes:
1. The main purpose of the code
2. Key functions/classes and their responsibilities
3. Important algorithms or patterns used
4. Dependencies and external interactions
5. Any notable optimizations or design decisions

Your summary should be detailed enough for a developer to understand the code's functionality without reading it.
""",

    "error_analysis": """Analyze the following error and suggest possible solutions:

Error message:
{error_message}

Code context:
```
{code_context}
```

Provide:
1. A clear explanation of what's causing the error
2. At least 2-3 potential solutions, with code examples
3. Best practices to avoid this error in the future
""",

    "code_improvement": """Review the following code and suggest improvements:
```
{code}
```

Focus on:
1. Performance optimizations
2. Code readability and maintainability
3. Potential bugs or edge cases
4. Better design patterns or algorithms
5. Modern language features that could be utilized

For each suggestion, provide a brief explanation of why it's an improvement and a code example.
"""
}

def track_performance(provider_key: str):
    """Decorator to track performance metrics for LLM calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            PERFORMANCE_METRICS["total_requests"] += 1
            PERFORMANCE_METRICS["provider_stats"][provider_key]["requests"] += 1

            try:
                result = func(*args, **kwargs)

                elapsed = time.time() - start_time
                PERFORMANCE_METRICS["total_time"] += elapsed
                PERFORMANCE_METRICS["provider_stats"][provider_key]["time"] += elapsed

                if isinstance(result, str):
                    PERFORMANCE_METRICS["total_tokens"] += len(result.split()) * 1.3

                return result
            except Exception as e:
                PERFORMANCE_METRICS["errors"] += 1
                raise

        return wrapper

    return decorator


def generate_embed(text: Union[str, List[str]]) -> List[List[float]]:
    """Generate embeddings for a single text or list of texts.

    Args:
        text (Union[str, List[str]]): Single string or list of strings to embed.

    Returns:
        List[List[float]]: List of embedding vectors.

    Raises:
        ValueError: If EMBEDDING_MODEL is not set.
        ollama.ResponseError: If Ollama encounters an error (e.g., model not found).
        Exception: If Google API encounters an error.
    """
    logger.debug(f"generate_embed called with {len(text) if isinstance(text, list) else 1} text(s)")
    logger.debug(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
    logger.debug(f"AI_EMBEDDING_PROVIDER = {AI_EMBEDDING_PROVIDER}")

    if not EMBEDDING_MODEL:
        logger.warning("EMBEDDING_MODEL not set in environment variables")
        if isinstance(text, str):
            return [[0.0] * 384]
        else:
            return [[0.0] * 384] * len(text)

    if isinstance(text, str):
        text = [text]

    try:
        if AI_EMBEDDING_PROVIDER == "google":
            logger.debug("Using Google provider for embeddings")
            if not AI_EMBEDDING_API_KEY or AI_EMBEDDING_API_KEY.lower() == "none":
                logger.warning("AI_EMBEDDING_API_KEY not set for Google provider")
                return [[0.0] * 384] * len(text)
            try:
                logger.debug("Calling Google API for embeddings")
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text
                )
                logger.debug("Google API returned embeddings successfully")
                return result["embedding"] if isinstance(text, str) else result["embeddings"]
            except Exception as e:
                logger.error(f"Google API error with model {EMBEDDING_MODEL}: {str(e)}")
                return [[0.0] * 384] * len(text)
        else:
            logger.debug("Using Ollama provider for embeddings")
            try:
                try:
                    logger.debug(f"Pulling Ollama model {EMBEDDING_MODEL}")
                    ollama.pull(EMBEDDING_MODEL)
                    logger.debug(f"Successfully pulled Ollama model {EMBEDDING_MODEL}")
                except ollama.ResponseError as pull_error:
                    logger.error(f"Error pulling Ollama model: {str(pull_error)}")
                    if pull_error.status_code != 404:
                        raise pull_error

                logger.debug("Calling Ollama API for embeddings")
                response = ollama.embed(model=EMBEDDING_MODEL, input=text)
                logger.debug("Ollama API returned embeddings successfully")
                return response.embeddings
            except ollama.ResponseError as e:
                logger.error(f"Ollama ResponseError: {str(e)}")
                if "model not found" in str(e).lower():
                    logger.error(f"Model {EMBEDDING_MODEL} not found and could not be pulled")
                    return [[0.0] * 384] * len(text)
                return [[0.0] * 384] * len(text)
    except Exception as e:
        logger.error(f"Unexpected error in generate_embed: {str(e)}")
        return [[0.0] * 384] * len(text)


def validate_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate and normalize message format.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries.

    Returns:
        List[Dict[str, str]]: Validated and normalized messages.

    Raises:
        ValueError: If messages are invalid.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    normalized_messages = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} is not a dictionary")

        if "role" not in msg:
            raise ValueError(f"Message {i} is missing 'role' field")

        if "content" not in msg:
            raise ValueError(f"Message {i} is missing 'content' field")

        role = msg["role"].lower()
        if role not in ["user", "assistant", "system"]:
            role = "user" if role == "human" else "assistant"

        normalized_messages.append({"role": role, "content": str(msg["content"])})

    return normalized_messages


def generate_response(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    template_name: Optional[str] = None,
    template_vars: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a response using the chat model with enhanced features.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
        system_prompt (Optional[str], optional): Optional system prompt to set context. Defaults to None.
        template_name (Optional[str], optional): Name of a prompt template to use. Defaults to None.
        template_vars (Optional[Dict[str, str]], optional): Variables to format the template with. Defaults to None.
        temperature (float, optional): Temperature for response generation. Defaults to 0.7.
        max_tokens (Optional[int], optional): Maximum tokens to generate. Defaults to None.

    Returns:
        str: Generated response text.

    Raises:
        ValueError: If CHAT_MODEL is not set or if parameters are invalid.
        ollama.ResponseError: If Ollama encounters an error (e.g., model not found).
        Exception: If Google API encounters an error.
    """
    if not CHAT_MODEL:
        raise ValueError("CHAT_MODEL not set in environment variables")

    if template_name:
        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found")

        template = PROMPT_TEMPLATES[template_name]
        if template_vars:
            try:
                formatted_prompt = template.format(**template_vars)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}")
        else:
            formatted_prompt = template

        messages = [{"role": "user", "content": formatted_prompt}]

    messages = validate_messages(messages)

    if AI_CHAT_PROVIDER == "google":
        return _generate_response_google(messages, system_prompt, temperature, max_tokens)
    else:
        return _generate_response_ollama(messages, system_prompt, temperature, max_tokens)

@track_performance("google")
def _generate_response_google(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a response using Google AI.

    Args:
        messages (List[Dict[str, str]]): Validated message list.
        system_prompt (Optional[str], optional): System prompt. Defaults to None.
        temperature (float, optional): Temperature. Defaults to 0.7.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.

    Returns:
        str: Generated response.

    Raises:
        ValueError: If API key is not set.
        Exception: If API call fails.
    """
    if not AI_CHAT_API_KEY or AI_CHAT_API_KEY.lower() == "none":
        raise ValueError("AI_CHAT_API_KEY not set for Google provider")

    try:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens if max_tokens else 4096,
            "top_p": 0.95,
            "top_k": 40,
        }

        model = genai.GenerativeModel(
            model_name=CHAT_MODEL,
            generation_config=generation_config,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )

        chat_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": msg["content"]})

        if system_prompt:
            response = model.generate_content(chat_history, system_instruction=system_prompt)
        else:
            response = model.generate_content(chat_history)

        return response.text
    except Exception as e:
        raise Exception(f"Google API error with model {CHAT_MODEL}: {str(e)}")


@track_performance("ollama")
def _generate_response_ollama(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a response using Ollama.

    Args:
        messages (List[Dict[str, str]]): Validated message list.
        system_prompt (Optional[str], optional): System prompt. Defaults to None.
        temperature (float, optional): Temperature. Defaults to 0.7.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.

    Returns:
        str: Generated response.

    Raises:
        ollama.ResponseError: If Ollama encounters an error.
    """
    try:
        try:
            ollama.pull(CHAT_MODEL)
        except ollama.ResponseError as pull_error:
            if pull_error.status_code != 404:
                raise pull_error

        options = {"temperature": temperature}

        if max_tokens:
            options["num_predict"] = max_tokens

        if system_prompt:
            options["system"] = system_prompt

        response = ollama.chat(model=CHAT_MODEL, messages=messages, options=options)
        return response.message.content
    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            raise ollama.ResponseError(f"Model {CHAT_MODEL} not found and could not be pulled", 404)
        raise


async def generate_response_stream(
    messages: List[Dict[str, str]], system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using the chat model.

    Args:
        messages (List[Dict[str, str]]):
            List of message dictionaries with 'role' and 'content'.
        system_prompt (Optional[str], optional):
            Optional system prompt to set context. Defaults to None.

    Yields:
        AsyncGenerator[str, None]: Generated response text chunks.

    Raises:
        ValueError: If CHAT_MODEL is not set.
        ollama.ResponseError: If Ollama encounters an error (e.g., model not found).
        Exception: If Google API encounters an error.
    """
    if not CHAT_MODEL:
        raise ValueError("CHAT_MODEL not set in environment variables")

    try:
        if AI_CHAT_PROVIDER == "google":
            if not AI_CHAT_API_KEY or AI_CHAT_API_KEY.lower() == "none":
                raise ValueError("AI_CHAT_API_KEY not set for Google provider")

            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 4096,
                "top_p": 0.95,
                "top_k": 40,
            }

            model = genai.GenerativeModel(
                model_name=CHAT_MODEL,
                generation_config=generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ],
            )

            chat_history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": msg["content"]})

            if system_prompt:
                response = model.generate_content(
                    chat_history, system_instruction=system_prompt, stream=True
                )
            else:
                response = model.generate_content(chat_history, stream=True)

            async for chunk in response:
                yield chunk.text
        else:
            client = OllamaAsyncClient()
            try:
                try:
                    await client.pull(CHAT_MODEL)
                except ollama.ResponseError as pull_error:
                    if pull_error.status_code != 404:
                        raise pull_error

                if system_prompt:
                    async for chunk in await client.chat(
                        model=CHAT_MODEL,
                        messages=messages,
                        options={"system": system_prompt},
                        stream=True,
                    ):
                        yield chunk.message.content
                else:
                    async for chunk in await client.chat(model=CHAT_MODEL, messages=messages, stream=True):
                        yield chunk.message.content
            except ollama.ResponseError as e:
                if "model not found" in str(e).lower():
                    raise ollama.ResponseError(
                        f"Model {CHAT_MODEL} not found and could not be pulled", 404
                    )
                raise
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        raise


def generate_description(
    prompt: str,
    template_name: Optional[str] = None,
    template_vars: Optional[Dict[str, str]] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using the description model with enhanced features.

    Args:
        prompt (str): The prompt to generate a description from.
        template_name (Optional[str], optional): Name of a prompt template to use. Defaults to None.
        template_vars (Optional[Dict[str, str]], optional): Variables to format the template with. Defaults to None.
        temperature (float, optional): Temperature for response generation. Defaults to 0.3.
        max_tokens (Optional[int], optional): Maximum tokens to generate. Defaults to None.

    Returns:
        str: Generated description text.

    Raises:
        ValueError: If DESCRIPTION_MODEL is not set or if parameters are invalid.
        ollama.ResponseError: If Ollama encounters an error.
        Exception: If Google API encounters an error.
    """
    if not DESCRIPTION_MODEL:
        raise ValueError("DESCRIPTION_MODEL not set in environment variables")

    if template_name:
        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found")

        template = PROMPT_TEMPLATES[template_name]
        if template_vars:
            try:
                formatted_prompt = template.format(**template_vars)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}")
        else:
            formatted_prompt = template

        prompt = formatted_prompt

    if AI_DESCRIPTION_PROVIDER == "google":
        return _generate_description_google(prompt, temperature, max_tokens)
    else:
        return _generate_description_ollama(prompt, temperature, max_tokens)


@track_performance("google")
def _generate_description_google(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Google AI.

    Args:
        prompt (str): The prompt text.
        temperature (float, optional): Temperature. Defaults to 0.3.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ValueError: If API key is not set.
        Exception: If API call fails.
    """
    if not AI_DESCRIPTION_API_KEY or AI_DESCRIPTION_API_KEY.lower() == "none":
        raise ValueError("AI_DESCRIPTION_API_KEY not set for Google provider")

    try:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens if max_tokens else 4096,
            "top_p": 0.95,
            "top_k": 40,
        }

        model = genai.GenerativeModel(
            model_name=DESCRIPTION_MODEL,
            generation_config=generation_config,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Google API error with model {DESCRIPTION_MODEL}: {str(e)}")


@track_performance("ollama")
def _generate_description_ollama(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Ollama.

    Args:
        prompt (str): The prompt text.
        temperature (float, optional): Temperature. Defaults to 0.3.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ollama.ResponseError: If Ollama encounters an error.
    """
    try:
        try:
            ollama.pull(DESCRIPTION_MODEL)
        except ollama.ResponseError as pull_error:
            if pull_error.status_code != 404:
                raise pull_error

        options = {"temperature": temperature}

        if max_tokens:
            options["num_predict"] = max_tokens

        response = ollama.chat(
            model=DESCRIPTION_MODEL, messages=[{"role": "user", "content": prompt}], options=options
        )
        return response.message.content
    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            raise ollama.ResponseError(f"Model {DESCRIPTION_MODEL} not found and could not be pulled", 404)
        raise


def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for LLM usage.

    Returns:
        Dict[str, Any]: Dictionary with performance metrics.
    """
    metrics = PERFORMANCE_METRICS.copy()

    if metrics["total_requests"] > 0:
        metrics["avg_time_per_request"] = metrics["total_time"] / metrics["total_requests"]
        metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["total_requests"]
        metrics["error_rate"] = metrics["errors"] / metrics["total_requests"]
    else:
        metrics["avg_time_per_request"] = 0
        metrics["avg_tokens_per_request"] = 0
        metrics["error_rate"] = 0

    for provider, stats in metrics["provider_stats"].items():
        if stats["requests"] > 0:
            stats["avg_time_per_request"] = stats["time"] / stats["requests"]
        else:
            stats["avg_time_per_request"] = 0

    return metrics


def reset_performance_metrics() -> None:
    """Reset all performance metrics to zero."""
    global PERFORMANCE_METRICS
    PERFORMANCE_METRICS = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_time": 0.0,
        "errors": 0,
        "provider_stats": {
            "ollama": {"requests": 0, "time": 0.0},
            "google": {"requests": 0, "time": 0.0},
        },
    }
