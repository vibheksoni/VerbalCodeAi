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
- Thinking tokens parsing and analysis
- Chat logging
"""

import asyncio
import datetime
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
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

AI_CHAT_PROVIDER: str = os.getenv("AI_CHAT_PROVIDER", "ollama")
AI_EMBEDDING_PROVIDER: str = os.getenv("AI_EMBEDDING_PROVIDER", "ollama")
AI_DESCRIPTION_PROVIDER: str = os.getenv("AI_DESCRIPTION_PROVIDER", "ollama")
AI_CHAT_API_KEY: str = os.getenv("AI_CHAT_API_KEY")
AI_EMBEDDING_API_KEY: str = os.getenv("AI_EMBEDDING_API_KEY")
AI_DESCRIPTION_API_KEY: str = os.getenv("AI_DESCRIPTION_API_KEY")

CHAT_MODEL: str = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
DESCRIPTION_MODEL: str = os.getenv("DESCRIPTION_MODEL")

CHAT_LOGS_ENABLED: bool = os.getenv("CHAT_LOGS", "FALSE").upper() == "TRUE"


@dataclass
class ThinkTokens:
    """Class to store information about thinking tokens in AI responses."""

    total_tokens: int
    total_words: int
    tokens: List[str]
    thinking_found: bool

    def __str__(self) -> str:
        """Return a string representation of the thinking tokens."""
        if not self.thinking_found:
            return "No thinking blocks found"
        return f"Thinking tokens: {self.total_tokens}, Words: {self.total_words}"


def parse_thinking_tokens(
    response: str,
) -> Tuple[str, ThinkTokens, str]:
    """Parse thinking tokens from an AI response.

    Args:
        response (str): The AI response text.

    Returns:
        Tuple[str, ThinkTokens, str]: A tuple containing:
            - The full response (unchanged)
            - A ThinkTokens object with thinking information
            - The response with thinking blocks removed
    """
    thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = thinking_pattern.findall(response)

    if not matches:
        alt_patterns = [
            r"<thinking>(.*?)</thinking>",
            r"\[thinking\](.*?)\[/thinking\]",
            r"\{thinking\}(.*?)\{/thinking\}",
        ]

        for pattern in alt_patterns:
            alt_thinking_pattern = re.compile(pattern, re.DOTALL)
            matches = alt_thinking_pattern.findall(response)
            if matches:
                break

    if not matches:
        tokens_info = ThinkTokens(
            total_tokens=0, total_words=0, tokens=[], thinking_found=False
        )
        return response, tokens_info, response

    all_thinking_text = " ".join(matches)
    tokens = all_thinking_text.split()

    tokens_info = ThinkTokens(
        total_tokens=len(tokens),
        total_words=len(tokens),
        tokens=tokens,
        thinking_found=True,
    )

    clean_response = response
    for pattern in [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"\[thinking\].*?\[/thinking\]",
        r"\{thinking\}.*?\{/thinking\}",
    ]:
        clean_response = re.sub(pattern, "", clean_response, flags=re.DOTALL)

    return response, tokens_info, clean_response


PERFORMANCE_METRICS: Dict[
    str, Union[int, float, Dict[str, Dict[str, Union[int, float]]]]
] = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0.0,
    "errors": 0,
    "provider_stats": {
        "ollama": {"requests": 0, "time": 0.0},
        "google": {"requests": 0, "time": 0.0},
    },
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
    is_small_batch = isinstance(text, str) or (isinstance(text, list) and len(text) <= 3)
    if is_small_batch:
        logger.debug(f"generate_embed called with {len(text) if isinstance(text, list) else 1} text(s)")
        logger.debug(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
        logger.debug(f"AI_EMBEDDING_PROVIDER = {AI_EMBEDDING_PROVIDER}")
    else:
        logger.debug(f"generate_embed called with batch of {len(text)} texts")

    if not EMBEDDING_MODEL:
        logger.warning("EMBEDDING_MODEL not set in environment variables")
        if isinstance(text, str):
            return [[0.0] * 384]
        else:
            return [[0.0] * 384] * len(text)

    single_input = isinstance(text, str)
    if single_input:
        text = [text]

    if not text or (isinstance(text, list) and len(text) == 0):
        logger.warning("Empty input provided to generate_embed")
        return []

    if isinstance(text, list) and len(text) > 100:
        logger.warning(f"Very large batch of {len(text)} texts provided to generate_embed, consider splitting")

    global _embedding_cache
    if not hasattr(generate_embed, '_embedding_cache'):
        generate_embed._embedding_cache = {}

    cache_hits = []
    cache_misses = []
    cache_keys = []

    for i, t in enumerate(text):
        cache_key = hash(t)
        cache_keys.append(cache_key)

        if cache_key in generate_embed._embedding_cache:
            cache_hits.append((i, generate_embed._embedding_cache[cache_key]))
        else:
            cache_misses.append((i, t))

    if len(cache_hits) == len(text):
        if is_small_batch:
            logger.debug("All embeddings found in cache")
        result = [None] * len(text)
        for i, embedding in cache_hits:
            result[i] = embedding
        return result[0] if single_input else result

    texts_to_process = [t for _, t in cache_misses]

    try:
        embeddings_for_misses = []

        if AI_EMBEDDING_PROVIDER == "google":
            if is_small_batch:
                logger.debug("Using Google provider for embeddings")
            if not AI_EMBEDDING_API_KEY or AI_EMBEDDING_API_KEY.lower() == "none":
                logger.warning("AI_EMBEDDING_API_KEY not set for Google provider")
                embeddings_for_misses = [[0.0] * 384] * len(texts_to_process)
            else:
                try:
                    if is_small_batch:
                        logger.debug("Calling Google API for embeddings")
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=texts_to_process
                    )
                    if is_small_batch:
                        logger.debug("Google API returned embeddings successfully")
                    embeddings_for_misses = result["embeddings"]
                except Exception as e:
                    logger.error(f"Google API error with model {EMBEDDING_MODEL}: {str(e)}")
                    embeddings_for_misses = [[0.0] * 384] * len(texts_to_process)
        else:
            if is_small_batch:
                logger.debug("Using Ollama provider for embeddings")
            try:
                if not hasattr(generate_embed, '_model_pulled'):
                    try:
                        logger.debug(f"Pulling Ollama model {EMBEDDING_MODEL}")
                        ollama.pull(EMBEDDING_MODEL)
                        logger.debug(f"Successfully pulled Ollama model {EMBEDDING_MODEL}")
                        generate_embed._model_pulled = True
                    except ollama.ResponseError as pull_error:
                        logger.error(f"Error pulling Ollama model: {str(pull_error)}")
                        if pull_error.status_code != 404:
                            raise pull_error

                if is_small_batch:
                    logger.debug("Calling Ollama API for embeddings")
                response = ollama.embed(model=EMBEDDING_MODEL, input=texts_to_process)
                if is_small_batch:
                    logger.debug("Ollama API returned embeddings successfully")
                embeddings_for_misses = response.embeddings
            except ollama.ResponseError as e:
                logger.error(f"Ollama ResponseError: {str(e)}")
                embeddings_for_misses = [[0.0] * 384] * len(texts_to_process)

        for (i, t), embedding in zip(cache_misses, embeddings_for_misses):
            cache_key = cache_keys[i]
            generate_embed._embedding_cache[cache_key] = embedding

        result = [None] * len(text)
        for i, embedding in cache_hits:
            result[i] = embedding

        for (i, _), embedding in zip(cache_misses, embeddings_for_misses):
            result[i] = embedding

        if len(generate_embed._embedding_cache) > 1000:
            keys_to_remove = list(generate_embed._embedding_cache.keys())[:200]
            for key in keys_to_remove:
                del generate_embed._embedding_cache[key]

        return result[0] if single_input else result

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
    project_path: Optional[str] = None,
    parse_thinking: bool = True,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Union[str, Tuple[str, ThinkTokens, str]]:
    """Generate a response using the chat model with enhanced features.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
        system_prompt (Optional[str], optional): Optional system prompt to set context. Defaults to None.
        template_name (Optional[str], optional): Name of a prompt template to use. Defaults to None.
        template_vars (Optional[Dict[str, str]], optional): Variables to format the template with. Defaults to None.
        temperature (float, optional): Temperature for response generation. Defaults to 0.7.
        max_tokens (Optional[int], optional): Maximum tokens to generate. Defaults to None.
        project_path (Optional[str], optional): Path to the project for chat logging. Defaults to None.
        parse_thinking (bool, optional): Whether to parse thinking tokens. Defaults to True.
        provider (Optional[str], optional): Override the AI provider. Defaults to None (use environment variable).
        api_key (Optional[str], optional): Override the API key. Defaults to None (use environment variable).
        model (Optional[str], optional): Override the model name. Defaults to None (use environment variable).

    Returns:
        Union[str, Tuple[str, ThinkTokens, str]]:
            If parse_thinking is True, returns a tuple containing:
                - The full response
                - A ThinkTokens object with thinking information
                - The response with thinking blocks removed
            Otherwise, returns just the response string.

    Raises:
        ValueError: If model is not set or if parameters are invalid.
        ollama.ResponseError: If Ollama encounters an error (e.g., model not found).
        Exception: If Google API encounters an error.
    """
    chat_provider = provider or AI_CHAT_PROVIDER
    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model or CHAT_MODEL

    if not chat_model:
        raise ValueError("Model not set. Either set CHAT_MODEL environment variable or provide model parameter.")

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

    user_query = ""
    for msg in messages:
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    if chat_provider.lower() == "google":
        response = _generate_response_google(
            messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
        )
    else:
        response = _generate_response_ollama(
            messages, system_prompt, temperature, max_tokens, chat_model
        )

    if CHAT_LOGS_ENABLED and project_path:
        log_chat(user_query, response, project_path)

    if parse_thinking:
        return parse_thinking_tokens(response)
    else:
        return response

@track_performance("google")
def _generate_response_google(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using Google AI.

    Args:
        messages (List[Dict[str, str]]): Validated message list.
        system_prompt (Optional[str], optional): System prompt. Defaults to None.
        temperature (float, optional): Temperature. Defaults to 0.7.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.
        api_key (Optional[str], optional): Override the API key. Defaults to None (use environment variable).
        model_name (Optional[str], optional): Override the model name. Defaults to None (use environment variable).

    Returns:
        str: Generated response.

    Raises:
        ValueError: If API key is not set.
        Exception: If API call fails.
    """
    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model_name or CHAT_MODEL

    if not chat_api_key or chat_api_key.lower() == "none":
        raise ValueError("API key not set for Google provider")

    try:
        if api_key:
            genai.configure(api_key=chat_api_key)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens if max_tokens else 4096,
            "top_p": 0.95,
            "top_k": 40,
        }

        model = genai.GenerativeModel(
            model_name=chat_model,
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
        logger.error(f"Google API error with model {chat_model}: {str(e)}", exc_info=True)
        raise Exception(f"Google API error with model {chat_model}: {str(e)}")


@track_performance("ollama")
def _generate_response_ollama(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using Ollama.

    Args:
        messages (List[Dict[str, str]]): Validated message list.
        system_prompt (Optional[str], optional): System prompt. Defaults to None.
        temperature (float, optional): Temperature. Defaults to 0.7.
        max_tokens (Optional[int], optional): Max tokens. Defaults to None.
        model_name (Optional[str], optional): Override the model name. Defaults to None (use environment variable).

    Returns:
        str: Generated response.

    Raises:
        ollama.ResponseError: If Ollama encounters an error.
    """
    chat_model = model_name or CHAT_MODEL

    try:
        try:
            ollama.pull(chat_model)
        except ollama.ResponseError as pull_error:
            if pull_error.status_code != 404:
                raise pull_error

        options = {"temperature": temperature}

        if max_tokens:
            options["num_predict"] = max_tokens

        if system_prompt:
            options["system"] = system_prompt

        response = ollama.chat(model=chat_model, messages=messages, options=options)
        return response.message.content
    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            raise ollama.ResponseError(f"Model {chat_model} not found and could not be pulled", 404)
        raise


async def generate_response_stream(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    project_path: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using the chat model.

    Args:
        messages (List[Dict[str, str]]):
            List of message dictionaries with 'role' and 'content'.
        system_prompt (Optional[str], optional):
            Optional system prompt to set context. Defaults to None.
        project_path (Optional[str], optional):
            Path to the project for chat logging. Defaults to None.
        provider (Optional[str], optional):
            Override the AI provider. Defaults to None (use environment variable).
        api_key (Optional[str], optional):
            Override the API key. Defaults to None (use environment variable).
        model (Optional[str], optional):
            Override the model name. Defaults to None (use environment variable).

    Yields:
        AsyncGenerator[str, None]: Generated response text chunks.

    Raises:
        ValueError: If model is not set.
        ollama.ResponseError: If Ollama encounters an error (e.g., model not found).
        Exception: If Google API encounters an error.
    """
    chat_provider = provider or AI_CHAT_PROVIDER
    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model or CHAT_MODEL

    if not chat_model:
        raise ValueError("Model not set. Either set CHAT_MODEL environment variable or provide model parameter.")

    user_query = ""
    for msg in messages:
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    full_response = []

    try:
        if chat_provider.lower() == "google":
            if not chat_api_key or chat_api_key.lower() == "none":
                raise ValueError("API key not set for Google provider")

            if api_key:
                genai.configure(api_key=chat_api_key)

            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 4096,
                "top_p": 0.95,
                "top_k": 40,
            }

            model_obj = genai.GenerativeModel(
                model_name=chat_model,
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

            try:
                logger.info("Using non-streaming mode for Google API and simulating streaming")

                if system_prompt:
                    response = model_obj.generate_content(
                        chat_history, system_instruction=system_prompt, stream=False
                    )
                else:
                    response = model_obj.generate_content(chat_history, stream=False)

                response_text = response.text

                chunk_size = 20
                for i in range(0, len(response_text), chunk_size):
                    chunk_text = response_text[i:i+chunk_size]
                    full_response.append(chunk_text)
                    yield chunk_text
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error generating response from Google API: {str(e)}")
                yield f"Error generating response: {str(e)}"
        else:
            client = OllamaAsyncClient()
            try:
                try:
                    await client.pull(chat_model)
                except ollama.ResponseError as pull_error:
                    if pull_error.status_code != 404:
                        raise pull_error

                if system_prompt:
                    async for chunk in await client.chat(
                        model=chat_model,
                        messages=messages,
                        options={"system": system_prompt},
                        stream=True,
                    ):
                        chunk_text = chunk.message.content
                        full_response.append(chunk_text)
                        yield chunk_text
                else:
                    async for chunk in await client.chat(model=chat_model, messages=messages, stream=True):
                        chunk_text = chunk.message.content
                        full_response.append(chunk_text)
                        yield chunk_text
            except ollama.ResponseError as e:
                if "model not found" in str(e).lower():
                    raise ollama.ResponseError(
                        f"Model {chat_model} not found and could not be pulled", 404
                    )
                raise
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        raise

    if CHAT_LOGS_ENABLED and project_path:
        complete_response = "".join(full_response)
        log_chat(user_query, complete_response, project_path)


def generate_description(
    prompt: str,
    template_name: Optional[str] = None,
    template_vars: Optional[Dict[str, str]] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    project_path: Optional[str] = None,
) -> str:
    """Generate a description using the description model with enhanced features.

    Args:
        prompt (str): The prompt to generate a description from.
        template_name (Optional[str], optional): Name of a prompt template to use. Defaults to None.
        template_vars (Optional[Dict[str, str]], optional): Variables to format the template with. Defaults to None.
        temperature (float, optional): Temperature for response generation. Defaults to 0.3.
        max_tokens (Optional[int], optional): Maximum tokens to generate. Defaults to None.
        project_path (Optional[str], optional): Path to the project for chat logging. Defaults to None.

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
        response = _generate_description_google(prompt, temperature, max_tokens)
    else:
        response = _generate_description_ollama(prompt, temperature, max_tokens)
    if CHAT_LOGS_ENABLED and project_path:
        log_chat(prompt, response, project_path)
    return response


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


def log_chat(query: str, response: str, project_path: str) -> None:
    """Log chat interactions to a JSON file.

    Args:
        query (str): The user's query.
        response (str): The AI's response.
        project_path (str): The path to the indexed project.
    """
    if not CHAT_LOGS_ENABLED:
        return

    try:
        logs_dir = Path("chat_logs")
        logs_dir.mkdir(exist_ok=True)

        project_name = Path(project_path).name
        project_logs_dir = logs_dir / project_name
        project_logs_dir.mkdir(exist_ok=True)

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = project_logs_dir / f"{today}.json"

        timestamp = datetime.datetime.now().isoformat()
        chat_entry = {
            "timestamp": timestamp,
            "query": query,
            "response": response,
        }

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {"chats": []}
        else:
            log_data = {"chats": []}

        log_data["chats"].append(chat_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Chat logged to {log_file}")
    except Exception as e:
        logger.error(f"Error logging chat: {e}")
