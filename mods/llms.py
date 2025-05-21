"""LLM integration module for VerbalCodeAI.

This module provides a unified interface to different LLM providers,
with support for text generation, embeddings, and streaming responses.

Supported providers:
- Ollama (local models)
- Google AI (cloud-based models)
- OpenAI (OpenAI models)
- Anthropic (Claude models)
- Groq (high-performance LLMs)
- OpenRouter (various cloud models)

Features:
- Automatic model pulling for Ollama
- Streaming response support
- Error handling and validation
- Configurable system prompts
- Performance metrics
- Thinking tokens parsing and analysis
- Chat logging
- Memory management for conversations
- Token optimization
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
    Set,
    Tuple,
    Union,
)

import google.generativeai as genai
import ollama
import requests
from dotenv import load_dotenv
from ollama import AsyncClient as OllamaAsyncClient

import openai
from anthropic import Anthropic, AsyncAnthropic
from groq import Groq, AsyncGroq

logger = logging.getLogger("VerbalCodeAI.LLMs")

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

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
MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "TRUE").upper() == "TRUE"
MAX_MEMORY_ITEMS: int = int(os.getenv("MAX_MEMORY_ITEMS", "10"))

def get_current_provider() -> str:
    """Get the current AI provider based on environment variables."""
    if anthropic_client:
        return "anthropic"
    elif openai_client:
        return "openai"
    else:
        return "ollama" if AI_CHAT_PROVIDER == "ollama" else "google" if AI_CHAT_PROVIDER == "google" else "openrouter" if AI_CHAT_PROVIDER == "openrouter" else "groq" if AI_CHAT_PROVIDER == "groq" else "unknown"

class ConversationMemory:
    """Manages memory for AI conversations to provide context and reduce redundant API calls."""

    def __init__(self, max_items: int = MAX_MEMORY_ITEMS) -> None:
        """Initialize the conversation memory.

        Args:
            max_items (int): Maximum number of memory items to store.
        """
        self.memories: List[Dict[str, Any]] = []
        self.max_items: int = max_items
        self.embeddings_cache: Dict[str, List[float]] = {}
        self.logger: logging.Logger = logging.getLogger("VerbalCodeAI.LLMs.Memory")

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory item.

        Args:
            content (str): The memory content.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the memory.
        """
        if not content.strip():
            return

        memory_item = {
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        for existing in self.memories:
            if existing["content"] == content:
                existing["timestamp"] = memory_item["timestamp"]
                existing["metadata"].update(memory_item["metadata"])
                return

        self.memories.append(memory_item)

        if len(self.memories) > self.max_items:
            self.memories = self.memories[-self.max_items:]

        if content in self.embeddings_cache:
            del self.embeddings_cache[content]

        self.logger.debug(f"Added memory: {content[:50]}...")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available.

        Args:
            text (str): Text to get embedding for.

        Returns:
            List[float]: Embedding vector.
        """
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        try:
            embedding = generate_embed(text)
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return []

    def _calculate_similarity(self, query_embedding: List[float], memory_embedding: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            query_embedding (List[float]): Query embedding.
            memory_embedding (List[float]): Memory embedding.

        Returns:
            float: Similarity score between 0 and 1.
        """
        if not query_embedding or not memory_embedding:
            return 0.0

        dot_product = sum(a * b for a, b in zip(query_embedding, memory_embedding))

        magnitude_a = sum(a * a for a in query_embedding) ** 0.5
        magnitude_b = sum(b * b for b in memory_embedding) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def get_relevant_memories(self, query: str, max_results: int = 3, similarity_threshold: float = 0.5) -> List[str]:
        """Get memories relevant to the current query using semantic search.

        Args:
            query (str): The current query.
            max_results (int): Maximum number of relevant memories to return.
            similarity_threshold (float): Minimum similarity score to include a memory.

        Returns:
            List[str]: List of relevant memory contents.
        """
        if not self.memories:
            return []

        try:
            query_embedding = self._get_embedding(query)

            if not query_embedding:
                self.logger.warning("Falling back to recency-based memory retrieval")
                return [m["content"] for m in self.memories[-max_results:]]

            memory_scores = []
            for memory in self.memories:
                content = memory["content"]
                memory_embedding = self._get_embedding(content)

                if memory_embedding:
                    similarity = self._calculate_similarity(query_embedding, memory_embedding)
                    memory_scores.append((content, similarity))
                else:
                    memory_scores.append((content, 0.0))

            memory_scores.sort(key=lambda x: x[1], reverse=True)

            relevant_memories = [
                content for content, score in memory_scores if score >= similarity_threshold
            ][:max_results]

            if not relevant_memories and self.memories:
                return [self.memories[-1]["content"]]

            return relevant_memories

        except Exception as e:
            self.logger.error(f"Error in semantic memory retrieval: {e}")
            return [m["content"] for m in self.memories[-max_results:]]

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self.embeddings_cache = {}
        self.logger.debug("Cleared all memories")

    def format_for_prompt(self, query: str = "", max_items: int = 3) -> str:
        """Format memories for inclusion in a prompt.

        Args:
            query (str): The current query to find relevant memories.
            max_items (int): Maximum number of memories to include.

        Returns:
            str: Formatted memories string.
        """
        if not self.memories:
            return ""

        if query:
            relevant_memories = self.get_relevant_memories(query, max_results=max_items)
            if not relevant_memories:
                return ""

            formatted = "Previous information from this conversation that may be relevant to your query:\n"

            for i, memory_content in enumerate(relevant_memories, 1):
                formatted += f"{i}. {memory_content}\n"
        else:
            recent_memories = self.memories[-max_items:]
            formatted = "Previous information from this conversation:\n"

            for i, memory in enumerate(recent_memories, 1):
                formatted += f"{i}. {memory['content']}\n"

        return formatted


conversation_memory = ConversationMemory()


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


def parse_thinking_tokens(response: str) -> Tuple[str, ThinkTokens, str]:
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
        "openai": {"requests": 0, "time": 0.0},
        "anthropic": {"requests": 0, "time": 0.0},
        "groq": {"requests": 0, "time": 0.0},
        "openrouter": {"requests": 0, "time": 0.0},
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

openai_client = None
if (AI_CHAT_PROVIDER == 'openai' or
        AI_EMBEDDING_PROVIDER == 'openai' or
        AI_DESCRIPTION_PROVIDER == 'openai'):

    api_key: Optional[str] = None
    if AI_CHAT_PROVIDER == 'openai' and AI_CHAT_API_KEY and AI_CHAT_API_KEY.lower() != 'none':
        api_key = AI_CHAT_API_KEY
    elif AI_EMBEDDING_PROVIDER == 'openai' and AI_EMBEDDING_API_KEY and AI_EMBEDDING_API_KEY.lower() != 'none':
        api_key = AI_EMBEDDING_API_KEY
    elif AI_DESCRIPTION_PROVIDER == 'openai' and AI_DESCRIPTION_API_KEY and AI_DESCRIPTION_API_KEY.lower() != 'none':
        api_key = AI_DESCRIPTION_API_KEY

    if api_key:
        openai_client = openai.OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialized")

anthropic_client = None
if AI_CHAT_PROVIDER == 'anthropic':
    if AI_CHAT_API_KEY and AI_CHAT_API_KEY.lower() != 'none':
        anthropic_client = Anthropic(api_key=AI_CHAT_API_KEY)
        logger.debug("Anthropic client initialized for chat")
    else:
        logger.warning("AI_CHAT_API_KEY not set or invalid for Anthropic provider")
elif AI_DESCRIPTION_PROVIDER == 'anthropic':
    if AI_DESCRIPTION_API_KEY and AI_DESCRIPTION_API_KEY.lower() != 'none':
        anthropic_client = Anthropic(api_key=AI_DESCRIPTION_API_KEY)
        logger.debug("Anthropic client initialized for description")
    else:
        logger.warning("AI_DESCRIPTION_API_KEY not set or invalid for Anthropic provider")

groq_client = None
PROMPT_TEMPLATES = {
    "code_description": """Analyze the following code and provide a concise description of its purpose and functionality.
Focus on the main functionality, key components, and how they interact.

CODE:
{code}

Respond with a clear, concise description (2-3 sentences) that explains what this code does.
Include the primary purpose, key components, and any notable design patterns or techniques used.
""",

    "code_summary": """Summarize the following code file:```
{code}```

Provide a comprehensive summary that includes:
1. The main purpose of the code and its role in the larger system
2. Key functions/classes and their responsibilities with brief explanations of their interfaces
3. Important algorithms or patterns used and why they were chosen
4. Dependencies and external interactions, including any APIs or libraries leveraged
5. Any notable optimizations or design decisions that impact performance or maintainability
6. Potential areas for improvement or technical debt

Your summary should be detailed enough for a developer to understand the code's functionality without reading it,
while highlighting the most important aspects that would be relevant for maintenance or extension.
""",

    "error_analysis": """Analyze the following error and suggest possible solutions:

Error message:
{error_message}

Code context:
```
{code_context}
```

Provide:
1. A clear explanation of what's causing the error, including the specific line or operation that's failing
2. At least 2-3 potential solutions, with code examples that can be directly implemented
3. Best practices to avoid this error in the future
4. Any diagnostic steps that could help further isolate the issue if your solutions don't resolve it
5. Potential side effects or considerations when implementing the suggested fixes
""",

    "code_improvement": """Review the following code and suggest improvements:
```
{code}
```

Focus on:
1. Performance optimizations that would have measurable impact
2. Code readability and maintainability improvements
3. Potential bugs, edge cases, or error handling gaps
4. Better design patterns or algorithms that would simplify the code
5. Modern language features that could be utilized to make the code more concise or safer
6. Security considerations or potential vulnerabilities
7. Testing strategies that would help ensure code correctness

For each suggestion:
- Explain why it's an improvement and what specific problem it solves
- Provide a concrete code example showing the implementation
- Note any trade-offs or considerations when adopting the change
- Prioritize suggestions based on their impact and implementation difficulty
""",

    "max_mode_analysis": """You are an expert code analysis assistant with deep knowledge of software architecture and programming languages.
Analyze the following code files and provide comprehensive insights about the codebase.

CODE FILES:
{code_files}

USER QUERY:
{query}

Provide a detailed response that:
1. Directly answers the user's query with specific references to the code
2. Explains relevant code sections with line numbers and file names
3. Identifies patterns, relationships, and dependencies between components
4. Suggests improvements or solutions if applicable
5. Includes code examples when helpful

Your analysis should be thorough, technically precise, and focused on helping the user understand
exactly how the code works in relation to their query. Prioritize depth of understanding over breadth.
""",

    "intent_detection": """Analyze the following user message and determine its intent.
The message is from a user interacting with a code assistant AI that helps with understanding and navigating codebases.

USER MESSAGE:
{message}

Classify the intent into one of these categories:
1. GREETING: Simple greetings like "hi", "hello", "hey", etc.
2. FAREWELL: Messages like "bye", "goodbye", "see you", etc.
3. GRATITUDE: Messages expressing thanks like "thank you", "thanks", etc.
4. SMALL_TALK: General chitchat not related to code or the project (e.g., "how are you?")
5. SIMPLE_QUESTION: A simple question that doesn't require deep code analysis (e.g., "what's your name?")
6. CODE_QUESTION: A question about code or the codebase that requires analysis (e.g., "how does this function work?", "what's this codebase about?", "explain the project structure")
7. FEATURE_REQUEST: User asking for a new feature or enhancement (e.g., "can you add support for...")
8. HELP_REQUEST: User asking for help with the tool itself (e.g., "how do I use this tool?")
9. FEEDBACK: User providing feedback about the assistant (e.g., "you're doing great", "that wasn't helpful")
10. OTHER: Any other type of message

Important guidelines:
- Questions about the codebase, project structure, or specific code should be classified as CODE_QUESTION
- If the message asks about what the codebase does or what it's for, classify as CODE_QUESTION
- If in doubt between SIMPLE_QUESTION and CODE_QUESTION, prefer CODE_QUESTION
- If the message contains both a greeting and a question, prioritize the question part

Respond with ONLY the category name, nothing else.
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
        List[List[float]]: List of embedding vectors. Each vector is a list of floats.
        For a single input text, returns a list containing one embedding vector.
        For multiple input texts, returns a list of embedding vectors, one for each input text.
        If the embedding generation fails, returns a list of default embeddings (all zeros).

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
        elif AI_EMBEDDING_PROVIDER == "openai":
            if is_small_batch:
                logger.debug("Using OpenAI provider for embeddings")
            if not openai_client:
                logger.warning("OpenAI client not initialized for embeddings")
                embeddings_for_misses = [[0.0] * 384] * len(texts_to_process)
            else:
                try:
                    if is_small_batch:
                        logger.debug("Calling OpenAI API for embeddings")

                    batch_size = 100
                    all_embeddings = []

                    for i in range(0, len(texts_to_process), batch_size):
                        batch = texts_to_process[i:i+batch_size]
                        response = openai_client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            input=batch
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)

                    if is_small_batch:
                        logger.debug("OpenAI API returned embeddings successfully")

                    embeddings_for_misses = all_embeddings
                except Exception as e:
                    logger.error(f"OpenAI API error with model {EMBEDDING_MODEL}: {str(e)}")
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

                if hasattr(response, 'embeddings'):
                    embeddings_for_misses = []
                    for embedding in response.embeddings:
                        if not isinstance(embedding, list):
                            logger.warning(f"Embedding has wrong type: {type(embedding)}, using default embedding")
                            embeddings_for_misses.append([0.0] * 384)
                        elif len(embedding) == 0:
                            logger.warning("Embedding is empty, using default embedding")
                            embeddings_for_misses.append([0.0] * 384)
                        else:
                            if isinstance(embedding[0], list):
                                embeddings_for_misses.append(embedding)
                            else:
                                embeddings_for_misses.append(embedding)
                else:
                    logger.warning("Ollama response does not have embeddings attribute")
                    embeddings_for_misses = [[0.0] * 384] * len(texts_to_process)
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
    use_memory: bool = True,
    add_to_memory: bool = True,
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
        use_memory (bool, optional): Whether to use conversation memory. Defaults to True.
        add_to_memory (bool, optional): Whether to add the response to memory. Defaults to True.

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

    if MEMORY_ENABLED and use_memory and conversation_memory.memories:
        memory_text = conversation_memory.format_for_prompt(query=user_query)
        if memory_text:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{memory_text}"
            else:
                system_prompt = memory_text
            logger.debug("Added memory context to prompt using semantic search")

    try:
        if chat_provider.lower() == "google":
            response = _generate_response_google(
                messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
            )
        elif chat_provider.lower() == "openai":
            response = _generate_response_openai(
                messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
            )
        elif chat_provider.lower() == "anthropic":
            response = _generate_response_anthropic(
                messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
            )
        elif chat_provider.lower() == "groq":
            response = _generate_response_groq(
                messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
            )
        elif chat_provider.lower() == "openrouter":
            try:
                response = _generate_response_openrouter(
                    messages, system_prompt, temperature, max_tokens, chat_api_key, chat_model
                )
            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"OpenRouter rate limit hit. Falling back to Ollama: {str(e)}")
                    response = _generate_response_ollama(
                        messages, system_prompt, temperature, max_tokens, "llama3.2"
                    )
                else:
                    raise
        else:
            response = _generate_response_ollama(
                messages, system_prompt, temperature, max_tokens, chat_model
            )
    except Exception as e:
        error_msg = f"Error generating response with provider '{chat_provider}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)

    chat_id = ""
    if CHAT_LOGS_ENABLED and project_path:
        chat_id = log_chat(user_query, response, project_path)

    if MEMORY_ENABLED and add_to_memory:
        _, _, clean_response = parse_thinking_tokens(response)

        memory_entry = create_memory_entry(user_query, clean_response)
        if memory_entry:
            conversation_memory.add_memory(
                memory_entry,
                metadata={
                    "query": user_query,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chat_id": chat_id
                }
            )

    if parse_thinking:
        return parse_thinking_tokens(response)
    else:
        return response


def create_memory_entry(query: str, response: str, max_length: int = 200) -> str:
    """Create a concise memory entry from a query and response.

    Args:
        query (str): The user's query.
        response (str): The AI's response.
        max_length (int, optional): Maximum length of the memory entry. Defaults to 200.

    Returns:
        str: A concise memory entry, or empty string if it couldn't be created.
    """
    if len(response) <= max_length:
        return response

    try:
        if len(query) < 50:
            prefix = f"Q: {query.strip()}\nA: "
            max_response_length = max_length - len(prefix)

            if max_response_length > 50:
                paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
                if paragraphs:
                    first_para = paragraphs[0]
                    if len(first_para) <= max_response_length:
                        return prefix + first_para

                    sentences = [s.strip() for s in first_para.split('.') if s.strip()]
                    result = ""
                    for sentence in sentences:
                        if len(result) + len(sentence) + 1 <= max_response_length:
                            result += sentence + "."
                        else:
                            break

                    if result:
                        return prefix + result

        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            first_para = paragraphs[0]
            if len(first_para) <= max_length:
                return first_para

            sentences = [s.strip() for s in first_para.split('.') if s.strip()]
            result = ""
            for sentence in sentences:
                if len(result) + len(sentence) + 1 <= max_length:
                    result += sentence + "."
                else:
                    break

            if result:
                return result

        return response[:max_length] + "..."
    except Exception as e:
        logger.error(f"Error creating memory entry: {e}")
        return response[:max_length] + "..."


def detect_intent(query: str) -> str:
    """Detect the intent of a user query using the LLM.

    Args:
        query (str): The user query.

    Returns:
        str: The detected intent category.
    """
    logger.debug(f"Detecting intent for query: {query}")

    if len(query.strip()) <= 5:
        lower_query = query.lower().strip()
        if lower_query in ["hi", "hey", "hello"]:
            logger.debug("Quick intent detection: GREETING")
            return "GREETING"
        if lower_query in ["bye", "goodbye"]:
            logger.debug("Quick intent detection: FAREWELL")
            return "FAREWELL"
        if lower_query in ["thanks", "thx", "ty"]:
            logger.debug("Quick intent detection: GRATITUDE")
            return "GRATITUDE"

    try:
        template_vars = {"message": query}
        intent_response = generate_response(
            messages=[],
            template_name="intent_detection",
            template_vars=template_vars,
            temperature=0.1,
            parse_thinking=False,
            use_memory=False,
            add_to_memory=False
        )

        intent = intent_response.strip().upper()

        valid_intents = [
            "GREETING", "FAREWELL", "GRATITUDE", "SMALL_TALK",
            "SIMPLE_QUESTION", "CODE_QUESTION", "FEATURE_REQUEST",
            "HELP_REQUEST", "FEEDBACK", "OTHER"
        ]

        if intent in valid_intents:
            logger.debug(f"LLM detected intent: {intent}")
            return intent
        else:
            logger.warning(f"Invalid intent detected: {intent}, defaulting to CODE_QUESTION")
            return "CODE_QUESTION"

    except Exception as e:
        logger.error(f"Error detecting intent: {e}")
        return "CODE_QUESTION"


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
            system_instruction=system_prompt
        )

        chat_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": msg["content"]})

        if system_prompt:
            response = model.generate_content(chat_history)
        else:
            response = model.generate_content(chat_history)

        return response.text
    except Exception as e:
        logger.error(f"Google API error with model {chat_model}: {str(e)}", exc_info=True)
        raise Exception(f"Google API error with model {chat_model}: {str(e)}")


@track_performance("openai")
def _generate_response_openai(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using OpenAI.

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
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global openai_client

    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model_name or CHAT_MODEL

    if not chat_api_key or chat_api_key.lower() == "none":
        raise ValueError("API key not set for OpenAI provider")

    try:
        if api_key or not openai_client:
            openai_client = openai.OpenAI(api_key=chat_api_key)
            logger.debug("OpenAI client initialized or reinitialized with provided API key")

        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        completion_params = {
            "model": chat_model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens:
            completion_params["max_tokens"] = max_tokens

        response = openai_client.chat.completions.create(**completion_params)

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error with model {chat_model}: {str(e)}", exc_info=True)
        raise Exception(f"OpenAI API error with model {chat_model}: {str(e)}")


@track_performance("anthropic")
def _generate_response_anthropic(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using Anthropic Claude.

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
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global anthropic_client

    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model_name or CHAT_MODEL

    if not chat_api_key or chat_api_key.lower() == "none":
        raise ValueError("API key not set for Anthropic provider")

    try:
        if api_key or not anthropic_client:
            anthropic_client = Anthropic(api_key=chat_api_key)
            logger.debug("Anthropic client initialized or reinitialized with provided API key")

        formatted_messages = []
        
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            elif not system_prompt:
                system_prompt = msg["content"]

        if not max_tokens:
            max_tokens = 4096

        completion_params = {
            "model": chat_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            completion_params["system"] = system_prompt

        response = anthropic_client.messages.create(**completion_params)

        return response.content[0].text
    except Exception as e:
        logger.error(f"Anthropic API error with model {chat_model}: {str(e)}", exc_info=True)
        raise Exception(f"Anthropic API error with model {chat_model}: {str(e)}")


@track_performance("groq")
def _generate_response_groq(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using Groq.

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
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global groq_client

    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model_name or CHAT_MODEL

    if not chat_api_key or chat_api_key.lower() == "none":
        raise ValueError("API key not set for Groq provider")

    try:
        local_groq_client = Groq(api_key=chat_api_key)
        logger.debug("Groq client initialized with provided API key")

        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        completion_params = {
            "model": chat_model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens:
            completion_params["max_tokens"] = max_tokens

        response = local_groq_client.chat.completions.create(**completion_params)

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error with model {chat_model}: {str(e)}", exc_info=True)
        raise Exception(f"Groq API error with model {chat_model}: {str(e)}")


@track_performance("openrouter")
def _generate_response_openrouter(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate a response using OpenRouter.

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
        Exception: If OpenRouter API encounters an error.
    """
    import time

    chat_api_key = api_key or AI_CHAT_API_KEY
    chat_model = model_name or CHAT_MODEL

    if not chat_api_key or chat_api_key.lower() == "none":
        raise ValueError("API key not set for OpenRouter provider")

    formatted_messages = []

    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})

    formatted_messages.extend(messages)

    payload = {
        "model": chat_model,
        "messages": formatted_messages,
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {chat_api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 3
    retry_delay = 2

    for retry in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get('X-RateLimit-Reset', 0)) / 1000 - time.time()
                if retry_after <= 0:
                    retry_after = retry_delay * (2 ** retry)

                logger.warning(f"OpenRouter rate limit hit. Retrying in {retry_after:.1f} seconds. Attempt {retry+1}/{max_retries}")

                if retry < max_retries - 1:
                    time.sleep(min(retry_after, 15))
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                    raise Exception(f"Rate limit exceeded: {error_msg}. Try again in a few minutes or switch to a different provider.")

            if response.status_code != 200:
                error_message = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_message)

                if retry < max_retries - 1 and response.status_code >= 500:
                    delay = retry_delay * (2 ** retry)
                    logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_message)

            response_data = response.json()

            if "choices" not in response_data or len(response_data["choices"]) == 0:
                error_message = "OpenRouter API returned no choices"
                logger.warning(f"{error_message}. Attempt {retry+1}/{max_retries}")

                if retry < max_retries - 1:
                    delay = retry_delay * (2 ** retry)
                    logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_message)

            message_content = response_data["choices"][0]["message"]["content"]
            return message_content

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error when calling OpenRouter API: {str(e)}", exc_info=True)

            if retry < max_retries - 1:
                delay = retry_delay * (2 ** retry)
                logger.warning(f"Network error. Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                time.sleep(delay)
            else:
                raise Exception(f"Network error when calling OpenRouter API: {str(e)}")

        except Exception as e:
            logger.error(f"OpenRouter API error with model {chat_model}: {str(e)}", exc_info=True)
            raise Exception(f"OpenRouter API error with model {chat_model}: {str(e)}")

    raise Exception(f"Failed to get response from OpenRouter after {max_retries} attempts")


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
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    use_memory: bool = True,
    add_to_memory: bool = True
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
        max_tokens (Optional[int], optional):
            Maximum tokens to generate. Defaults to None.
        use_memory (bool, optional):
            Whether to use conversation memory. Defaults to True.
        add_to_memory (bool, optional):
            Whether to add the response to memory. Defaults to True.

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

    if MEMORY_ENABLED and use_memory and conversation_memory.memories:
        memory_text = conversation_memory.format_for_prompt(query=user_query)
        if memory_text:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{memory_text}"
            else:
                system_prompt = memory_text
            logger.debug("Added memory context to streaming prompt using semantic search")

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
                system_instruction=system_prompt
            )

            chat_history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": msg["content"]})

            try:
                logger.info("Using non-streaming mode for Google API and simulating streaming")

                if system_prompt:
                    response = model_obj.generate_content(
                        chat_history, stream=False
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
        elif chat_provider.lower() == "openai":
            global openai_client

            if not chat_api_key or chat_api_key.lower() == "none":
                raise ValueError("API key not set for OpenAI provider")

            try:
                if api_key or not openai_client:
                    openai_client = openai.OpenAI(api_key=chat_api_key)
                    logger.debug("OpenAI client initialized or reinitialized with provided API key")

                formatted_messages = []

                if system_prompt:
                    formatted_messages.append({"role": "system", "content": system_prompt})

                for msg in messages:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})

                logger.info("Using streaming mode for OpenAI API")

                completion_params = {
                    "model": chat_model,
                    "messages": formatted_messages,
                    "temperature": 0.7,
                    "stream": True
                }

                if max_tokens:
                    completion_params["max_tokens"] = max_tokens

                stream = openai_client.chat.completions.create(**completion_params)

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        full_response.append(chunk_text)
                        yield chunk_text

            except Exception as e:
                logger.error(f"Error generating response from OpenAI API: {str(e)}")
                yield f"Error generating response: {str(e)}"
        elif chat_provider.lower() == "anthropic":
            global anthropic_client

            if not chat_api_key or chat_api_key.lower() == "none":
                raise ValueError("API key not set for Anthropic provider")

            try:
                if api_key or not anthropic_client:
                    anthropic_client = Anthropic(api_key=chat_api_key)
                    logger.debug("Anthropic client initialized or reinitialized with provided API key")

                formatted_messages = []

                for msg in messages:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})

                logger.info("Using streaming mode for Anthropic API")

                if not max_tokens:
                    max_tokens = 4096

                completion_params = {
                    "model": chat_model,
                    "messages": formatted_messages,
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "stream": True
                }

                if system_prompt:
                    completion_params["system"] = system_prompt

                with anthropic_client.messages.stream(**completion_params) as stream:
                    for text in stream.text_stream:
                        full_response.append(text)
                        yield text

            except Exception as e:
                logger.error(f"Error generating response from Anthropic API: {str(e)}")
                yield f"Error generating response: {str(e)}"
        elif chat_provider.lower() == "groq":
            global groq_client

            if not chat_api_key or chat_api_key.lower() == "none":
                raise ValueError("API key not set for Groq provider")

            try:
                local_groq_client = Groq(api_key=chat_api_key)
                logger.debug("Groq client initialized with provided API key")

                formatted_messages = []

                if system_prompt:
                    formatted_messages.append({"role": "system", "content": system_prompt})

                for msg in messages:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})

                logger.info("Using streaming mode for Groq API")

                completion_params = {
                    "model": chat_model,
                    "messages": formatted_messages,
                    "temperature": 0.7,
                    "stream": True
                }

                if max_tokens:
                    completion_params["max_tokens"] = max_tokens

                stream = local_groq_client.chat.completions.create(**completion_params)

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        full_response.append(chunk_text)
                        yield chunk_text

            except Exception as e:
                logger.error(f"Error generating response from Groq API: {str(e)}")
                yield f"Error generating response: {str(e)}"
        elif chat_provider.lower() == "openrouter":
            if not chat_api_key or chat_api_key.lower() == "none":
                raise ValueError("API key not set for OpenRouter provider")

            import time

            formatted_messages = []

            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})

            formatted_messages.extend(messages)

            payload = {
                "model": chat_model,
                "messages": formatted_messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {chat_api_key}",
                "Content-Type": "application/json"
            }

            max_retries = 3
            retry_delay = 2

            for retry in range(max_retries):
                try:
                    logger.info(f"Using non-streaming mode for OpenRouter API and simulating streaming (attempt {retry+1}/{max_retries})")

                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 429:
                        retry_after = int(response.headers.get('X-RateLimit-Reset', 0)) / 1000 - time.time()
                        if retry_after <= 0:
                            retry_after = retry_delay * (2 ** retry)

                        error_msg = f"OpenRouter rate limit hit. Retrying in {retry_after:.1f} seconds. Attempt {retry+1}/{max_retries}"
                        logger.warning(error_msg)
                        yield f"Rate limit exceeded. Retrying... ({retry+1}/{max_retries})"

                        if retry < max_retries - 1:
                            await asyncio.sleep(min(retry_after, 15))
                            continue
                        else:
                            error_data = response.json() if response.text else {}
                            error_msg = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                            error_message = f"Rate limit exceeded: {error_msg}. Try again in a few minutes or switch to a different provider."
                            logger.error(error_message)
                            yield f"Error: {error_message}"
                            return

                    if response.status_code != 200:
                        error_message = f"OpenRouter API error: {response.status_code} - {response.text}"
                        logger.error(error_message)

                        if retry < max_retries - 1 and response.status_code >= 500:
                            delay = retry_delay * (2 ** retry)
                            logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                            time.sleep(delay)
                            continue
                        else:
                            yield f"Error generating response: {error_message}"
                            return

                    response_data = response.json()

                    if "choices" not in response_data or len(response_data["choices"]) == 0:
                        error_message = "OpenRouter API returned no choices"
                        logger.warning(f"{error_message}. Attempt {retry+1}/{max_retries}")

                        if retry < max_retries - 1:
                            delay = retry_delay * (2 ** retry)
                            logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(error_message)
                            yield f"Error generating response: {error_message}"
                            return

                    response_text = response_data["choices"][0]["message"]["content"]

                    chunk_size = 20
                    for i in range(0, len(response_text), chunk_size):
                        chunk_text = response_text[i:i+chunk_size]
                        full_response.append(chunk_text)
                        yield chunk_text
                        await asyncio.sleep(0.01)

                    break

                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error when calling OpenRouter API: {str(e)}", exc_info=True)

                    if retry < max_retries - 1:
                        delay = retry_delay * (2 ** retry)
                        logger.warning(f"Network error. Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                        yield f"Network error. Retrying... ({retry+1}/{max_retries})"
                        await asyncio.sleep(delay)
                    else:
                        yield f"Error: Network error when calling OpenRouter API: {str(e)}"
                        return

                except Exception as e:
                    logger.error(f"Error generating response from OpenRouter API: {str(e)}")
                    yield f"Error generating response: {str(e)}"
                    return
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

    complete_response = "".join(full_response)

    chat_id = ""
    if CHAT_LOGS_ENABLED and project_path:
        chat_id = log_chat(user_query, complete_response, project_path)

    if MEMORY_ENABLED and add_to_memory:
        _, _, clean_response = parse_thinking_tokens(complete_response)

        memory_entry = create_memory_entry(user_query, clean_response)
        if memory_entry:
            conversation_memory.add_memory(
                memory_entry,
                metadata={
                    "query": user_query,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chat_id": chat_id
                }
            )


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
        template_name (str, optional): Name of a prompt template to use. Defaults to None.
        template_vars (Dict[str, str], optional): Variables to format the template with. Defaults to None.
        temperature (float, optional): Temperature for response generation. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens to generate. Defaults to None.
        project_path (str, optional): Path to the project for chat logging. Defaults to None.

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

    try:
        if AI_DESCRIPTION_PROVIDER == "google":
            response = _generate_description_google(prompt, temperature, max_tokens)
        elif AI_DESCRIPTION_PROVIDER == "openai":
            response = _generate_description_openai(prompt, temperature, max_tokens)
        elif AI_DESCRIPTION_PROVIDER == "anthropic":
            response = _generate_description_anthropic(prompt, temperature, max_tokens)
        elif AI_DESCRIPTION_PROVIDER == "groq":
            response = _generate_description_groq(prompt, temperature, max_tokens)
        elif AI_DESCRIPTION_PROVIDER == "openrouter":
            try:
                response = _generate_description_openrouter(prompt, temperature, max_tokens)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"OpenRouter rate limit hit. Falling back to Ollama: {str(e)}")
                    response = _generate_description_ollama(prompt, temperature, max_tokens)
                else:
                    raise
        else:
            response = _generate_description_ollama(prompt, temperature, max_tokens)
    except Exception as e:
        error_msg = f"Error generating description with provider '{AI_DESCRIPTION_PROVIDER}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)
    if CHAT_LOGS_ENABLED and project_path:
        log_chat(prompt, response, project_path)
    return response


@track_performance("openrouter")
def _generate_description_openrouter(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using OpenRouter.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ValueError: If API key is not set.
        Exception: If API call fails.
    """
    import time

    if not AI_DESCRIPTION_API_KEY or AI_DESCRIPTION_API_KEY.lower() == "none":
        raise ValueError("AI_DESCRIPTION_API_KEY not set for OpenRouter provider")

    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": DESCRIPTION_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {AI_DESCRIPTION_API_KEY}",
        "Content-Type": "application/json"
    }

    max_retries = 3
    retry_delay = 2

    for retry in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get('X-RateLimit-Reset', 0)) / 1000 - time.time()
                if retry_after <= 0:
                    retry_after = retry_delay * (2 ** retry)

                logger.warning(f"OpenRouter rate limit hit. Retrying in {retry_after:.1f} seconds. Attempt {retry+1}/{max_retries}")

                if retry < max_retries - 1:
                    time.sleep(min(retry_after, 15))
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                    raise Exception(f"Rate limit exceeded: {error_msg}. Try again in a few minutes or switch to a different provider.")

            if response.status_code != 200:
                error_message = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_message)

                if retry < max_retries - 1 and response.status_code >= 500:
                    delay = retry_delay * (2 ** retry)
                    logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_message)

            response_data = response.json()

            if "choices" not in response_data or len(response_data["choices"]) == 0:
                error_message = "OpenRouter API returned no choices"
                logger.warning(f"{error_message}. Attempt {retry+1}/{max_retries}")

                if retry < max_retries - 1:
                    delay = retry_delay * (2 ** retry)
                    logger.warning(f"Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_message)

            message_content = response_data["choices"][0]["message"]["content"]
            return message_content

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error when calling OpenRouter API: {str(e)}", exc_info=True)

            if retry < max_retries - 1:
                delay = retry_delay * (2 ** retry)
                logger.warning(f"Network error. Retrying in {delay} seconds. Attempt {retry+1}/{max_retries}")
                time.sleep(delay)
            else:
                raise Exception(f"Network error when calling OpenRouter API: {str(e)}")

        except Exception as e:
            logger.error(f"OpenRouter API error with model {DESCRIPTION_MODEL}: {str(e)}", exc_info=True)
            raise Exception(f"OpenRouter API error with model {DESCRIPTION_MODEL}: {str(e)}")

    raise Exception(f"Failed to get response from OpenRouter after {max_retries} attempts")


@track_performance("google")
def _generate_description_google(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Google AI.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

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


@track_performance("openai")
def _generate_description_openai(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using OpenAI.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global openai_client

    if not AI_DESCRIPTION_API_KEY or AI_DESCRIPTION_API_KEY.lower() == "none":
        raise ValueError("AI_DESCRIPTION_API_KEY not set for OpenAI provider")

    try:
        if not openai_client:
            openai_client = openai.OpenAI(api_key=AI_DESCRIPTION_API_KEY)
            logger.debug("OpenAI client initialized for description generation")

        formatted_messages = [{"role": "user", "content": prompt}]

        completion_params = {
            "model": DESCRIPTION_MODEL,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens:
            completion_params["max_tokens"] = max_tokens

        response = openai_client.chat.completions.create(**completion_params)

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error with model {DESCRIPTION_MODEL}: {str(e)}")


@track_performance("anthropic")
def _generate_description_anthropic(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Anthropic Claude.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global anthropic_client

    if not AI_DESCRIPTION_API_KEY or AI_DESCRIPTION_API_KEY.lower() == "none":
        raise ValueError("API key not set for Anthropic provider")

    try:
        if not anthropic_client:
            anthropic_client = Anthropic(api_key=AI_DESCRIPTION_API_KEY)
            logger.debug("Anthropic client initialized for description generation")

        formatted_messages = [{"role": "user", "content": prompt}]

        if not max_tokens:
            max_tokens = 4096

        completion_params = {
            "model": DESCRIPTION_MODEL,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = anthropic_client.messages.create(**completion_params)

        return response.content[0].text
    except Exception as e:
        raise Exception(f"Anthropic API error with model {DESCRIPTION_MODEL}: {str(e)}")


@track_performance("groq")
def _generate_description_groq(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Groq.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

    Returns:
        str: Generated description.

    Raises:
        ValueError: If API key is not set or client is not initialized.
        Exception: If API call fails.
    """
    global groq_client

    if not AI_DESCRIPTION_API_KEY or AI_DESCRIPTION_API_KEY.lower() == "none":
        raise ValueError("API key not set for Groq provider")

    try:
        local_groq_client = Groq(api_key=AI_DESCRIPTION_API_KEY)
        logger.debug("Groq client initialized for description generation")

        formatted_messages = [{"role": "user", "content": prompt}]

        completion_params = {
            "model": DESCRIPTION_MODEL,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens:
            completion_params["max_tokens"] = max_tokens

        response = local_groq_client.chat.completions.create(**completion_params)

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Groq API error with model {DESCRIPTION_MODEL}: {str(e)}")


@track_performance("ollama")
def _generate_description_ollama(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a description using Ollama.

    Args:
        prompt (str): The prompt text.
        temperature (float): Temperature. Defaults to 0.3.
        max_tokens (Optional[int]): Max tokens. Defaults to None.

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
            "openai": {"requests": 0, "time": 0.0},
            "anthropic": {"requests": 0, "time": 0.0},
            "groq": {"requests": 0, "time": 0.0},
            "openrouter": {"requests": 0, "time": 0.0},
        },
    }


def log_chat(query: str, response: str, project_path: str, feedback: str = None) -> str:
    """Log chat interactions to a JSON file.

    Args:
        query (str): The user's query.
        response (str): The AI's response.
        project_path (str): The path to the indexed project.
        feedback (str, optional): User feedback on the response. Defaults to None.

    Returns:
        str: The ID of the chat entry (timestamp) for later reference.
    """
    if not CHAT_LOGS_ENABLED:
        return ""

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
            "id": timestamp,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "feedback": feedback,
            "feedback_timestamp": datetime.datetime.now().isoformat() if feedback else None
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
        return timestamp
    except Exception as e:
        logger.error(f"Error logging chat: {e}")
        return ""


def add_feedback(chat_id: str, feedback: str, project_path: str) -> bool:
    """Add feedback to a previously logged chat.

    Args:
        chat_id (str): The ID of the chat entry (timestamp).
        feedback (str): User feedback on the response.
        project_path (str): The path to the indexed project.

    Returns:
        bool: True if feedback was added successfully, False otherwise.
    """
    if not CHAT_LOGS_ENABLED:
        return False

    try:
        logs_dir = Path("chat_logs")
        if not logs_dir.exists():
            logger.error("Chat logs directory does not exist")
            return False

        project_name = Path(project_path).name
        project_logs_dir = logs_dir / project_name
        if not project_logs_dir.exists():
            logger.error(f"Project logs directory does not exist: {project_name}")
            return False

        log_files = list(project_logs_dir.glob("*.json"))
        if not log_files:
            logger.error(f"No log files found for project: {project_name}")
            return False

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                for chat in log_data.get("chats", []):
                    if chat.get("id") == chat_id or chat.get("timestamp") == chat_id:
                        chat["feedback"] = feedback
                        chat["feedback_timestamp"] = datetime.datetime.now().isoformat()

                        with open(log_file, "w", encoding="utf-8") as f:
                            json.dump(log_data, f, ensure_ascii=False, indent=2)

                        logger.info(f"Added feedback to chat {chat_id} in {log_file}")
                        return True
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
                continue

        logger.warning(f"Chat entry with ID {chat_id} not found in any log file")
        return False
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        return False
