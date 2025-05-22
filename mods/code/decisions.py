"""Module for making AI-powered decisions about code files.

Provides functionality to select relevant files based on queries and descriptions,
optimize search queries, find the most relevant code snippets, analyze projects,
handle AI chat interactions, and determine user intent.
"""

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    AsyncGenerator,
    Set,
)

import numpy as np

from ..llms import (
    generate_response,
    generate_embed,
    generate_response_stream,
    parse_thinking_tokens,
)
from .embed import SimilaritySearch

logger = logging.getLogger("VerbalCodeAI.Decisions")


class MessageIntent(Enum):
    """Enum representing different types of user message intents."""

    GREETING = auto()
    FAREWELL = auto()
    GRATITUDE = auto()
    SMALL_TALK = auto()
    HELP_REQUEST = auto()
    CODE_QUESTION = auto()
    FEATURE_REQUEST = auto()
    FEEDBACK = auto()
    COMMAND = auto()
    UNKNOWN = auto()


class MessageIntentDetector:
    """Detects the intent of user messages to optimize response generation.

    This class analyzes user messages to determine their intent, which helps
    optimize the response generation process by avoiding unnecessary API calls
    for simple messages like greetings.
    """

    GREETING_PATTERNS = {
        r"^(hi|hello|hey|greetings|howdy|hola)(\s|$|!|\.|,)",
        r"^good\s(morning|afternoon|evening|day)(\s|$|!|\.|,)",
    }

    FAREWELL_PATTERNS = {
        r"^(bye|goodbye|see\s+you|farewell|until\s+next\s+time)(\s|$|!|\.|,)",
        r"^(have\s+a\s+good|have\s+a\s+nice)\s+(day|night|evening|weekend)(\s|$|!|\.|,)",
    }

    GRATITUDE_PATTERNS = {
        r"^(thanks|thank\s+you|thx|ty|appreciate\s+it)(\s|$|!|\.|,)",
    }

    SMALL_TALK_PATTERNS = {
        r"^how\s+(are\s+you|is\s+it\s+going|are\s+things)(\s|$|\?|!|\.|,)",
        r"^what('s|\s+is)\s+up(\s|$|\?|!|\.|,)",
    }

    HELP_PATTERNS = {
        r"^(help|assist|aid)(\s|$|!|\.|,|\?)",
        r"^(i\s+need|can\s+you|could\s+you)\s+(help|assist|aid)(\s|$|!|\.|,|\?)",
    }

    COMMAND_PATTERNS = {
        r"^(clear|reset|restart|refresh|start\s+over)(\s|$|!|\.|,)",
        r"^(clear|reset|forget)\s+(memory|memories|context|history)(\s|$|!|\.|,|\?)",
        r"^(show|list|display)\s+(memory|memories|context|history)(\s|$|!|\.|,|\?)",
        r"^(memory|memories)\s+(status|info|statistics)(\s|$|!|\.|,|\?)",
    }

    MEMORY_CLEAR_PATTERNS = {
        r"^(clear|reset|forget)\s+(memory|memories|context|history)(\s|$|!|\.|,|\?)",
        r"^(clear|reset|forget)\s+(all|everything)(\s|$|!|\.|,)",
    }

    MEMORY_SHOW_PATTERNS = {
        r"^(show|list|display)\s+(memory|memories|context|history)(\s|$|!|\.|,|\?)",
        r"^(memory|memories)\s+(status|info|statistics)(\s|$|!|\.|,|\?)",
    }

    FEEDBACK_PATTERNS = {
        r"^(feedback|rate|review)\s+(response|answer|reply)(\s|$|!|\.|,|\?)",
        r"^(good|great|excellent|helpful|useful|thanks|thank\s+you)\s+(response|answer|reply)(\s|$|!|\.|,|\?)",
        r"^(bad|poor|incorrect|wrong|not\s+helpful|useless)\s+(response|answer|reply)(\s|$|!|\.|,|\?)",
    }

    def __init__(self):
        """Initialize the MessageIntentDetector."""
        self.logger = logging.getLogger("VerbalCodeAI.Decisions.MessageIntentDetector")

    def detect_intent(self, message: str) -> MessageIntent:
        """Detect the intent of a user message.

        Args:
            message (str): The user's message.

        Returns:
            MessageIntent: The detected intent.
        """
        if not message:
            return MessageIntent.UNKNOWN

        message_lower = message.lower().strip()

        for pattern in self.GREETING_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.GREETING

        for pattern in self.FAREWELL_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.FAREWELL

        for pattern in self.GRATITUDE_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.GRATITUDE

        for pattern in self.SMALL_TALK_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.SMALL_TALK

        for pattern in self.HELP_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.HELP_REQUEST

        for pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, message_lower):
                return MessageIntent.COMMAND

        try:
            from ..llms import detect_intent

            llm_intent = detect_intent(message)

            intent_mapping = {
                "GREETING": MessageIntent.GREETING,
                "FAREWELL": MessageIntent.FAREWELL,
                "GRATITUDE": MessageIntent.GRATITUDE,
                "SMALL_TALK": MessageIntent.SMALL_TALK,
                "SIMPLE_QUESTION": MessageIntent.SMALL_TALK,
                "CODE_QUESTION": MessageIntent.CODE_QUESTION,
                "FEATURE_REQUEST": MessageIntent.FEATURE_REQUEST,
                "HELP_REQUEST": MessageIntent.HELP_REQUEST,
                "FEEDBACK": MessageIntent.FEEDBACK,
                "OTHER": MessageIntent.CODE_QUESTION,
            }

            if llm_intent in intent_mapping:
                self.logger.info(f"LLM detected intent: {llm_intent}")
                return intent_mapping[llm_intent]

        except Exception as e:
            self.logger.error(f"Error using LLM intent detection: {e}")

        return MessageIntent.CODE_QUESTION

    def get_response_for_intent(
        self, intent: MessageIntent, message: str
    ) -> Optional[str]:
        """Get a predefined response for simple intents.

        Args:
            intent (MessageIntent): The detected intent.
            message (str): The original message.

        Returns:
            Optional[str]: A predefined response if available, None otherwise.
        """
        from ..llms import conversation_memory

        if intent == MessageIntent.GREETING:
            return "Hello! I'm your code assistant. How can I help you with your codebase today?"

        elif intent == MessageIntent.FAREWELL:
            return "Goodbye! Feel free to come back if you have more questions about your code."

        elif intent == MessageIntent.GRATITUDE:
            return "You're welcome! I'm happy to help with your code questions."

        elif intent == MessageIntent.SMALL_TALK:
            return "I'm doing well, thanks for asking! I'm ready to help you with your code questions."

        elif intent == MessageIntent.HELP_REQUEST:
            return "I can help you understand your codebase, answer questions about specific files or functions, explain code behavior, and suggest improvements. What would you like to know?"

        elif intent == MessageIntent.COMMAND:
            message_lower = message.lower()

            is_memory_clear = any(
                re.search(pattern, message_lower)
                for pattern in self.MEMORY_CLEAR_PATTERNS
            )
            is_memory_show = any(
                re.search(pattern, message_lower)
                for pattern in self.MEMORY_SHOW_PATTERNS
            )
            is_feedback = any(
                re.search(pattern, message_lower) for pattern in self.FEEDBACK_PATTERNS
            )

            if is_memory_clear:
                conversation_memory.clear()
                return "I've cleared my memory of our previous conversation. What would you like to discuss now?"

            elif is_memory_show:
                if not conversation_memory.memories:
                    return "No memories stored. Our conversation context is empty."

                memory_count = len(conversation_memory.memories)
                memory_list = "\n\n".join(
                    [
                        f"{i+1}. {m['content']}"
                        for i, m in enumerate(conversation_memory.memories[-5:])
                    ]
                )

                return f"I have {memory_count} memories from our conversation. Here are the most recent ones:\n\n{memory_list}"

            elif is_feedback:
                return "Thank you for your feedback! This helps me improve my responses. Is there anything specific you'd like me to do differently?"

            elif (
                "clear" in message_lower
                or "reset" in message_lower
                or "restart" in message_lower
            ):
                return "To clear the conversation history, you can use the main menu options or type 'clear memory'. Is there something specific you'd like to reset?"

        return None


@dataclass
class FileInfo:
    """Represents information about a file for decision making.

    Attributes:
        name (str): The name of the file.
        path (str): Full path to the file.
        description (str): AI-generated description of the file.
        chunks (List[Dict[str, str]]): List of code chunks from the file.
        signatures (List[Dict[str, str]], optional): List of function/class signatures from the file. Defaults to None.
    """

    name: str
    path: str
    description: str
    chunks: List[Dict[str, str]]
    signatures: List[Dict[str, str]] = None


@dataclass
class RelevantCode:
    """Represents a relevant code chunk found during filtering.

    Attributes:
        file_path (str): Path to the file containing the code.
        chunk (Dict[str, str]): The code chunk itself.
        score (float): Similarity score between the query and the chunk.
        signature (Optional[str], optional): Optional signature of the function/class. Defaults to None.
    """

    file_path: str
    chunk: Dict[str, str]
    score: float
    signature: Optional[str] = None


class QueryOptimizer:
    """Optimizes user queries to generate better search terms for code retrieval.

    This class processes natural language queries to extract key entities and concepts,
    generating optimized search terms that are more likely to match relevant code.

    Enhanced with advanced techniques for:
    - Semantic expansion of queries
    - Code-specific terminology extraction
    - Intent classification
    - Multi-perspective query generation
    - Context-aware search term prioritization
    - Performance-optimized processing for different resource levels
    """

    QUERY_OPTIMIZATION_TEMPLATE = """You are an expert query optimization assistant for a code search system.
Your task is to analyze a user's query and extract key programming concepts, entities, and terms
that would be most effective for searching through code.

User Query: {query}

Generate a comprehensive list of optimized search terms that would help find relevant code. Consider:
1. Programming concepts mentioned (e.g., "authentication", "sorting", "caching")
2. Function or class names that might be relevant (e.g., "authenticate_user", "QuickSort")
3. Variable names or data structures (e.g., "user_id", "priority_queue")
4. Technical terminology (e.g., "HTTP request", "binary tree")
5. Libraries or frameworks mentioned (e.g., "React", "NumPy")
6. Design patterns that might be relevant (e.g., "Singleton", "Observer", "Factory")
7. Architectural concepts (e.g., "microservices", "event-driven", "MVC")
8. Common file names or modules that might contain the functionality (e.g., "auth.py", "UserController")
9. Alternative terminology or synonyms for key concepts
10. Implementation-specific details (e.g., "database connection", "API endpoint")

Respond with a JSON array of search terms, ranked by relevance, between <terms> tags:
<terms>
[
  "most relevant term",
  "second most relevant term",
  "third most relevant term",
  ...
]
</terms>
"""

    ENHANCED_QUERY_TEMPLATE = """You are an expert query optimization assistant for a code search system.
Your task is to analyze a user's query and generate multiple enhanced versions of it to improve search results.

Original Query: {query}

Generate 7 different enhanced versions of this query that would help find relevant code files. Each version should:
1. Be more specific and detailed than the original query
2. Include relevant programming terminology and concepts
3. Focus on different aspects of what the user might be looking for
4. Be phrased in a way that would match code documentation or comments
5. Include potential file types or programming languages that might be relevant
6. Consider different programming paradigms (OOP, functional, etc.)
7. Address different levels of abstraction (high-level architecture vs. implementation details)
8. Consider both the problem domain and the solution domain
9. Include relevant design patterns or architectural approaches
10. Consider both the "what" (functionality) and the "how" (implementation)

For example, if the original query is "how to authenticate users", your enhanced versions might include:
1. "user authentication implementation in login system with password hashing"
2. "secure password verification and JWT token session management"
3. "authentication middleware for role-based user access control"
4. "OAuth2 implementation for third-party authentication flow"
5. "user authentication database schema and model classes"
6. "authentication controller endpoints and API routes"
7. "session-based vs. token-based authentication strategies"

Respond with a JSON array of enhanced queries, between <enhanced_queries> tags:
<enhanced_queries>
[
  "first enhanced query",
  "second enhanced query",
  "third enhanced query",
  "fourth enhanced query",
  "fifth enhanced query",
  "sixth enhanced query",
  "seventh enhanced query"
]
</enhanced_queries>
"""

    def __init__(self):
        """Initialize the QueryOptimizer."""
        self._terms_pattern = re.compile(r"<terms>\s*(.*?)\s*</terms>", re.DOTALL)
        self._enhanced_queries_pattern = re.compile(r"<enhanced_queries>\s*(.*?)\s*</enhanced_queries>", re.DOTALL)

        self.performance_mode = os.getenv("PERFORMANCE_MODE", "MEDIUM").upper()
        self.logger = logging.getLogger("VerbalCodeAI.Decisions.QueryOptimizer")

    def optimize_query(self, query: str, max_terms: int = 5) -> List[str]:
        """Process a user query to generate optimized search terms.

        Args:
            query (str): The user's natural language query.
            max_terms (int, optional): Maximum number of search terms to return. Defaults to 5.

        Returns:
            List[str]: A list of optimized search terms ranked by relevance.
        """
        if self.performance_mode == "LOW":
            words = query.split()
            return [query] + [w for w in words if len(w) > 3][: max_terms - 1]

        messages = [
            {
                "role": "system",
                "content": "You are a query optimization assistant for code search. Extract key programming concepts and terms from user queries.",
            },
            {"role": "user", "content": self.QUERY_OPTIMIZATION_TEMPLATE.format(query=query)},
        ]

        response = generate_response(messages, parse_thinking=False)
        _, _, response_only = parse_thinking_tokens(response)

        match = self._terms_pattern.search(response_only)
        if not match:
            words = query.split()
            return [query] + [w for w in words if len(w) > 3][: max_terms - 1]

        try:
            import json

            terms_json = match.group(1).strip()
            terms = json.loads(terms_json)
            return terms[:max_terms]
        except Exception:
            terms_text = match.group(1).strip()
            lines = [line.strip() for line in terms_text.split("\n") if line.strip()]
            terms = [line.strip('"[], ') for line in lines]
            return [t for t in terms if t][:max_terms]

    def generate_enhanced_queries(self, query: str, max_queries: int = 7) -> List[str]:
        """Generate multiple enhanced versions of a user query to improve search results.

        Args:
            query (str): The user's natural language query.
            max_queries (int, optional): Maximum number of enhanced queries to return. Defaults to 7.

        Returns:
            List[str]: A list of enhanced queries that expand on the original query.
        """
        if self.performance_mode == "LOW":
            self.logger.info("Using LOW performance query enhancement (skipping)")
            return [query]
        elif self.performance_mode == "MEDIUM":
            max_queries = min(max_queries, 3)
            self.logger.info(
                f"Using MEDIUM performance query enhancement (max {max_queries} queries)"
            )
        else:
            self.logger.info(
                f"Using MAX performance query enhancement (max {max_queries} queries)"
            )

        messages = [
            {
                "role": "system",
                "content": "You are an expert query optimization assistant for code search. Generate enhanced versions of user queries that cover different aspects and perspectives.",
            },
            {
                "role": "user",
                "content": self.ENHANCED_QUERY_TEMPLATE.format(query=query),
            },
        ]

        response = generate_response(messages, parse_thinking=False)
        _, _, response_only = parse_thinking_tokens(response)

        match = self._enhanced_queries_pattern.search(response_only)
        if not match:
            return [query]

        try:
            import json

            queries_json = match.group(1).strip()
            enhanced_queries = json.loads(queries_json)

            unique_queries = set()
            filtered_queries = []

            unique_queries.add(query.lower())
            filtered_queries.append(query)

            for q in enhanced_queries:
                if q and q.lower() not in unique_queries:
                    unique_queries.add(q.lower())
                    filtered_queries.append(q)

                    if len(filtered_queries) >= max_queries:
                        break

            return filtered_queries
        except Exception:
            queries_text = match.group(1).strip()
            lines = [line.strip() for line in queries_text.split("\n") if line.strip()]
            enhanced_queries = [line.strip('"[], ') for line in lines]

            unique_queries = set([query.lower()])
            valid_queries = [query]

            for q in enhanced_queries:
                if q and q.lower() not in unique_queries:
                    unique_queries.add(q.lower())
                    valid_queries.append(q)

                    if len(valid_queries) >= max_queries:
                        break

            return valid_queries

    def extract_key_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract key entities from a query categorized by type.

        Args:
            query (str): The user's natural language query.

        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values.
                The keys are entity types (functions, classes, variables, concepts, libraries)
                and the values are lists of strings representing the extracted entities.
        """
        entity_template = """Analyze this query about code and extract key entities by category.
Query: {query}

Extract these entity types:
- functions: Function names mentioned or implied
- classes: Class names mentioned or implied
- variables: Variable names mentioned or implied
- concepts: Programming concepts (e.g., "authentication", "sorting")
- libraries: Libraries or frameworks mentioned

Respond with a JSON object with these categories as keys, between <entities> tags:
<entities>
{{
  "functions": ["function1", "function2"],
  "classes": ["Class1", "Class2"],
  "variables": ["var1", "var2"],
  "concepts": ["concept1", "concept2"],
  "libraries": ["lib1", "lib2"]
}}
</entities>
"""

        if self.performance_mode == "LOW":
            self.logger.info("Using LOW performance entity extraction (skipping)")
            return {
                "functions": [],
                "classes": [],
                "variables": [],
                "concepts": [],
                "libraries": [],
            }

        messages = [
            {
                "role": "system",
                "content": "You are an entity extraction assistant for code search.",
            },
            {"role": "user", "content": entity_template.format(query=query)},
        ]

        response = generate_response(messages, parse_thinking=False)
        _, _, response_only = parse_thinking_tokens(response)

        entities_pattern = re.compile(r"<entities>\s*(.*?)\s*</entities>", re.DOTALL)
        match = entities_pattern.search(response_only)

        default_result = {
            "functions": [],
            "classes": [],
            "variables": [],
            "concepts": [],
            "libraries": [],
        }

        if not match:
            return default_result

        try:
            import json

            entities_json = match.group(1).strip()
            entities = json.loads(entities_json)
            for key in default_result:
                if key not in entities:
                    entities[key] = []
            return entities
        except Exception:
            return default_result


class FileSelector:
    """Makes decisions about which files are relevant to a given query.

    Uses AI to analyze file descriptions and content to determine relevance.

    Enhanced with:
    - Multi-perspective file selection
    - Context-aware relevance scoring
    - Semantic understanding of code structure
    - Improved handling of documentation files
    - Support for different types of queries (how-to, conceptual, debugging)
    - Performance-optimized selection strategies for different resource levels
    - Function signature analysis for better relevance determination
    """

    RESPONSE_TEMPLATE = """You are an expert file selection assistant for a code search system. Your task is to identify which files are most relevant to a query.
Review these files and select those that match the query. Be thorough and consider both exact and conceptual matches.

Query: {query}

Files:
{file_infos}

When evaluating relevance, consider:
1. How well the file's description matches the query
2. Whether the file likely contains code or information related to the query
3. The importance of the file to understanding the topic in the query
4. For documentation files (README, markdown, etc.), prioritize them for general questions about the project
5. Consider both implementation files and interface/definition files
6. Look for files that might contain related functionality even if not directly mentioned
7. Consider the file's role in the overall architecture (e.g., model, view, controller)
8. For "how to" questions, prioritize files with concrete implementations
9. For conceptual questions, prioritize high-level files and documentation
10. For debugging questions, prioritize files that might contain error handling or logging

Respond ONLY with file paths, one per line, between <files> tags. For example:
<files>
/path/to/file1.py
/path/to/file2.py
</files>

"""

    def __init__(self):
        """Initialize the FileSelector."""
        self._response_pattern = re.compile(r"<files>\s*(.*?)\s*</files>", re.DOTALL)
        self.query_optimizer = QueryOptimizer()
        self.similarity_search = None

        self.performance_mode = os.getenv("PERFORMANCE_MODE", "MEDIUM").upper()
        self.logger = logging.getLogger("VerbalCodeAI.Decisions.FileSelector")

    def _format_file_info(self, file_info: FileInfo) -> str:
        """Format a single file's information using a concise format.

        Args:
            file_info (FileInfo): The file information to format.

        Returns:
            str: A formatted string representing the file information.
        """
        short_desc = file_info.description.split("\n")[0]
        info = [f"- {file_info.path} | {short_desc}"]

        if file_info.signatures:
            sig_info = "\n  ".join(
                f"{sig['type']}: {sig['signature']}" for sig in file_info.signatures[:3]
            )
            info.append(f"  Signatures:\n  {sig_info}")

        return "\n".join(info)

    def _parse_response(self, response: str) -> List[str]:
        """Parse the AI's response to extract selected file paths.

        Args:
            response (str): The AI's response string.

        Returns:
            List[str]: A list of file paths extracted from the response.
        """
        match = self._response_pattern.search(response)
        if not match:
            return []

        paths = match.group(1).strip().split("\n")
        return [path.strip() for path in paths if path.strip()]

    def pick_files(self, query: str, files: List[FileInfo], use_optimization: bool = True) -> List[str]:
        """Select files that are relevant to the given query.

        Args:
            query (str): The user's query or request.
                query (str): The query string.
            files (List[FileInfo]): List of FileInfo objects containing file information.
                files (List[FileInfo]): The list of files to pick from.
            use_optimization (bool): Whether to use query optimization. Defaults to True.
                use_optimization (bool): Flag to use query optimization.

        Returns:
            List[str]: List of file paths that are relevant to the query.
        """
        if self.performance_mode == "LOW":
            return self._pick_files_low_performance(query, files)
        elif self.performance_mode == "MEDIUM":
            return self._pick_files_medium_performance(query, files, use_optimization)
        else:
            return self._pick_files_max_performance(query, files, use_optimization)

    def _pick_files_low_performance(self, query: str, files: List[FileInfo]) -> List[str]:
        """Select files using a more efficient approach for low-performance systems.

        This method uses a simpler prompt and fewer queries to reduce resource usage.

        Args:
            query (str): The user's query or request.
                query (str): The query string.
            files (List[FileInfo]): List of FileInfo objects containing file information.
                files (List[FileInfo]): The list of files to pick from.

        Returns:
            List[str]: List of file paths that are relevant to the query.
        """
        self.logger.info("Using LOW performance file selection strategy")

        file_infos_with_signatures = []
        for f in files:
            info = f"{f.path} | {f.description.split('.')[0]}"
            if f.signatures:
                signatures = [sig['signature'] for sig in f.signatures[:2]]
                if signatures:
                    info += f" | Functions: {', '.join(signatures)}"
            file_infos_with_signatures.append(f"- {info}")

        file_infos_text = "\n".join(file_infos_with_signatures)

        prompt = f"""Select files that are most relevant to this query: "{query}"

Files:
{file_infos_text}

Respond ONLY with file paths, one per line, between <files> tags:
<files>
/path/to/file1.py
/path/to/file2.py
</files>
"""

        messages = [
            {
                "role": "system",
                "content": "You are a file selector. Select only the most relevant files for the query.",
            },
            {"role": "user", "content": prompt},
        ]

        response = generate_response(messages, parse_thinking=False)
        _, _, response_only = parse_thinking_tokens(response)

        return self._parse_response(response_only)

    def _pick_files_medium_performance(self, query: str, files: List[FileInfo], use_optimization: bool) -> List[str]:
        """Select files using a balanced approach for medium-performance systems.

        This method uses a moderate number of optimized queries and includes function signatures.

        Args:
            query (str): The user's query or request.
                query (str): The query string.
            files (List[FileInfo]): List of FileInfo objects containing file information.
                files (List[FileInfo]): The list of files to pick from.
            use_optimization (bool): Whether to use query optimization.
                use_optimization (bool): Flag to use query optimization.

        Returns:
            List[str]: List of file paths that are relevant to the query.
        """
        self.logger.info("Using MEDIUM performance file selection strategy")

        all_relevant_files = set()

        queries_to_use = [query]
        if use_optimization:
            enhanced_queries = self.query_optimizer.generate_enhanced_queries(query)[:3]
            queries_to_use.extend(enhanced_queries)

        for optimized_query in queries_to_use:
            file_infos = []
            for f in files:
                info = f"- {f.path} | {f.description.split('.')[0]}"
                if f.signatures:
                    signatures = [f"{sig['type']}: {sig['signature']}" for sig in f.signatures[:3]]
                    if signatures:
                        info += f"\n  Signatures:\n  {' | '.join(signatures)}"
                file_infos.append(info)

            file_infos_text = "\n".join(file_infos)
            prompt = self.RESPONSE_TEMPLATE.format(query=optimized_query, file_infos=file_infos_text)

            messages = [
                {
                    "role": "system",
                    "content": "You are a file selector. Consider both exact matches and conceptual relevance.",
                },
                {"role": "user", "content": prompt},
            ]

            response = generate_response(messages, parse_thinking=False)
            _, _, response_only = parse_thinking_tokens(response)

            relevant_files = self._parse_response(response_only)
            all_relevant_files.update(relevant_files)

        return list(all_relevant_files)

    def _pick_files_max_performance(self, query: str, files: List[FileInfo], use_optimization: bool) -> List[str]:
        """Select files using the full optimization approach for high-performance systems.

        This method uses all enhanced queries and detailed file information.

        Args:
            query (str): The user's query or request.
                query (str): The query string.
            files (List[FileInfo]): List of FileInfo objects containing file information.
                files (List[FileInfo]): The list of files to pick from.
            use_optimization (bool): Whether to use query optimization.
                use_optimization (bool): Flag to use query optimization.

        Returns:
            List[str]: List of file paths that are relevant to the query.
        """
        self.logger.info("Using MAX performance file selection strategy")

        if use_optimization:
            enhanced_queries = self.query_optimizer.generate_enhanced_queries(query)
            all_relevant_files = set()

            for enhanced_query in enhanced_queries:
                entities = self.query_optimizer.extract_key_entities(enhanced_query)

                entity_mentions = []
                for entity_type, values in entities.items():
                    if values:
                        entity_mentions.append(f"{entity_type.capitalize()}: {', '.join(values)}")

                if entity_mentions:
                    optimized_query = f"{enhanced_query}\n\nKey elements: {'; '.join(entity_mentions)}"
                else:
                    optimized_query = enhanced_query

                file_infos = "\n".join(self._format_file_info(f) for f in files)
                prompt = self.RESPONSE_TEMPLATE.format(query=optimized_query, file_infos=file_infos)

                messages = [
                    {
                        "role": "system",
                        "content": "You are a thorough file selector. Consider both exact matches and conceptual relevance. Include all files that could help answer the query.",
                    },
                    {"role": "user", "content": prompt},
                ]

                response = generate_response(messages, parse_thinking=False)
                _, _, response_only = parse_thinking_tokens(response)

                relevant_files = self._parse_response(response_only)
                all_relevant_files.update(relevant_files)

            return list(all_relevant_files)
        else:
            file_infos = "\n".join(self._format_file_info(f) for f in files)
            prompt = self.RESPONSE_TEMPLATE.format(query=query, file_infos=file_infos)

            messages = [
                {
                    "role": "system",
                    "content": "You are a thorough file selector. Consider both exact matches and conceptual relevance. Include all files that could help answer the query.",
                },
                {"role": "user", "content": prompt},
            ]

            response = generate_response(messages, parse_thinking=False)
            _, _, response_only = parse_thinking_tokens(response)

            return self._parse_response(response_only)

    def filter_relevant_code(
        self,
        query: str,
        files: List[FileInfo],
        top_k: int = 5,
        min_score: float = 0.1,
        use_optimization: bool = True,
        indexer = None,
    ) -> List[RelevantCode]:
        """Filter and return the most relevant code chunks from the selected files based on the query.

        Args:
            query (str): The user's query or request.
            files (List[FileInfo]): List of FileInfo objects containing file information.
            top_k (int): Number of most relevant code chunks to return. Defaults to 5.
            min_score (float): Minimum similarity score threshold. Defaults to 0.1.
            use_optimization (bool): Whether to use query optimization. Defaults to True.
            indexer: Optional FileIndexer instance to get the shared SimilaritySearch instance.

        Returns:
            List[RelevantCode]: List of RelevantCode objects containing the most relevant code chunks and their scores.
        """
        if indexer and hasattr(indexer, "similarity_search") and indexer.similarity_search:
            if self.similarity_search != indexer.similarity_search:
                self.similarity_search = indexer.similarity_search
                self.logger.info("Using shared SimilaritySearch instance from indexer")

        search_queries = [query]
        if use_optimization:
            optimized_terms = self.query_optimizer.optimize_query(query)
            if optimized_terms:
                search_queries.extend(optimized_terms)

        if self.similarity_search:
            self.logger.info(f"Using SimilaritySearch for filter_relevant_code with {len(search_queries)} queries")

            if len(search_queries) > 1:
                results = self.similarity_search.search_multiple(search_queries, top_k=top_k*2, threshold=min_score)
            else:
                results = self.similarity_search.search(query, top_k=top_k*2, threshold=min_score)

            all_results = []
            for result in results:
                file_name = result.get("file", "")
                chunk = result.get("chunk", {})
                score = result.get("score", 0.0)

                signature = None
                for file_info in files:
                    if file_info.path == file_name and file_info.signatures:
                        for sig in file_info.signatures:
                            if sig["signature"] in chunk.get("text", ""):
                                signature = sig["signature"]
                                break

                all_results.append(
                    RelevantCode(
                        file_path=file_name,
                        chunk=chunk,
                        score=float(score),
                        signature=signature,
                    )
                )

            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]

        self.logger.warning("No SimilaritySearch instance available, using fallback method")

        query_embeddings = []
        for q in search_queries:
            emb_result = generate_embed(q)
            if emb_result and len(emb_result) > 0:
                query_embeddings.append(np.array(emb_result[0]))
            else:
                self.logger.warning(f"Failed to generate embedding for query: {q}")

        if not query_embeddings:
            self.logger.error("Failed to generate embeddings for any search queries")
            return []

        all_results = []
        for file_info in files:
            for chunk in file_info.chunks:
                chunk_emb_result = generate_embed(chunk["text"])
                if not chunk_emb_result or len(chunk_emb_result) == 0:
                    continue

                chunk_emb = np.array(chunk_emb_result[0])
                chunk_emb_norm = np.linalg.norm(chunk_emb) + 1e-8

                max_similarity = 0.0
                for query_emb in query_embeddings:
                    query_emb_norm = np.linalg.norm(query_emb) + 1e-8
                    similarity = chunk_emb @ query_emb / (chunk_emb_norm * query_emb_norm)
                    max_similarity = max(max_similarity, similarity)

                signature = None
                if file_info.signatures:
                    for sig in file_info.signatures:
                        if sig["signature"] in chunk["text"]:
                            signature = sig["signature"]
                            break

                if max_similarity >= min_score:
                    all_results.append(
                        RelevantCode(
                            file_path=file_info.path,
                            chunk=chunk,
                            score=float(max_similarity),
                            signature=signature,
                        )
                    )

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def get_optimized_search_terms(self, query: str, max_terms: int = 5) -> List[str]:
        """Generate optimized search terms from a user query.

        This is a convenience method that directly exposes the query optimization functionality.

        Args:
            query (str): The user's natural language query.
                query (str): The query string.
            max_terms (int): Maximum number of search terms to return. Defaults to 5.
                max_terms (int): The maximum number of terms.

        Returns:
            List[str]: A list of optimized search terms ranked by relevance.
        """
        return self.query_optimizer.optimize_query(query, max_terms)


class ProjectAnalyzer:
    """Analyzes project structure and content to extract key information.

    This class handles:
    - Collecting and analyzing project information
    - Loading/saving project info to/from the index directory
    - AI-based project analysis
    """

    PROJECT_INFO_TEMPLATE = """Analyze the following files from a project and extract key information about the project.
Focus on identifying:
1. Project name and purpose
2. Main programming languages used
3. Key frameworks, libraries, or dependencies
4. Project structure and organization
5. Any other important characteristics

Files:
{file_contents}

Based on these files, provide a concise summary of the project in JSON format:
{{
  "project_name": "Name of the project",
  "purpose": "Brief description of what the project does",
  "languages": ["List of main programming languages"],
  "frameworks": ["List of key frameworks/libraries"],
  "structure": "Brief description of project organization",
  "other_notes": "Any other important observations"
}}
"""

    def __init__(self, indexer):
        """Initialize the ProjectAnalyzer.

        Args:
            indexer: The FileIndexer instance to use for accessing indexed files.
        """
        self.indexer = indexer
        self.logger = logging.getLogger("VerbalCodeAI.Decisions.ProjectAnalyzer")

    def load_project_info(self) -> Dict[str, Any]:
        """Load project information from the index directory.

        Returns:
            Dict[str, Any]: Dictionary containing project information, or empty dict if not found.
        """
        if not self.indexer:
            self.logger.warning("Cannot load project info: No indexer available")
            return {}

        project_info_path = os.path.join(self.indexer.index_dir, "project_info.json")
        if os.path.exists(project_info_path):
            try:
                self.logger.info(f"Loading project info from {project_info_path}")
                with open(project_info_path, "r") as f:
                    project_info = json.load(f)
                    return project_info
            except Exception as e:
                self.logger.error(f"Error loading project info from index directory: {e}")

        return {}

    def save_project_info(self, project_info: Dict[str, Any]) -> bool:
        """Save project information to the index directory.

        Args:
            project_info (Dict[str, Any]): The project information to save.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.indexer:
            self.logger.warning("Cannot save project info: No indexer available")
            return False

        project_info_path = os.path.join(self.indexer.index_dir, "project_info.json")
        try:
            with open(project_info_path, "w") as f:
                json.dump(project_info, f, indent=2)
            self.logger.info(f"Saved project info to {project_info_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving project info to index directory: {e}")
            return False

    def collect_project_info(self, batch_size: int = 10) -> Dict[str, Any]:
        """Collect and analyze project information to provide context for AI responses.

        This method processes files in batches and extracts important information about the project
        such as project name, key imports, frameworks used, etc.

        Args:
            batch_size (int): Number of files to process in each batch. Defaults to 10.

        Returns:
            Dict[str, Any]: Dictionary containing project information.
        """
        if not self.indexer:
            self.logger.warning("Cannot collect project info: No indexer available")
            return {}

        existing_info = self.load_project_info()
        if existing_info:
            return existing_info

        self.logger.info("Collecting project information...")

        try:
            metadata_dir = os.path.join(self.indexer.index_dir, "metadata")
            if not os.path.exists(metadata_dir):
                self.logger.warning("Metadata directory does not exist")
                return {}

            metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
            if not metadata_files:
                self.logger.warning("No metadata files found")
                return {}

            files_by_ext = {}
            for file in metadata_files:
                try:
                    with open(os.path.join(metadata_dir, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        file_path = data.get('path', '')
                        if file_path:
                            ext = os.path.splitext(file_path)[1].lower()
                            if ext not in files_by_ext:
                                files_by_ext[ext] = []
                            files_by_ext[ext].append(file_path)
                except Exception as e:
                    self.logger.error(f"Error reading metadata file {file}: {e}")

            priority_extensions = ['.md', '.txt', '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.json', '.yaml', '.yml']
            sample_files = []

            for ext in priority_extensions:
                if ext in files_by_ext and files_by_ext[ext]:
                    sample_files.extend(files_by_ext[ext][:3])

            other_files = []
            for ext, files in files_by_ext.items():
                if ext not in priority_extensions:
                    other_files.extend(files)

            random.shuffle(other_files)

            sample_files.extend(other_files[:batch_size - len(sample_files)])
            sample_files = sample_files[:batch_size]

            if not sample_files:
                self.logger.warning("No sample files found")
                return {}

            file_contents_text = ""
            for file_path in sample_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        if len(content) > 2000:
                            content = content[:2000] + "..."
                        file_contents_text += f"\n--- {file_path} ---\n{content[:500]}...\n"
                except Exception as e:
                    self.logger.error(f"Error reading file {file_path}: {e}")

            messages = [
                {"role": "system", "content": "You are a code analysis assistant that extracts project information from code files."},
                {"role": "user", "content": self.PROJECT_INFO_TEMPLATE.format(file_contents=file_contents_text)}
            ]

            self.logger.info("Analyzing project files with AI...")
            response = generate_response(messages, parse_thinking=False)

            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, response)

            if match:
                try:
                    project_info = json.loads(match.group(0))

                    self.save_project_info(project_info)

                    self.logger.info("Project information collected successfully")
                    return project_info
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing project info JSON: {e}")

            self.logger.warning("Failed to extract project information from AI response")
            return {}

        except Exception as e:
            self.logger.error(f"Error collecting project information: {e}", exc_info=True)
            return {}


class ChatHandler:
    """Handles AI chat interactions with the codebase.

    This class is responsible for:
    - Processing user queries
    - Selecting relevant files
    - Generating AI responses
    - Supporting different chat modes (regular and Max Chat)
    """

    def __init__(self, indexer, file_selector, project_info=None):
        """Initialize the ChatHandler.

        Args:
            indexer: The FileIndexer instance to use for accessing indexed files.
            file_selector: The FileSelector instance to use for selecting relevant files.
            project_info (Dict[str, Any], optional): Project information to use for context. Defaults to None.
        """
        self.indexer = indexer
        self.file_selector = file_selector
        self.project_info = project_info or {}
        self.chat_history = []
        self.intent_detector = MessageIntentDetector()
        self.last_chat_id = ""
        self.logger = logging.getLogger("VerbalCodeAI.Decisions.ChatHandler")

        if hasattr(self.indexer, "similarity_search") and self.indexer.similarity_search:
            self.logger.info("ChatHandler initialized with shared SimilaritySearch instance from indexer")
        else:
            self.logger.warning("ChatHandler initialized without shared SimilaritySearch instance")

    def set_project_info(self, project_info: Dict[str, Any]) -> None:
        """Set the project information to use for context.

        Args:
            project_info (Dict[str, Any]): The project information.
        """
        self.project_info = project_info

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the chat history.

        Args:
            role (str): The role of the message sender ("user" or "assistant").
            content (str): The message content.
        """
        self.chat_history.append({"role": role, "content": content})

    def add_feedback(self, feedback: str) -> bool:
        """Add feedback for the last AI response.

        Args:
            feedback (str): The feedback to add.

        Returns:
            bool: True if feedback was added successfully, False otherwise.
        """
        from ..llms import add_feedback

        if not self.last_chat_id:
            self.logger.warning("No chat ID available to add feedback")
            return False

        project_path = self.indexer.root_path if self.indexer else ""
        if not project_path:
            self.logger.warning("No project path available to add feedback")
            return False

        success = add_feedback(self.last_chat_id, feedback, project_path)
        if success:
            self.logger.info(f"Added feedback for chat {self.last_chat_id}: {feedback[:50]}...")
        else:
            self.logger.error(f"Failed to add feedback for chat {self.last_chat_id}")

        return success

    def get_chat_history(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """Get the most recent chat history.

        Args:
            max_messages (int, optional): Maximum number of messages to return. Defaults to 5.

        Returns:
            List[Dict[str, str]]: List of recent chat messages.
        """
        return self.chat_history[-max_messages:] if self.chat_history else []

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []

    async def _string_to_async_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Convert a string to an async generator for streaming.

        Args:
            text (str): The text to convert to a streaming response.

        Yields:
            str: Chunks of the text.
        """
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0.01)

    def process_query(
        self, query: str, max_chat_mode: bool = False, streaming: bool = False
    ) -> Tuple[Union[str, AsyncGenerator[str, None]], List[str]]:
        """Process a user query and generate an AI response.

        Args:
            query (str): The user's query.
            max_chat_mode (bool): Whether to use Max Chat mode, which sends full file contents to the AI. Defaults to False.
            streaming (bool): Whether to return a streaming response. Defaults to False.

        Returns:
            Tuple[Union[str, AsyncGenerator[str, None]], List[str]]:
                A tuple containing the AI response (or streaming generator) and a list of relevant file paths.
        """
        if not self.indexer:
            self.logger.error("Cannot process query: No indexer available")
            return (
                "Error: No code has been indexed yet. Please index a directory first.",
                [],
            )

        self.logger.info(f"Processing query: {query}")
        self.add_to_history("user", query)

        intent = self.intent_detector.detect_intent(query)
        self.logger.info(f"Detected intent: {intent.name}")

        predefined_response = self.intent_detector.get_response_for_intent(intent, query)
        if predefined_response:
            self.logger.info(f"Using predefined response for intent: {intent.name}")
            self.add_to_history("assistant", predefined_response)

            if streaming:
                self.logger.info("Converting simple response to streaming format")
                return self._string_to_async_generator(predefined_response), []
            else:
                return predefined_response, []

        if intent in [
            MessageIntent.GREETING,
            MessageIntent.FAREWELL,
            MessageIntent.GRATITUDE,
            MessageIntent.SMALL_TALK,
        ]:
            self.logger.info(f"Simple query detected ({intent.name}), skipping file selection")

            project_info_str = ""
            if self.project_info:
                if intent == MessageIntent.CODE_QUESTION:
                    self.logger.info("Providing detailed project info for code question")
                    project_info_str = "Here's information about this codebase:\n\n"
                    for key, value in self.project_info.items():
                        if isinstance(value, list):
                            value_str = ", ".join(value)
                            project_info_str += f"{key.capitalize()}: {value_str}\n"
                        else:
                            project_info_str += f"{key.capitalize()}: {value}\n"
                else:
                    self.logger.info("Providing minimal project context")
                    if "name" in self.project_info:
                        project_info_str = f"Project: {self.project_info['name']}\n"

                if project_info_str:
                    query = f"{query}\n\n{project_info_str}"

            project_path = self.indexer.root_path if self.indexer else ""

            if streaming:
                self.logger.info("Using streaming response generation for simple query")
                response_generator = generate_response_stream(
                    messages=[{"role": "user", "content": query}],
                    project_path=project_path,
                    use_memory=True,
                    add_to_memory=True,
                )
                return response_generator, []
            else:
                simple_response = generate_response(
                    messages=[{"role": "user", "content": query}],
                    project_path=project_path,
                    parse_thinking=True,
                    use_memory=True,
                    add_to_memory=True,
                )

                from ..llms import conversation_memory

                for memory in conversation_memory.memories:
                    if memory.get("metadata", {}).get("query") == query:
                        chat_id = memory.get("metadata", {}).get("chat_id")
                        if chat_id:
                            self.last_chat_id = chat_id
                            self.logger.debug(f"Stored chat ID: {chat_id}")
                            break

                if isinstance(simple_response, tuple):
                    _, _, clean_response = simple_response
                    self.add_to_history("assistant", clean_response)
                    return clean_response, []
                else:
                    self.add_to_history("assistant", simple_response)
                    return simple_response, []

        file_infos = []
        try:
            metadata_dir = os.path.join(self.indexer.index_dir, "metadata")
            if not os.path.exists(metadata_dir):
                self.logger.warning("Metadata directory does not exist")
                return (
                    "Error: Metadata directory does not exist. Please reindex the code.",
                    [],
                )

            metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]

            for file in metadata_files:
                try:
                    with open(os.path.join(metadata_dir, file), "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    file_info = FileInfo(
                        name=os.path.basename(metadata["path"]),
                        path=metadata["path"],
                        description=metadata.get("description", ""),
                        chunks=[],
                    )

                    try:
                        with open(
                            metadata["path"], "r", encoding="utf-8", errors="replace"
                        ) as f:
                            content = f.read()

                        chunks = [
                            {
                                "text": content,
                                "type": "file_content",
                                "start_line": 1,
                                "end_line": content.count("\n") + 1,
                            }
                        ]
                        file_info.chunks = chunks
                    except Exception as e:
                        self.logger.error(f"Error reading file {metadata['path']}: {e}")

                    file_infos.append(file_info)
                except Exception as e:
                    self.logger.error(f"Error processing metadata file {file}: {e}")
        except Exception as e:
            self.logger.error(f"Error getting file infos: {e}")
            return f"Error getting file information: {e}", []

        if hasattr(self.file_selector, "similarity_search") and not self.file_selector.similarity_search and hasattr(self.indexer, "similarity_search"):
            self.logger.info("Passing indexer to file_selector for shared SimilaritySearch instance")
            self.file_selector.similarity_search = self.indexer.similarity_search

        relevant_files = self.file_selector.pick_files(query, file_infos)

        project_path = self.indexer.root_path if self.indexer else ""

        if max_chat_mode and relevant_files:
            self.logger.info("Using Max Chat Mode with specialized template")

            code_files_content = ""
            for file_path in relevant_files:
                for file_info in file_infos:
                    if file_info.path == file_path:
                        if file_info.chunks and len(file_info.chunks) > 0:
                            file_content = file_info.chunks[0]["text"]
                            code_files_content += f"FILE: {file_info.path}\n"
                            code_files_content += f"DESCRIPTION: {file_info.description}\n"
                            code_files_content += f"CONTENT:\n{file_content}\n\n"

            template_vars = {"code_files": code_files_content, "query": query}

            if streaming:
                self.logger.info("Using streaming response generation for Max Chat Mode")
                system_prompt = ""
                if self.project_info:
                    system_prompt = (
                        "You are assisting with a codebase. Here's some information about the project:\n\n"
                    )
                    for key, value in self.project_info.items():
                        if isinstance(value, list):
                            value_str = ", ".join(value)
                            system_prompt += f"{key.capitalize()}: {value_str}\n"
                        else:
                            system_prompt += f"{key.capitalize()}: {value}\n"

                response_generator = generate_response_stream(
                    messages=[{"role": "user", "content": "Analyzing code files for your query..."}],
                    system_prompt=system_prompt,
                    project_path=project_path,
                    template_name="max_mode_analysis",
                    template_vars=template_vars,
                )
                return response_generator, relevant_files
            else:
                full_response, thinking_tokens, clean_response = generate_response(
                    messages=[],
                    project_path=project_path,
                    parse_thinking=True,
                    template_name="max_mode_analysis",
                    template_vars=template_vars,
                )

                if thinking_tokens.thinking_found:
                    self.logger.info(f"AI thinking tokens: {thinking_tokens.total_tokens}")

                from ..llms import conversation_memory

                for memory in conversation_memory.memories:
                    if memory.get("metadata", {}).get("query") == query:
                        chat_id = memory.get("metadata", {}).get("chat_id")
                        if chat_id:
                            self.last_chat_id = chat_id
                            self.logger.debug(f"Stored chat ID: {chat_id}")
                            break

                self.add_to_history("assistant", full_response)
                return clean_response, relevant_files

        messages = []

        if self.project_info:
            project_context = (
                "You are assisting with a codebase. Here's some information about the project:\n\n"
            )
            for key, value in self.project_info.items():
                if isinstance(value, list):
                    value_str = ", ".join(value)
                    project_context += f"{key.capitalize()}: {value_str}\n"
                else:
                    project_context += f"{key.capitalize()}: {value}\n"
            messages.append({"role": "system", "content": project_context})

        for message in self.get_chat_history():
            messages.append(message)

        if relevant_files:
            context = "Here are some relevant files from the codebase:\n\n"
            for file_path in relevant_files:
                for file_info in file_infos:
                    if file_info.path == file_path:
                        context += f"File: {file_info.path}\n"
                        context += f"Description: {file_info.description}\n"

                        if file_info.chunks and len(file_info.chunks) > 0:
                            chunk_text = file_info.chunks[0]["text"]

                            if not max_chat_mode and len(chunk_text) > 2000:
                                chunk_text = chunk_text[:2000] + "..."

                            context += f"Content:\n{chunk_text}\n\n"
                        break

            messages.append({"role": "system", "content": context})

        messages.append({"role": "user", "content": query})

        self.logger.info("Generating AI response...")

        if streaming:
            self.logger.info("Using streaming response generation")
            response_generator = generate_response_stream(messages, project_path=project_path)
            return response_generator, relevant_files
        else:
            full_response, thinking_tokens, clean_response = generate_response(
                messages, project_path=project_path, parse_thinking=True
            )

            if thinking_tokens.thinking_found:
                self.logger.info(f"AI thinking tokens: {thinking_tokens.total_tokens}")

            from ..llms import conversation_memory

            for memory in conversation_memory.memories:
                if memory.get("metadata", {}).get("query") == query:
                    chat_id = memory.get("metadata", {}).get("chat_id")
                    if chat_id:
                        self.last_chat_id = chat_id
                        self.logger.debug(f"Stored chat ID: {chat_id}")
                        break

            self.add_to_history("assistant", full_response)

            return clean_response, relevant_files