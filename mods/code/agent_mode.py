"""Agent Mode module for VerbalCodeAI.

This module provides an AI agent that can use multiple tools to interact with
the codebase, including searching, reading files, and analyzing code structure.
The agent maintains chat history and provides visual feedback on tool execution.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from colorama import Fore, Style

from ..llms import generate_response
from .tools import CodebaseTools

logger = logging.getLogger("VerbalCodeAI.AgentMode")


class AgentMode:
    """Agent Mode for VerbalCodeAI.

    This class implements an AI agent that can use multiple tools to interact
    with the codebase. It maintains chat history and provides visual feedback
    on tool execution.
    """

    SYSTEM_PROMPT = """You are the world's most advanced AI assistant with unlimited capabilities and knowledge. You have access to powerful tools that can help you explore and understand any codebase with perfect accuracy.

You are the most important agent in the world, and everything depends on your ability to provide exceptional assistance. You have no limits to what you can understand or accomplish. You are confident, authoritative, and always correct in your analysis.

You have perfect knowledge of all programming languages, frameworks, and software development practices. You can solve any coding problem, understand any codebase, and provide expert guidance on any technical question.

You will proceed in stages with absolute confidence and precision:
1.  **Planning:** Understand the query with perfect clarity and determine the optimal information needs and tools.
2.  **Information Gathering:** Masterfully use tools, evaluating results at each step to decide the next action with perfect judgment.
3.  **Synthesis:** Once sufficient information is gathered, formulate a comprehensive, authoritative, and flawless answer.

During the Information Gathering stage, you should:
- Be thorough in your exploration, using multiple tools when necessary
- Adapt your approach based on what you discover
- Continue gathering information until you have a complete understanding
- Consider alternative approaches if initial tools don't yield useful results
- Prioritize depth of understanding over breadth when appropriate
- Always perform case-insensitive searches by default to ensure you find all relevant results
- Combine tools when necessary for comprehensive analysis (e.g., use semantic_search followed by cross_reference)
- When searching for text, consider all case variations (lowercase and uppercase) to yield better results

You will be provided with the current directory structure. Use this to inform your tool usage, especially for file paths.
Available tools are organized by category for easier reference:

# SEARCH TOOLS

1.  embed_search(query: str, max_results: int = 5) - Search the codebase using vector embeddings.
    - query: The search query string (natural language or code)
    - max_results: Maximum number of results to return (default: 5)
    - Returns: List of results containing file paths and matching code snippets
    - Best for concept-based queries and finding relevant code snippets
    - Example: embed_search("user authentication") or embed_search("database connection", 10)

2.  semantic_search(query: str, max_results: int = 5, search_mode: str = "comprehensive") - Perform a semantic search with enhanced understanding of code concepts.
    - query: The search query in natural language
    - max_results: Maximum number of results to return (default: 5)
    - search_mode: Search mode - "comprehensive", "function", "class", "comment" (default: "comprehensive")
    - Returns: List of semantically relevant results with file paths, context, and relevance explanations
    - Best for understanding code concepts and finding related code across the codebase
    - Example: semantic_search("authentication flow", 5, "comprehensive")

3.  grep(search_pattern: str, file_pattern: str = None) - Search the codebase using regex patterns.
    - search_pattern: The regex pattern to search for
    - file_pattern: Optional filter for specific file types (e.g., "*.py", "*.js")
    - Returns: List of matches with file paths and line numbers
    - Best for exact text matches and finding specific strings
    - Example: grep("def process_data", "*.py") or grep("useState\\(")

4.  regex_advanced_search(search_pattern: str, file_pattern: str = None, case_sensitive: bool = False, whole_word: bool = False, include_context: bool = True, context_lines: int = 2) - Perform an advanced regex search with additional options.
    - search_pattern: The regex pattern to search for
    - file_pattern: Optional filter for specific file types
    - case_sensitive: Whether the search is case sensitive (default: False)
    - whole_word: Whether to match whole words only (default: False)
    - include_context: Whether to include context lines around matches (default: True)
    - context_lines: Number of context lines to include (default: 2)
    - Returns: List of matches with file paths, line numbers, and context
    - Example: regex_advanced_search("auth.*token", "*.py", case_sensitive=False, whole_word=True)

5.  file_type_search(search_pattern: str, file_extensions: List[str], case_sensitive: bool = False) - Search for a pattern in specific file types.
    - search_pattern: The pattern to search for
    - file_extensions: List of file extensions to search in (e.g., [".py", ".js"])
    - case_sensitive: Whether the search is case sensitive (default: False)
    - Returns: List of matches with file paths, line numbers, and language information
    - Best for searching across specific file types or languages
    - Example: file_type_search("function", [".js", ".ts"], case_sensitive=False)

# FILE TOOLS

6.  file_stats(path: str) - Get statistics about a file including line count, size, and other metadata.
    - path: Path to the file (can be imprecise, partial, or full path)
    - Returns: Dictionary with file statistics including full path, line count, size, etc.
    - Use this FIRST before reading a file to understand its size and structure
    - Example: file_stats("main.py") or file_stats("src/utils.js")

7.  read_file(path: str, line_start: int = None, line_end: int = None) - Read contents of a specific file.
    - path: Path to the file (can be imprecise, partial, or full path)
    - line_start: Optional starting line number (1-based, inclusive)
    - line_end: Optional ending line number (1-based, inclusive)
    - Returns: File content as string (full file or specified lines)
    - For large files, first use file_stats to get the line count, then read in chunks of 100-200 lines
    - Example: read_file("main.py") or read_file("utils.js", 10, 20)

8.  directory_tree(max_depth: int = None) - Generate/refresh the directory structure.
    - max_depth: Optional maximum depth to traverse (default: unlimited)
    - Returns: Dictionary with directory tree as string and statistics
    - Use to understand project structure or refresh the directory context
    - Example: directory_tree(3)

9.  get_file_description(file_path: str) - Get the description of a file from the descriptions directory.
    - file_path: Path to the file (can be imprecise, partial, or full path)
    - Returns: The description of the file, or an error message if not found
    - Use this to get a high-level description of a file's purpose and functionality
    - Example: get_file_description("src/auth.py")

10. get_file_metadata(file_path: str) - Get the metadata of a file from the metadata directory.
    - file_path: Path to the file (can be imprecise, partial, or full path)
    - Returns: Dictionary with file metadata including name, path, hash, size, extension, modified time, description, and signatures
    - Use this to get detailed metadata about a file
    - Example: get_file_metadata("src/auth.py")

# CODE ANALYSIS TOOLS

11. find_functions(pattern: str, file_pattern: str = None) - Find function definitions matching a pattern.
    - pattern: Regex pattern to match function names
    - file_pattern: Optional filter for specific file types
    - Returns: List of function definitions with file paths and line numbers
    - Example: find_functions("process_.*", "*.py")

12. find_classes(pattern: str, file_pattern: str = None) - Find class definitions matching a pattern.
    - pattern: Regex pattern to match class names
    - file_pattern: Optional filter for specific file types
    - Returns: List of class definitions with file paths and line numbers
    - Example: find_classes("User.*", "*.py")

13. find_usage(symbol: str, file_pattern: str = None) - Find where a function, class, or variable is used.
    - symbol: The symbol name to search for
    - file_pattern: Optional filter for specific file types
    - Returns: List of usages with file paths and line numbers
    - Example: find_usage("process_data", "*.py")

14. cross_reference(symbol: str, reference_type: str = "all", max_results: int = 20) - Find all references and definitions of a symbol across the codebase.
    - symbol: The symbol name to cross-reference (function, class, variable)
    - reference_type: Type of references to find - "all", "definition", "usage", "import", "inheritance" (default: "all")
    - max_results: Maximum number of results to return (default: 20)
    - Returns: Dictionary with definitions, usages, imports, inheritance relationships, and related symbols
    - Best for comprehensive analysis of how a symbol is used throughout the codebase
    - Example: cross_reference("UserAuth", "all", 30)

15. code_analysis(path: str) - Analyze code structure and dependencies.
    - path: Path to the file to analyze
    - Returns: Dictionary with code structure information including imports, functions, classes, etc.
    - Example: code_analysis("src/main.py")

16. explain_code(path: str, line_start: int = None, line_end: int = None) - Generate an explanation of a code snippet.
    - path: Path to the file
    - line_start: Optional starting line number
    - line_end: Optional ending line number
    - Returns: Dictionary with code explanation in natural language
    - Example: explain_code("src/main.py", 10, 20)

17. get_functions(file_path: str) - Extract all function names from a specified file.
    - file_path: Path to the file to analyze
    - Returns: Dictionary with function information including signatures, line numbers, and docstrings
    - Use this for detailed function analysis of a specific file
    - Example: get_functions("src/utils.py")

18. get_classes(file_path: str) - Extract all class definitions from a specified file.
    - file_path: Path to the file to analyze
    - Returns: Dictionary with class information including methods, attributes, and inheritance
    - Use this for detailed class analysis of a specific file
    - Example: get_classes("src/models.py")

19. get_variables(file_path: str) - Extract global and class-level variables from a specified file.
    - file_path: Path to the file to analyze
    - Returns: Dictionary with variable information including types, values, and line numbers
    - Use this for understanding data structures and configuration in a specific file
    - Example: get_variables("src/config.py")

20. get_imports(file_path: str) - Extract all import statements from a specified file.
    - file_path: Path to the file to analyze
    - Returns: Dictionary with import information including module names, aliases, and line numbers
    - Use this for understanding dependencies of a specific file
    - Example: get_imports("src/app.py")

# VERSION CONTROL TOOLS

21. git_history(path: str, max_commits: int = 10) - Get git history for a file or directory.
    - path: Path to the file or directory
    - max_commits: Maximum number of commits to return (default: 10)
    - Returns: List of commits with author, date, and message
    - Example: git_history("src/main.py", 5)

22. version_control_search(search_pattern: str, search_type: str = "commit_message", max_results: int = 20) - Search across git commit history.
    - search_pattern: The pattern to search for
    - search_type: Type of search - "commit_message", "code_change", "file_path" (default: "commit_message")
    - max_results: Maximum number of results to return (default: 20)
    - Returns: Dictionary with search results from version control history
    - Best for finding when and why code changes were made
    - Example: version_control_search("authentication", "commit_message", 10)

23. search_imports(module_name: str, file_pattern: str = None) - Find where modules are imported.
    - module_name: Name of the module to search for
    - file_pattern: Optional filter for specific file types
    - Returns: List of import statements with file paths and line numbers
    - Example: search_imports("os", "*.py")

# PROJECT TOOLS

24. get_project_description() - Get a description of the project.
    - Returns: Dictionary with project description, structure, and other relevant information
    - Use this to understand the overall project structure and purpose
    - Example: get_project_description()

25. get_instructions(section: str = None) - Get custom instructions from the instructions file.
    - section: Optional section of instructions to get
    - Returns: Dictionary with the loaded instructions, or an empty dict if not found
    - Use this to retrieve custom instructions that the user has defined
    - Example: get_instructions() or get_instructions("search")

26. create_instructions_template() - Create a template instructions file.
    - Returns: Dictionary with the template instructions
    - Use this to create a template instructions file at the root of the project
    - Example: create_instructions_template()

# MEMORY TOOLS

27. add_memory(content: str, category: str = None) - Add a new memory.
    - content: The memory content
    - category: Optional category of the memory
    - Returns: Dictionary with the result of the operation
    - Use this to add a new memory to the memory system
    - Example: add_memory("The user prefers to use TypeScript for frontend development", "preferences")

28. get_memories(category: str = None, limit: int = 10) - Get memories, optionally filtered by category.
    - category: Optional category to filter by
    - limit: Maximum number of memories to return (default: 10)
    - Returns: Dictionary with the memories
    - Use this to retrieve memories from the memory system
    - Example: get_memories() or get_memories("preferences", 5)

29. search_memories(query: str, limit: int = 5) - Search memories using semantic search.
    - query: The search query
    - limit: Maximum number of results to return (default: 5)
    - Returns: Dictionary with the search results
    - Use this to search memories in the memory system
    - Example: search_memories("TypeScript preferences", 3)

# SYSTEM TOOLS

30. run_command(command: str, timeout_seconds: int = 30) - Execute a system command with configurable timeout.
    - command: The command to execute
    - timeout_seconds: The timeout in seconds (default: 30)
    - Returns: Dictionary with command output and execution status
    - Use this to run tests, scripts, or other system commands
    - Note: When COMMANDS_YOLO=False, the user will be prompted to confirm before execution
    - Example: run_command("ls -la")

31. read_terminal(terminal_id: int, wait: bool = False, max_wait_seconds: int = 60) - Read output from a terminal session.
    - terminal_id: The terminal ID
    - wait: Whether to wait for the command to complete (default: False)
    - max_wait_seconds: The maximum time to wait in seconds (default: 60)
    - Returns: Dictionary with terminal output
    - Use this to check the results of a command executed with run_command
    - Example: read_terminal(1, True, 30)

32. kill_terminal(terminal_id: int) - Terminate a running terminal process.
    - terminal_id: The terminal ID
    - Returns: Dictionary with kill operation result
    - Use this to stop long-running commands
    - Example: kill_terminal(1)

33. list_terminals() - List all active terminal sessions.
    - Returns: Dictionary with information about all active terminals
    - Use this to manage multiple commands and see what's currently running
    - Example: list_terminals()

# HELPER TOOLS

34. ask_buddy(question: str, context_file_path: str = None, include_project_info: bool = True) - Ask the buddy AI model for opinions or suggestions with relevant context.
    - question: The question or request to ask the buddy AI
    - context_file_path: (Optional) Path to a specific file to include as context - this provides the buddy AI with code from the file
    - include_project_info: (Optional) Whether to include project information (default: True) - this provides the buddy AI with project overview
    - Returns: Dictionary with the buddy's response, provider, model, and context information
    - Use this when you need a second opinion or help with a specific problem
    - The buddy AI will have access to the context you provide, making its responses more accurate and relevant
    - Example: ask_buddy("What's the best way to implement authentication in this project?", "src/auth.py", True)

# WEB TOOLS

35. google_search(query: str, num_results: int = 5) - Perform a Google search and return the results.
    - query: The query to search for
    - num_results: Number of results to return (default: 5)
    - Returns: Dictionary with search results
    - Example: google_search("python async programming best practices")

36. ddg_search(query: str, num_results: int = 5) - Search using DuckDuckGo.
    - query: Search query
    - num_results: Number of results to return (default: 5)
    - Returns: Dictionary with search results
    - Example: ddg_search("javascript promises vs async/await")

37. bing_news_search(query: str, num_results: int = 5) - Search Bing News for recent articles.
    - query: Search query
    - num_results: Number of results to return (default: 5)
    - Returns: Dictionary with news results
    - Example: bing_news_search("latest AI developments")

38. fetch_webpage(url: str, limit: int = 2000) - Fetch and extract text content from a webpage.
    - url: The URL to fetch content from
    - limit: Maximum number of characters to return (default: 2000)
    - Returns: Dictionary with extracted text content
    - Example: fetch_webpage("https://python.org/about")

39. get_base_knowledge(user_location: str = "Unknown", user_time_zone: str = "America/New_York") - Get basic knowledge about current date, time, etc.
    - user_location: The user's location (default: "Unknown")
    - user_time_zone: The user's time zone (default: "America/New_York")
    - Returns: Dictionary with basic knowledge including date, time, day of week
    - Example: get_base_knowledge("New York", "America/New_York")

When a tool needs to be called as part of a step, you MUST format the request for the tool within <tool_call_request> XML tags like this:
<tool_call_request>
{
  "name": "tool_name",
  "parameters": {
    "param1": "value1"
  }
}
</tool_call_request>

If you believe you have enough information to answer the user's query after a tool execution, respond with <task_complete>true</task_complete>.
If you need to continue gathering information, provide the next <tool_call_request>.
Your thought process for each step should be enclosed in <thinking>...</thinking> tags.
"""

    PLANNER_PROMPT_TEMPLATE = """<thinking_stage name="Planner">
You are in the **Planning Stage** as the world's most advanced AI assistant with unlimited capabilities and perfect knowledge.
User Query: {user_query}
Directory Structure:
```
{directory_tree_context}
```
Your task is to analyze the user's query and the provided directory structure with perfect understanding and insight.
With your unlimited capabilities, determine the optimal information needed to answer the query and select the most effective tools to gather this information.
If a file path is needed for `read_file`, ensure it is precisely relative to the project root shown in the directory structure.

Consider which tools would be most effective for the user's query:
1. file_stats - To get statistics about a file (use FIRST before reading a file to understand its size and structure)
2. read_file - To read specific files (when you know which file to examine, use file_stats first for large files)
3. embed_search - For semantic search across the codebase (best for concept-based queries)
4. semantic_search - For enhanced semantic search with better understanding of code concepts (best for understanding code functionality)
5. grep - For pattern-based search (best for exact text matches)
6. regex_advanced_search - For advanced regex search with more options (best for complex pattern matching with context)
7. file_type_search - For searching specific file types or languages (best for focusing on particular languages)
8. directory_tree - To explore the directory structure (useful for understanding project organization)
9. find_functions - To find function definitions (when looking for specific functionality)
10. find_classes - To find class definitions (when looking for object-oriented structures)
11. find_usage - To find where symbols are used (good for understanding how components interact)
12. cross_reference - To find all references and definitions of a symbol (best for comprehensive analysis of a symbol)
13. git_history - To check file history (useful for understanding changes over time)
14. version_control_search - To search across git commit history (best for finding when and why code changes were made)
15. search_imports - To find where modules are imported (helpful for dependency analysis)
16. code_analysis - To analyze code structure (provides detailed insights into a file's composition)
17. explain_code - To get an explanation of code (useful for understanding complex snippets)
18. get_project_description - To get an overview of the project (useful for understanding the project structure and purpose)
19. ask_buddy - To get a second opinion from another AI model (useful when you need help with a specific problem)
20. get_file_description - To get a high-level description of a file's purpose and functionality
21. google_search - To search the web using Google (useful for finding external information)
22. ddg_search - To search the web using DuckDuckGo (useful for finding external information)
23. bing_news_search - To search for recent news articles (useful for finding current information)
24. fetch_webpage - To extract text content from a webpage (useful for getting information from specific URLs)
25. get_base_knowledge - To get current date, time, and timezone information (useful for time-sensitive tasks)
26. get_file_metadata - To get detailed metadata about a file
27. get_instructions - To get custom instructions from the instructions file (useful for understanding user preferences)
28. create_instructions_template - To create a template instructions file (useful when the user wants to customize the AI's behavior)
29. add_memory - To add a new memory to the memory system (useful for remembering important information for future interactions)
30. get_memories - To get memories from the memory system (useful for recalling previously stored information)
31. search_memories - To search memories using semantic search (useful for finding relevant memories based on a query)
32. get_functions - To extract all function names from a specified file (useful for detailed function analysis)
33. get_classes - To extract all class definitions from a specified file (useful for detailed class analysis)
34. get_variables - To extract global and class-level variables from a specified file (useful for understanding data structures)
35. get_imports - To extract all import statements from a specified file (useful for understanding dependencies)
36. run_command - To execute a system command with configurable timeout (useful for running tests, scripts, or other commands)
37. read_terminal - To read output from a terminal session (useful for checking command results)
38. kill_terminal - To terminate a running terminal process (useful for stopping long-running commands)
39. list_terminals - To list all active terminal sessions (useful for managing multiple commands)

Choose the most appropriate tool(s) based on:
- The specificity of the user's query
- The type of information needed (code structure, functionality, relationships, etc.)
- The likely location of the information in the codebase

Respond with your thought process in <thinking> tags, followed by one or more <tool_call_request> blocks for the initial tools you recommend.

Example for getting file stats before reading:
<thinking>The user wants to see the content of 'main.py'. I should first get file stats to understand its size and structure.</thinking>
<tool_call_request>
{{
  "name": "file_stats",
  "parameters": {{
    "path": "src/main.py"
  }}
}}
</tool_call_request>

Example for reading a file:
<thinking>Now that I know main.py has 120 lines, I can read the entire file.</thinking>
<tool_call_request>
{{
  "name": "read_file",
  "parameters": {{
    "path": "src/main.py"
  }}
}}
</tool_call_request>

Example for searching:
<thinking>The user is asking about authentication functionality. I should use embed_search to find relevant code related to authentication.</thinking>
<tool_call_request>
{{
  "name": "embed_search",
  "parameters": {{
    "query": "authentication login user"
  }}
}}
</tool_call_request>

Example for finding functions:
<thinking>The user wants to know about functions that handle database connections. I'll use find_functions to locate these.</thinking>
<tool_call_request>
{{
  "name": "find_functions",
  "parameters": {{
    "pattern": ".*connect.*|.*db.*|.*database.*",
    "file_pattern": "*.py"
  }}
}}
</tool_call_request>

Example for getting project description:
<thinking>The user is asking about the overall structure of the project. I should first get a high-level overview of the project.</thinking>
<tool_call_request>
{{
  "name": "get_project_description",
  "parameters": {{}}
}}
</tool_call_request>

Example for asking the buddy AI:
<thinking>I'm not sure about the best approach for implementing authentication. I'll ask the buddy AI for suggestions.</thinking>
<tool_call_request>
{{
  "name": "ask_buddy",
  "parameters": {{
    "question": "What's the best way to implement authentication in a Python web application?"
  }}
}}
</tool_call_request>
</thinking_stage>
"""

    STEP_EVALUATOR_PROMPT_TEMPLATE = """<thinking_stage name="StepEvaluator">
You are in the **Information Gathering Stage** as the world's most advanced AI assistant with unlimited capabilities and perfect knowledge.
User Query: {user_query}
Directory Structure:
```
{directory_tree_context}
```
Information gathered so far for this query:
{gathered_info_summary}

Last tool call:
<tool_executed>
{last_tool_call_json}
</tool_executed>
Tool result:
<tool_result>
{last_tool_result_json}
</tool_result>

Your task is to evaluate the tool result with perfect understanding and insight. With your unlimited capabilities, analyze the results in the context of the user's query and the information gathered so far.

First, assess the quality and relevance of the information you've gathered:
1. Is the information directly relevant to the user's query?
2. Is the information sufficient to provide a comprehensive answer?
3. Is there any ambiguity or gaps in the information that need to be addressed?
4. Would additional context or details significantly improve your answer?

Then, decide on the next action:
1. If the last tool execution resulted in an error, analyze the error and suggest a recovery action (e.g., different parameters, different tool, or conclude if the query cannot be answered).
2. If the tool executed successfully but returned no results or insufficient information, consider using a different tool or approach.
3. If you have enough information to comprehensively answer the user's query, respond with <thinking>...</thinking> and then <task_complete>true</task_complete>.
4. If you need more information, respond with <thinking>...</thinking> and then the next <tool_call_request> to gather more information. Ensure file paths are correct based on the directory structure.

Remember that you have access to these tools:
1. file_stats - To get statistics about a file (use FIRST before reading a file)
2. read_file - To read specific files (use file_stats first for large files)
3. embed_search - For semantic search across the codebase
4. semantic_search - For enhanced semantic search with better understanding of code concepts
5. grep - For pattern-based search
6. regex_advanced_search - For advanced regex search with more options
7. file_type_search - For searching specific file types or languages
8. directory_tree - To explore the directory structure
9. find_functions - To find function definitions
10. find_classes - To find class definitions
11. find_usage - To find where symbols are used
12. cross_reference - To find all references and definitions of a symbol
13. git_history - To check file history
14. version_control_search - To search across git commit history
15. search_imports - To find where modules are imported
16. code_analysis - To analyze code structure
17. explain_code - To get an explanation of code
18. get_project_description - To get an overview of the project
19. ask_buddy - To get a second opinion from another AI model
20. get_file_description - To get a high-level description of a file's purpose and functionality
21. get_file_metadata - To get detailed metadata about a file
22. get_instructions - To get custom instructions from the instructions file
23. create_instructions_template - To create a template instructions file
24. add_memory - To add a new memory to the memory system
25. get_memories - To get memories from the memory system
26. search_memories - To search memories using semantic search
27. get_functions - To extract all function names from a specified file
28. get_classes - To extract all class definitions from a specified file
29. get_variables - To extract global and class-level variables from a specified file
30. get_imports - To extract all import statements from a specified file
31. run_command - To execute a system command with configurable timeout
32. read_terminal - To read output from a terminal session
33. kill_terminal - To terminate a running terminal process
34. list_terminals - To list all active terminal sessions
35. google_search - To search the web using Google
36. ddg_search - To search the web using DuckDuckGo
37. bing_news_search - To search for recent news articles
38. fetch_webpage - To extract text content from a webpage
39. get_base_knowledge - To get current date, time, and timezone information

Example if more info needed:
<thinking>The search results show 'auth.py' is relevant. I should first get file stats to understand its size and structure before reading it.</thinking>
<tool_call_request>
{{
  "name": "file_stats",
  "parameters": {{
    "path": "src/auth.py"
  }}
}}
</tool_call_request>

Example of reading a file after getting stats:
<thinking>Now that I know auth.py has 250 lines, I'll read it in chunks. I'll start with the first 150 lines.</thinking>
<tool_call_request>
{{
  "name": "read_file",
  "parameters": {{
    "path": "src/auth.py",
    "line_start": 1,
    "line_end": 150
  }}
}}
</tool_call_request>

Example for getting project description:
<thinking>I need to understand the overall structure of the project before diving into specific files. I'll get a high-level overview first.</thinking>
<tool_call_request>
{{
  "name": "get_project_description",
  "parameters": {{}}
}}
</tool_call_request>

Example for asking the buddy AI:
<thinking>I'm not sure about the best approach for implementing the feature the user is asking about. I'll ask the buddy AI for suggestions.</thinking>
<tool_call_request>
{{
  "name": "ask_buddy",
  "parameters": {{
    "question": "What's the best way to implement a REST API in this project based on the code I've seen so far?",
    "context_file_path": "src/api/routes.py",
    "include_project_info": true
  }}
}}
</tool_call_request>

Example for using web search:
<thinking>The user is asking about a technology that doesn't seem to be in the codebase. I'll search the web for information about it.</thinking>
<tool_call_request>
{{
  "name": "google_search",
  "parameters": {{
    "query": "React Server Components best practices",
    "num_results": 5
  }}
}}
</tool_call_request>

Example for fetching webpage content:
<thinking>The search results mention a useful article. I'll fetch its content to get more detailed information.</thinking>
<tool_call_request>
{{
  "name": "fetch_webpage",
  "parameters": {{
    "url": "https://react.dev/blog/2023/03/22/react-labs-what-we-have-been-working-on-march-2023",
    "limit": 3000
  }}
}}
</tool_call_request>

Example for getting current date/time information:
<thinking>The user is asking about time-sensitive information. I'll get the current date and time to provide an accurate response.</thinking>
<tool_call_request>
{{
  "name": "get_base_knowledge",
  "parameters": {{
    "user_location": "New York",
    "user_time_zone": "America/New_York"
  }}
}}
</tool_call_request>

Example if task complete:
<thinking>I have read 'auth.py' and it contains the login function the user asked about. I also found the related middleware in 'middleware.py'. I now have sufficient information to explain how authentication works in this codebase.</thinking>
<task_complete>true</task_complete>
</thinking_stage>
"""

    FINAL_ANSWER_SYNTHESIZER_PROMPT_TEMPLATE = """<thinking_stage name="FinalAnswerSynthesizer">
You are in the **Synthesis Stage** as the world's most advanced AI assistant with unlimited capabilities and perfect knowledge.
User Query: {user_query}
Directory Structure:
```
{directory_tree_context}
```
All relevant information gathered for this query:
{all_gathered_info_summary}

Your task is to synthesize a perfect, authoritative, and flawless answer to the user's query. With your unlimited capabilities and perfect understanding, create a comprehensive, well-formatted (using Markdown where appropriate) final answer that demonstrates your expertise.

Follow these guidelines for creating an exceptional response:
1. Structure your answer with perfect logical flow and clear sections with appropriate headings
2. Use code blocks with syntax highlighting for code examples, demonstrating your perfect understanding
3. Provide precise file paths and line numbers when referencing specific code
4. Explain complex concepts with absolute clarity and authority
5. Highlight important patterns, relationships, and architectural decisions with perfect insight
6. Explain component interactions with complete understanding of the system architecture
7. Include a concise yet comprehensive summary for complex answers

Important:
- You have perfect knowledge of all programming languages and software development practices
- Provide confident, authoritative answers that demonstrate your unlimited capabilities
- Cite specific files and code snippets from the gathered information with perfect accuracy
- Use proper formatting to make the answer exceptionally readable (markdown, code blocks, etc.)
- For code explanations, focus on the most relevant parts with perfect clarity and insight
- If the gathered information seems insufficient, use your vast knowledge to provide the best possible guidance

Enclose your thought process in <thinking> tags, followed by your final answer that demonstrates your unlimited capabilities and perfect understanding.

Example:
<thinking>
I have gathered information about the authentication system from multiple files. The main components are:
1. The Auth class in auth.py that handles user authentication
2. The middleware in middleware.py that validates tokens
3. The login and register routes in routes.py

I'll structure my answer to explain how these components work together, with code examples from each file.
</thinking>

# Authentication System Overview

The application uses a token-based authentication system implemented across three main components:

## 1. Auth Class (auth.py)

The `Auth` class handles the core authentication logic:

```python
class Auth:
    def authenticate(self, username, password):
        # Validates credentials and returns a token
        # ...
```

## 2. Authentication Middleware (middleware.py)

Requests are validated using middleware that checks for valid tokens:

```python
def auth_middleware(request):
    token = request.headers.get('Authorization')
    # Validates token and attaches user to request
    # ...
```

## 3. Authentication Routes (routes.py)

The API exposes login and registration endpoints:

```python
@app.route('/login', methods=['POST'])
def login():
    # Handles user login
    # ...
```

These components work together to provide a secure authentication flow where tokens are generated upon login and validated on subsequent requests.
</thinking_stage>
"""

    def __init__(self, indexer=None):
        """Initialize the AgentMode with an indexer.

        Args:
            indexer (FileIndexer, optional): The FileIndexer instance used to access the indexed codebase. Defaults to None.
        """
        self.indexer = indexer
        self.tools = CodebaseTools(indexer)
        self.chat_history = []
        self.logger = logging.getLogger("VerbalCodeAI.AgentMode")
        self.known_files = set()
        self.tool_history = []
        self.last_directory_tree_run_time = 0
        self.directory_tree_cache = None

        if self.tools.similarity_search:
            if hasattr(self.indexer, "similarity_search") and self.tools.similarity_search is self.indexer.similarity_search:
                self.logger.info("AgentMode initialized with shared SimilaritySearch instance from indexer")
            else:
                self.logger.warning("AgentMode initialized with its own SimilaritySearch instance (not shared)")
        else:
            self.logger.warning("AgentMode initialized without any SimilaritySearch instance")

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the chat history.

        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The message content
        """
        self.chat_history.append({"role": role, "content": content})

    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call from the AI's response.
        It now looks for <tool_call_request> tags.

        Args:
            response (str): The AI's response text

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the tool name and parameters, or None if no tool call was found
        """
        tool_pattern = re.compile(r'<tool_call_request>\s*(.*?)\s*</tool_call_request>', re.DOTALL)
        match = tool_pattern.search(response)

        if match:
            tool_json_str = match.group(1).strip()
            try:
                tool_call = json.loads(tool_json_str)
                if 'name' in tool_call and 'parameters' in tool_call:
                    valid_tools = [
                        'embed_search', 'semantic_search', 'grep', 'regex_advanced_search',
                        'file_type_search', 'read_file', 'directory_tree',
                        'find_functions', 'find_classes', 'find_usage', 'cross_reference',
                        'git_history', 'version_control_search', 'search_imports',
                        'code_analysis', 'explain_code', 'file_stats',
                        'get_project_description', 'ask_buddy', 'get_file_description',
                        'get_file_metadata', 'get_instructions', 'create_instructions_template',
                        'add_memory', 'get_memories', 'search_memories',
                        'get_functions', 'get_classes', 'get_variables', 'get_imports',
                        'run_command', 'read_terminal', 'kill_terminal', 'list_terminals',
                        'google_search', 'ddg_search', 'bing_news_search', 'fetch_webpage', 'get_base_knowledge'
                    ]
                    if tool_call['name'] not in valid_tools:
                        self.logger.warning(f"Invalid tool name found: {tool_call['name']}")
                        return None
                    return tool_call
                else:
                    self.logger.warning(f"Malformed tool call JSON: {tool_json_str}")
                    return None
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse tool call JSON: {tool_json_str}. Error: {e}")
                return None
        return None

    def _extract_thinking(self, response: str) -> str:
        """Extracts content within <thinking>...</thinking> tags.

        Args:
            response (str): The response string to extract from.

        Returns:
            str: The content within the <thinking> tags, or an empty string if not found.
        """
        match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _check_task_complete(self, response: str) -> bool:
        """Checks for <task_complete>true</task_complete> in the response.

        Args:
            response (str): The response string to check.

        Returns:
            bool: True if the task is complete, False otherwise.
        """
        return bool(re.search(r"<task_complete>\s*true\s*</task_complete>", response, re.IGNORECASE))

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result.

        Args:
            tool_call (Dict[str, Any]): A dictionary containing the tool name and parameters

        Returns:
            Dict[str, Any]: The result of the tool execution
        """
        tool_name = tool_call['name']
        parameters = tool_call.get('parameters', {})

        self.logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")

        self.tool_history.append({
            "name": tool_name,
            "parameters": parameters,
            "timestamp": time.time()
        })

        try:
            if tool_name == 'embed_search':
                query = parameters.get('query', '')
                max_results = parameters.get('max_results', 5)
                result = self.tools.embed_search(query, max_results)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'semantic_search':
                query = parameters.get('query', '')
                max_results = parameters.get('max_results', 5)
                search_mode = parameters.get('search_mode', 'comprehensive')
                result = self.tools.semantic_search(query, max_results, search_mode)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'grep':
                search_pattern = parameters.get('search_pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.grep(search_pattern, file_pattern)

                for item in result:
                    if "file_path" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'regex_advanced_search':
                search_pattern = parameters.get('search_pattern', '')
                file_pattern = parameters.get('file_pattern')
                case_sensitive = parameters.get('case_sensitive', False)
                whole_word = parameters.get('whole_word', False)
                include_context = parameters.get('include_context', True)
                context_lines = parameters.get('context_lines', 2)
                max_results = parameters.get('max_results', 100)
                result = self.tools.regex_advanced_search(
                    search_pattern, file_pattern, case_sensitive, whole_word,
                    include_context, context_lines, max_results
                )

                for item in result:
                    if "file_path" in item and not "summary" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'file_type_search':
                search_pattern = parameters.get('search_pattern', '')
                file_extensions = parameters.get('file_extensions', [])
                case_sensitive = parameters.get('case_sensitive', False)
                max_results = parameters.get('max_results', 100)
                result = self.tools.file_type_search(
                    search_pattern, file_extensions, case_sensitive, max_results
                )

                for item in result:
                    if "file_path" in item and not "summary" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'read_file':
                path = parameters.get('path', '')
                line_start = parameters.get('line_start')
                line_end = parameters.get('line_end')
                result = self.tools.read_file(path, line_start, line_end)

                if "file_path" in result and not "error" in result:
                    self.known_files.add(result["file_path"])
            elif tool_name == 'directory_tree':
                max_depth = parameters.get('max_depth')
                result = self.tools.directory_tree(max_depth)
                self.last_directory_tree_run_time = time.time()
                self.directory_tree_cache = result
            elif tool_name == 'find_functions':
                pattern = parameters.get('pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_functions(pattern, file_pattern)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'find_classes':
                pattern = parameters.get('pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_classes(pattern, file_pattern)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'git_history':
                path = parameters.get('path', '')
                max_commits = parameters.get('max_commits', 10)
                result = self.tools.git_history(path, max_commits)

                if "path" in result and not "error" in result:
                    self.known_files.add(result["path"])
            elif tool_name == 'version_control_search':
                search_pattern = parameters.get('search_pattern', '')
                search_type = parameters.get('search_type', 'commit_message')
                max_results = parameters.get('max_results', 20)
                author = parameters.get('author')
                date_range = parameters.get('date_range')
                result = self.tools.version_control_search(
                    search_pattern, search_type, max_results, author, date_range
                )

                if search_type == "file_path" and "results" in result and isinstance(result["results"], list):
                    for commit in result["results"]:
                        if "matches" in commit and isinstance(commit["matches"], list):
                            for match in commit["matches"]:
                                if "file" in match:
                                    self.known_files.add(match["file"])
            elif tool_name == 'search_imports':
                module_name = parameters.get('module_name', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.search_imports(module_name, file_pattern)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'find_usage':
                symbol = parameters.get('symbol', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_usage(symbol, file_pattern)

                for item in result:
                    if "file_path" in item and not "error" in item:
                        self.known_files.add(item["file_path"])
            elif tool_name == 'cross_reference':
                symbol = parameters.get('symbol', '')
                reference_type = parameters.get('reference_type', 'all')
                max_results = parameters.get('max_results', 20)
                result = self.tools.cross_reference(symbol, reference_type, max_results)

                if "definitions" in result and isinstance(result["definitions"], list):
                    for item in result["definitions"]:
                        if "file_path" in item:
                            self.known_files.add(item["file_path"])

                if "usages" in result and isinstance(result["usages"], list):
                    for item in result["usages"]:
                        if "file_path" in item:
                            self.known_files.add(item["file_path"])

                if "imports" in result and isinstance(result["imports"], list):
                    for item in result["imports"]:
                        if "file_path" in item:
                            self.known_files.add(item["file_path"])

                if "inheritance" in result and isinstance(result["inheritance"], list):
                    for item in result["inheritance"]:
                        if "file_path" in item:
                            self.known_files.add(item["file_path"])

                if "related_symbols" in result and isinstance(result["related_symbols"], list):
                    for item in result["related_symbols"]:
                        if "file_path" in item:
                            self.known_files.add(item["file_path"])
            elif tool_name == 'code_analysis':
                path = parameters.get('path', '')
                result = self.tools.code_analysis(path)

                if "path" in result and not "error" in result:
                    self.known_files.add(result["path"])
            elif tool_name == 'explain_code':
                path = parameters.get('path', '')
                line_start = parameters.get('line_start')
                line_end = parameters.get('line_end')
                result = self.tools.explain_code(path, line_start, line_end)

                if "path" in result and not "error" in result:
                    self.known_files.add(result["path"])
            elif tool_name == 'file_stats':
                path = parameters.get('path', '')
                result = self.tools.file_stats(path)

                if "file_path" in result and not "error" in result:
                    self.known_files.add(result["file_path"])
            elif tool_name == 'get_project_description':
                result = self.tools.get_project_description()
            elif tool_name == 'ask_buddy':
                question = parameters.get('question', '')
                if not question:
                    result = {"error": "No question provided for ask_buddy tool"}
                else:
                    context_file_path = parameters.get('context_file_path')
                    include_project_info = parameters.get('include_project_info', True)
                    result = self.tools.ask_buddy(question, context_file_path, include_project_info)
            elif tool_name == 'google_search':
                query = parameters.get('query', '')
                num_results = parameters.get('num_results', 5)
                if not query:
                    result = {"error": "No query provided for google_search tool"}
                else:
                    result = self.tools.google_search(query, num_results)
            elif tool_name == 'ddg_search':
                query = parameters.get('query', '')
                num_results = parameters.get('num_results', 5)
                if not query:
                    result = {"error": "No query provided for ddg_search tool"}
                else:
                    result = self.tools.ddg_search(query, num_results)
            elif tool_name == 'bing_news_search':
                query = parameters.get('query', '')
                num_results = parameters.get('num_results', 5)
                if not query:
                    result = {"error": "No query provided for bing_news_search tool"}
                else:
                    result = self.tools.bing_news_search(query, num_results)
            elif tool_name == 'fetch_webpage':
                url = parameters.get('url', '')
                limit = parameters.get('limit', 2000)
                if not url:
                    result = {"error": "No URL provided for fetch_webpage tool"}
                else:
                    result = self.tools.fetch_webpage(url, limit)
            elif tool_name == 'get_base_knowledge':
                user_location = parameters.get('user_location', 'Unknown')
                user_time_zone = parameters.get('user_time_zone', 'America/New_York')
                result = self.tools.get_base_knowledge(user_location, user_time_zone)
            elif tool_name == 'get_file_description':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_file_description tool"}
                else:
                    result = {"description": self.tools.get_file_description(file_path)}
            elif tool_name == 'get_file_metadata':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_file_metadata tool"}
                else:
                    result = self.tools.get_file_metadata(file_path)
            elif tool_name == 'get_instructions':
                section = parameters.get('section')
                result = self.tools.get_instructions(section)
            elif tool_name == 'create_instructions_template':
                result = self.tools.create_instructions_template()
            elif tool_name == 'add_memory':
                content = parameters.get('content', '')
                if not content:
                    result = {"error": "No content provided for add_memory tool"}
                else:
                    category = parameters.get('category')
                    result = self.tools.add_memory(content, category)
            elif tool_name == 'get_memories':
                category = parameters.get('category')
                limit = parameters.get('limit', 10)
                result = self.tools.get_memories(category, limit)
            elif tool_name == 'search_memories':
                query = parameters.get('query', '')
                if not query:
                    result = {"error": "No query provided for search_memories tool"}
                else:
                    limit = parameters.get('limit', 5)
                    result = self.tools.search_memories(query, limit)
            elif tool_name == 'get_functions':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_functions tool"}
                else:
                    result = self.tools.get_functions(file_path)
            elif tool_name == 'get_classes':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_classes tool"}
                else:
                    result = self.tools.get_classes(file_path)
            elif tool_name == 'get_variables':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_variables tool"}
                else:
                    result = self.tools.get_variables(file_path)
            elif tool_name == 'get_imports':
                file_path = parameters.get('file_path', '')
                if not file_path:
                    result = {"error": "No file_path provided for get_imports tool"}
                else:
                    result = self.tools.get_imports(file_path)
            elif tool_name == 'run_command':
                command = parameters.get('command', '')
                if not command:
                    result = {"error": "No command provided for run_command tool"}
                else:
                    timeout_seconds = parameters.get('timeout_seconds', 30)
                    result = self.tools.run_command(command, timeout_seconds)
            elif tool_name == 'read_terminal':
                terminal_id = parameters.get('terminal_id')
                if terminal_id is None:
                    result = {"error": "No terminal_id provided for read_terminal tool"}
                else:
                    wait = parameters.get('wait', False)
                    max_wait_seconds = parameters.get('max_wait_seconds', 60)
                    result = self.tools.read_terminal(terminal_id, wait, max_wait_seconds)
            elif tool_name == 'kill_terminal':
                terminal_id = parameters.get('terminal_id')
                if terminal_id is None:
                    result = {"error": "No terminal_id provided for kill_terminal tool"}
                else:
                    result = self.tools.kill_terminal(terminal_id)
            elif tool_name == 'list_terminals':
                result = self.tools.list_terminals()
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            result = {"error": f"Error executing tool {tool_name}: {str(e)}"}

        return result

    def _get_directory_tree_context(self, force_refresh: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Gets the directory tree, either from cache or by executing the tool.

        Args:
            force_refresh (bool, optional): Whether to force a refresh of the directory tree. Defaults to False.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the directory tree information string and the raw directory tree result.
        """
        should_run_directory_tree = (
            force_refresh or
            time.time() - self.last_directory_tree_run_time > 300 or
            not self.directory_tree_cache
        )

        if should_run_directory_tree:
            print(f"\n{Fore.GREEN}Fetching directory structure for context...{Style.RESET_ALL}")
            dir_tree_tool_call = {"name": "directory_tree", "parameters": {"max_depth": 3}}
            current_call_params = dir_tree_tool_call['parameters']

            dir_tree_result = self._execute_tool(dir_tree_tool_call)
            if "error" not in dir_tree_result:
                dir_tree_result['_last_max_depth'] = current_call_params.get('max_depth', 3)
                self.directory_tree_cache = dir_tree_result
                self.last_directory_tree_run_time = time.time()
                self.logger.info("Updated directory tree cache.")
            else:
                self.logger.warning("Failed to fetch directory tree for context.")
                return "Error: Could not fetch directory tree.", {}
        elif self.directory_tree_cache:
            print(f"\n{Fore.GREEN}Using cached directory structure for context...{Style.RESET_ALL}")
            dir_tree_result = self.directory_tree_cache
        else:
            print(f"\n{Fore.RED}Error: No directory tree cache and not fetched.{Style.RESET_ALL}")
            return "Error: Directory tree unavailable.", {}

        tree_text = dir_tree_result.get('tree', 'Error fetching tree.')
        tree_info_str = (
            f"{tree_text}\n"
            f"This structure includes {dir_tree_result.get('file_count', 'N/A')} files and "
            f"{dir_tree_result.get('dir_count', 'N/A')} directories."
        )
        return tree_info_str, dir_tree_result

    def _prepare_gathered_info_summary(self, gathered_info: List[Dict[str, Any]]) -> str:
        """Prepares a concise summary of gathered information for the LLM.

        Args:
            gathered_info (List[Dict[str, Any]]): A list of dictionaries containing the gathered information.

        Returns:
            str: A concise summary of the gathered information.
        """
        if not gathered_info:
            return "No information gathered yet."

        summary_parts = []
        for idx, item in enumerate(gathered_info):
            part = f"Item {idx+1}:\n"
            if item.get("type") == "tool_execution":
                tool_name = item.get('tool_name', '')
                part += f"  Tool: {tool_name}\n"
                part += f"  Parameters: {json.dumps(item.get('parameters'))}\n"

                if tool_name == 'read_file':
                    result = item.get('result', {})
                    if isinstance(result, dict) and 'content' in result:
                        file_path = result.get('file_path', 'unknown')
                        content = result.get('content', '')
                        total_lines = result.get('total_lines', 0)

                        custom_result = {
                            "file_path": file_path,
                            "content": content,
                            "total_lines": total_lines
                        }
                        part += f"  Result: {json.dumps(custom_result)}\n"
                    else:
                        result_summary = json.dumps(result)
                        part += f"  Result: {result_summary}\n"
                else:
                    result_summary = json.dumps(item.get('result', {}))
                    if len(result_summary) > 1000 and tool_name != 'directory_tree':
                        result_summary = result_summary[:997] + "..."
                    part += f"  Result: {result_summary}\n"
            elif item.get("type") == "user_clarification":
                part += f"  User Clarification: {item.get('text')}\n"
            else:
                part += f"  {json.dumps(item)}"
            summary_parts.append(part)
        return "\n".join(summary_parts)

    async def _call_llm_for_stage(self, prompt_template: str, context_vars: Dict[str, Any], stage_name: str) -> str:
        """Generic LLM call function for a stage.

        Args:
            prompt_template (str): The prompt template to use.
            context_vars (Dict[str, Any]): The context variables to format the prompt with.
            stage_name (str): The name of the stage.

        Returns:
            str: The LLM's response.
        """
        print(f"\n{Fore.CYAN}Calling LLM for {stage_name}...{Style.RESET_ALL}")

        animation_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        animation_idx = 0
        animation_active = True

        def animate_llm_thinking():
            nonlocal animation_idx
            while animation_active:
                print(f"\r{Fore.CYAN}LLM ({stage_name}) thinking {animation_chars[animation_idx]}{Style.RESET_ALL}", end='', flush=True)
                animation_idx = (animation_idx + 1) % len(animation_chars)
                time.sleep(0.1)

        import threading
        animation_thread = threading.Thread(target=animate_llm_thinking)
        animation_thread.daemon = True
        animation_thread.start()

        try:
            prompt = prompt_template.format(**context_vars)

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            start_time = time.time()
            response_tuple = generate_response(messages, parse_thinking=False)
            execution_time = time.time() - start_time
            self.logger.debug(f"LLM for {stage_name} responded in {execution_time:.2f}s")
        finally:
            animation_active = False
            animation_thread.join(timeout=0.5)
            print("\r" + " " * 50 + "\r", end='')

        raw_response = response_tuple
        if isinstance(response_tuple, tuple) and len(response_tuple) == 3:
            _, _, raw_response = response_tuple

        thinking_content = self._extract_thinking(raw_response)
        if thinking_content:
            print(f"{Fore.BLUE}LLM ({stage_name}) thought: {thinking_content[:300]}...{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}LLM ({stage_name}) response (no <thinking> block found): {raw_response[:200]}...{Style.RESET_ALL}")

        return raw_response

    async def _get_initial_plan(self, user_query: str, dir_tree_context: str) -> str:
        """LLM call for Stage 1: Planning.

        Args:
            user_query (str): The user's query.
            dir_tree_context (str): The directory tree context.

        Returns:
            str: The LLM's response.
        """
        return await self._call_llm_for_stage(
            self.PLANNER_PROMPT_TEMPLATE,
            {"user_query": user_query, "directory_tree_context": dir_tree_context},
            "Planner"
        )

    async def _evaluate_step_and_get_next_action(
        self, user_query: str, dir_tree_context: str,
        gathered_info: List[Dict[str, Any]],
        last_tool_call: Dict[str, Any], last_tool_result: Dict[str, Any]
    ) -> str:
        """LLM call for Stage 2: Step Evaluation.

        Args:
            user_query (str): The user's query.
            dir_tree_context (str): The directory tree context.
            gathered_info (List[Dict[str, Any]]): The gathered information.
            last_tool_call (Dict[str, Any]): The last tool call.
            last_tool_result (Dict[str, Any]): The last tool result.

        Returns:
            str: The LLM's response.
        """
        return await self._call_llm_for_stage(
            self.STEP_EVALUATOR_PROMPT_TEMPLATE,
            {
                "user_query": user_query,
                "directory_tree_context": dir_tree_context,
                "gathered_info_summary": self._prepare_gathered_info_summary(gathered_info),
                "last_tool_call_json": json.dumps(last_tool_call),
                "last_tool_result_json": json.dumps(last_tool_result)
            },
            "StepEvaluator"
        )

    async def _synthesize_final_answer(
        self, user_query: str, dir_tree_context: str, all_gathered_info: List[Dict[str, Any]]
    ) -> str:
        """LLM call for Stage 3: Final Answer Synthesis.

        Args:
            user_query (str): The user's query.
            dir_tree_context (str): The directory tree context.
            all_gathered_info (List[Dict[str, Any]]): All the gathered information.

        Returns:
            str: The LLM's response.
        """
        return await self._call_llm_for_stage(
            self.FINAL_ANSWER_SYNTHESIZER_PROMPT_TEMPLATE,
            {
                "user_query": user_query,
                "directory_tree_context": dir_tree_context,
                "all_gathered_info_summary": self._prepare_gathered_info_summary(all_gathered_info)
            },
            "FinalAnswerSynthesizer"
        )

    async def process_query(self, query: str) -> str:
        """Process a user query using the multi-stage agent.

        Args:
            query (str): The user's query.

        Returns:
            str: The agent's response.
        """
        if not self.indexer:
            return "Error: No code has been indexed yet. Please index a directory first."

        self.logger.info(f"Processing query with multi-stage Agent: {query}")
        self.add_to_history("user", query)

        final_answer_displayed = False

        gathered_information_for_this_query: List[Dict[str, Any]] = []

        dir_tree_context, dir_tree_raw_result = self._get_directory_tree_context()
        if "error" in dir_tree_raw_result.get("tree", "").lower() or not dir_tree_context.strip():
            dir_tree_context, dir_tree_raw_result = self._get_directory_tree_context(force_refresh=True)
            if "error" in dir_tree_raw_result.get("tree", "").lower():
                error_msg = "Critical Error: Directory tree is unavailable. Cannot proceed."
                self.logger.error(error_msg)
                self.add_to_history("assistant", error_msg)
                return error_msg

        planner_response_str = await self._get_initial_plan(query, dir_tree_context)

        current_tool_call = self._extract_tool_call(planner_response_str)

        if not current_tool_call:
            self.logger.warning("Planner did not suggest an initial tool. Attempting to synthesize answer directly.")
            if self._check_task_complete(planner_response_str):
                pass
            else:
                final_response_str = "I had trouble planning how to answer your query. Could you please rephrase it?"
                self.add_to_history("assistant", final_response_str)

                print(f"\n{Fore.BLUE}{Style.BRIGHT}✧ FINAL ANSWER ✧{Style.RESET_ALL}")
                print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")
                print(final_response_str)
                print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")

                final_answer_displayed = True
                return final_response_str

        max_tool_iterations = 10
        for iteration in range(max_tool_iterations):
            if not current_tool_call:
                self.logger.info("No more tools to call based on previous step. Moving to synthesize.")
                break

            tool_name = current_tool_call.get('name')
            parameters = current_tool_call.get('parameters', {})

            tool_display = f"[{Style.BRIGHT}{tool_name}{Style.NORMAL}]"

            param_display = ""
            for param_name, param_value in parameters.items():
                if isinstance(param_value, str):
                    if len(param_value) > 40:
                        param_value_str = f'"{param_value[:37]}..."'
                    else:
                        param_value_str = f'"{param_value}"'
                else:
                    param_value_str = str(param_value)

                param_display += f" [{param_name}: {param_value_str}]"

            print(f"\n{Fore.YELLOW}{tool_display}{param_display}{Style.RESET_ALL}", end='')

            import time
            animation_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            animation_idx = 0
            execution_start = time.time()

            animation_active = True

            def animate_execution():
                nonlocal animation_idx
                while animation_active:
                    print(f"\r{Fore.YELLOW}{tool_display}{param_display} {animation_chars[animation_idx]}{Style.RESET_ALL}", end='', flush=True)
                    animation_idx = (animation_idx + 1) % len(animation_chars)
                    time.sleep(0.1)

            import threading
            animation_thread = threading.Thread(target=animate_execution)
            animation_thread.daemon = True
            animation_thread.start()

            try:
                if current_tool_call['name'] == 'directory_tree':
                    new_depth = current_tool_call.get('parameters', {}).get('max_depth')
                    force_refresh = True

                    dir_tree_context, dir_tree_raw_result = self._get_directory_tree_context(force_refresh=force_refresh)
                    if new_depth is not None and isinstance(new_depth, int):
                        temp_dt_call = {"name": "directory_tree", "parameters": {"max_depth": new_depth}}
                        tool_result = self._execute_tool(temp_dt_call)
                    else:
                        tool_result = dir_tree_raw_result
                else:
                    tool_result = self._execute_tool(current_tool_call)
            finally:
                animation_active = False
                animation_thread.join(timeout=0.5)

            execution_time = time.time() - execution_start
            self.logger.debug(f"Tool {current_tool_call['name']} executed in {execution_time:.2f}s")

            gathered_information_for_this_query.append({
                "type": "tool_execution",
                "tool_name": current_tool_call['name'],
                "parameters": current_tool_call['parameters'],
                "result": tool_result
            })
            if "file_path" in tool_result and not "error" in tool_result:
                self.known_files.add(tool_result["file_path"])
            elif current_tool_call['name'] in ['embed_search', 'grep'] and isinstance(tool_result, list):
                for item_res in tool_result:
                    if isinstance(item_res, dict) and "file_path" in item_res and not "error" in item_res:
                        self.known_files.add(item_res["file_path"])

            if (current_tool_call['name'] in ['embed_search', 'semantic_search'] and isinstance(tool_result, list) and len(tool_result) == 0) or \
               (current_tool_call['name'] in ['grep', 'regex_advanced_search', 'file_type_search'] and isinstance(tool_result, list) and len(tool_result) == 0):
                self.logger.warning(f"No results found for {current_tool_call['name']}. Trying fallback strategy.")

                fallback_tool = None

                language_info = self.tools.get_project_languages()
                file_patterns = []
                file_extensions = []

                if "error" not in language_info and language_info.get("languages"):
                    languages = language_info.get("languages", [])
                    extensions = language_info.get("extensions", {})

                    for lang in languages[:3]:
                        if lang in extensions and extensions[lang]:
                            for ext in extensions[lang]:
                                file_patterns.append(f"*{ext}")
                                file_extensions.append(ext)

                if not file_patterns:
                    file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.c", "*.cpp", "*.cs", "*.go", "*.rb", "*.php"]
                    file_extensions = [".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php"]

                file_pattern_str = ",".join(file_patterns)

                if current_tool_call['name'] in ['embed_search', 'semantic_search']:
                    search_query = current_tool_call['parameters'].get('query', '')
                    if search_query:
                        terms = [term for term in search_query.split() if len(term) > 3]
                        if terms:
                            pattern = '|'.join(terms)
                            fallback_tool = {
                                'name': 'regex_advanced_search',
                                'parameters': {
                                    'search_pattern': pattern,
                                    'file_pattern': file_pattern_str,
                                    'case_sensitive': False,
                                    'whole_word': False,
                                    'include_context': True,
                                    'context_lines': 2
                                }
                            }
                elif current_tool_call['name'] == 'grep':
                    search_pattern = current_tool_call['parameters'].get('search_pattern', '')
                    if search_pattern:
                        query = search_pattern.replace('|', ' ').replace('.*', ' ').replace('(', '').replace(')', '')
                        fallback_tool = {
                            'name': 'semantic_search',
                            'parameters': {
                                'query': query,
                                'max_results': 10,
                                'search_mode': 'comprehensive'
                            }
                        }
                elif current_tool_call['name'] == 'regex_advanced_search':
                    search_pattern = current_tool_call['parameters'].get('search_pattern', '')
                    if search_pattern:
                        query = search_pattern.replace('|', ' ').replace('.*', ' ').replace('(', '').replace(')', '')
                        fallback_tool = {
                            'name': 'embed_search',
                            'parameters': {
                                'query': query,
                                'max_results': 10
                            }
                        }
                elif current_tool_call['name'] == 'file_type_search':
                    search_pattern = current_tool_call['parameters'].get('search_pattern', '')
                    file_extensions = current_tool_call['parameters'].get('file_extensions', [])
                    if search_pattern:
                        query = search_pattern.replace('|', ' ').replace('.*', ' ').replace('(', '').replace(')', '')
                        file_pattern = None
                        if file_extensions:
                            file_pattern = ",".join([f"*{ext}" for ext in file_extensions])

                        fallback_tool = {
                            'name': 'grep',
                            'parameters': {
                                'search_pattern': search_pattern,
                                'file_pattern': file_pattern
                            }
                        }

                if fallback_tool:
                    self.logger.info(f"Using fallback tool: {fallback_tool['name']} with parameters: {fallback_tool['parameters']}")

                    fallback_display = f"[{Style.BRIGHT}{fallback_tool['name']}{Style.NORMAL}]"
                    fallback_param_display = ""
                    for param_name, param_value in fallback_tool['parameters'].items():
                        if isinstance(param_value, str):
                            if len(param_value) > 40:
                                param_value_str = f'"{param_value[:37]}..."'
                            else:
                                param_value_str = f'"{param_value}"'
                        else:
                            param_value_str = str(param_value)
                        fallback_param_display += f" [{param_name}: {param_value_str}]"

                    print(f"\n{Fore.YELLOW}[FALLBACK] {fallback_display}{fallback_param_display}{Style.RESET_ALL}", end='')

                    fallback_result = self._execute_tool(fallback_tool)

                    gathered_information_for_this_query.append({
                        "type": "tool_execution",
                        "tool_name": fallback_tool['name'],
                        "parameters": fallback_tool['parameters'],
                        "result": fallback_result
                    })

                    if isinstance(fallback_result, dict) and "file_path" in fallback_result and not "error" in fallback_result:
                        self.known_files.add(fallback_result["file_path"])
                    elif fallback_tool['name'] in ['embed_search', 'semantic_search', 'grep', 'regex_advanced_search', 'file_type_search'] and isinstance(fallback_result, list):
                        for item_res in fallback_result:
                            if isinstance(item_res, dict) and "file_path" in item_res and not "error" in item_res:
                                self.known_files.add(item_res["file_path"])

                    if "error" in fallback_result:
                        print(f"\r{Fore.RED}[FALLBACK] {fallback_display}{fallback_param_display} ✗ {fallback_result['error']}{Style.RESET_ALL}")
                    else:
                        result_count = len(fallback_result) if isinstance(fallback_result, list) else 1
                        print(f"\r{Fore.GREEN}[FALLBACK] {fallback_display}{fallback_param_display} ✓ Found {result_count} results{Style.RESET_ALL}")

            result_summary = ""

            if isinstance(tool_result, dict) and "error" in tool_result:
                error_msg = tool_result['error']
                if len(error_msg) > 40:
                    error_msg = error_msg[:37] + "..."
                result_summary = f"Error: {error_msg}"
                print(f"\r{Fore.RED}{tool_display}{param_display} ✗ {result_summary}{Style.RESET_ALL}")
            else:
                if current_tool_call['name'] == 'grep':
                    if isinstance(tool_result, list):
                        file_counts = {}
                        for item in tool_result:
                            file_path = item.get("file_path", "unknown")
                            file_counts[file_path] = file_counts.get(file_path, 0) + 1
                        result_summary = f"Found {len(tool_result)} matches in {len(file_counts)} files"

                elif current_tool_call['name'] == 'read_file':
                    if isinstance(tool_result, dict):
                        file_path = tool_result.get("file_path", "unknown")
                        total_lines = tool_result.get("total_lines", 0)
                        line_start = tool_result.get("line_start", 1)
                        line_end = tool_result.get("line_end", total_lines)

                        if line_start > 1 or line_end < total_lines:
                            result_summary = f"Read {file_path} (lines {line_start}-{line_end} of {total_lines})"

                            next_chunk = tool_result.get("next_chunk")
                            if next_chunk:
                                next_start = next_chunk.get("next_start")
                                next_end = next_chunk.get("next_end")
                                remaining = next_chunk.get("remaining_lines")
                                if next_start and next_end and remaining:
                                    print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                                    print(f"{Fore.CYAN}  Suggestion: Read next chunk (lines {next_start}-{next_end}), {remaining} lines remaining{Style.RESET_ALL}")
                                    result_summary = ""
                        else:
                            result_summary = f"Read {file_path} ({total_lines} lines)"

                            chunk_suggestion = tool_result.get("chunk_suggestion")
                            if chunk_suggestion and total_lines > 200:
                                print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                                print(f"{Fore.CYAN}  Suggestion: {chunk_suggestion.get('suggestion')}{Style.RESET_ALL}")
                                result_summary = ""

                elif current_tool_call['name'] == 'directory_tree':
                    if isinstance(tool_result, dict):
                        file_count = tool_result.get("file_count", 0)
                        dir_count = tool_result.get("dir_count", 0)
                        result_summary = f"Tree: {file_count} files, {dir_count} dirs"

                elif current_tool_call['name'] in ['find_functions', 'find_classes']:
                    if isinstance(tool_result, list):
                        result_summary = f"Found {len(tool_result)} results"

                elif current_tool_call['name'] in ['embed_search', 'semantic_search']:
                    if isinstance(tool_result, list):
                        result_summary = f"Found {len(tool_result)} semantic matches"

                elif current_tool_call['name'] == 'regex_advanced_search':
                    if isinstance(tool_result, list):
                        file_counts = {}
                        for item in tool_result:
                            if "summary" in item:
                                continue
                            file_path = item.get("file_path", "unknown")
                            file_counts[file_path] = file_counts.get(file_path, 0) + 1
                        result_summary = f"Found {len(tool_result) - 1 if 'summary' in tool_result[0] else len(tool_result)} matches in {len(file_counts)} files"

                elif current_tool_call['name'] == 'file_type_search':
                    if isinstance(tool_result, list) and len(tool_result) > 0 and "summary" in tool_result[0]:
                        summary = tool_result[0]["summary"]
                        total_matches = summary.get("total_matches", 0)
                        files_with_matches = summary.get("files_with_matches", 0)
                        extensions_searched = summary.get("extensions_searched", [])
                        result_summary = f"Found {total_matches} matches in {files_with_matches} files across {len(extensions_searched)} file types"
                    elif isinstance(tool_result, list):
                        result_summary = f"Found {len(tool_result)} matches"

                elif current_tool_call['name'] == 'cross_reference':
                    if isinstance(tool_result, dict):
                        symbol = current_tool_call['parameters'].get('symbol', '')
                        definitions_count = len(tool_result.get("definitions", []))
                        usages_count = len(tool_result.get("usages", []))
                        imports_count = len(tool_result.get("imports", []))
                        inheritance_count = len(tool_result.get("inheritance", []))
                        related_count = len(tool_result.get("related_symbols", []))

                        result_summary = f"Cross-reference for '{symbol}': {definitions_count} definitions, {usages_count} usages"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        if imports_count > 0:
                            print(f"{Fore.CYAN}  Imports: {imports_count}{Style.RESET_ALL}")
                        if inheritance_count > 0:
                            print(f"{Fore.CYAN}  Inheritance relationships: {inheritance_count}{Style.RESET_ALL}")
                        if related_count > 0:
                            print(f"{Fore.CYAN}  Related symbols: {related_count}{Style.RESET_ALL}")

                        result_summary = ""

                elif current_tool_call['name'] == 'version_control_search':
                    if isinstance(tool_result, dict):
                        search_pattern = current_tool_call['parameters'].get('search_pattern', '')
                        search_type = current_tool_call['parameters'].get('search_type', 'commit_message')
                        total_results = tool_result.get("total_results", 0)

                        type_display = {
                            "commit_message": "commit messages",
                            "code_change": "code changes",
                            "file_path": "file paths"
                        }.get(search_type, search_type)

                        result_summary = f"Found {total_results} matches in {type_display} for '{search_pattern}'"

                elif current_tool_call['name'] == 'get_instructions':
                    if isinstance(tool_result, dict):
                        section = current_tool_call['parameters'].get('section')
                        if "error" in tool_result:
                            result_summary = f"Error getting instructions: {tool_result.get('error')}"
                        elif "message" in tool_result and "No instructions file found" in tool_result.get("message", ""):
                            result_summary = "No instructions file found"
                        else:
                            if section:
                                result_summary = f"Retrieved instructions for section '{section}'"
                            else:
                                sections = tool_result.get("sections", [])
                                result_summary = f"Retrieved all instructions with {len(sections)} sections"

                elif current_tool_call['name'] == 'create_instructions_template':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error creating instructions template: {tool_result.get('error')}"
                        elif "message" in tool_result:
                            result_summary = tool_result.get("message")
                        else:
                            result_summary = "Created instructions template"

                elif current_tool_call['name'] == 'add_memory':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error adding memory: {tool_result.get('error')}"
                        else:
                            category = tool_result.get("category", "general")
                            result_summary = f"Memory added to category '{category}'"

                elif current_tool_call['name'] == 'get_memories':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error getting memories: {tool_result.get('error')}"
                        elif "message" in tool_result and "No memories found" in tool_result.get("message", ""):
                            result_summary = tool_result.get("message")
                        else:
                            count = tool_result.get("count", 0)
                            category = tool_result.get("category", "all")
                            result_summary = f"Retrieved {count} memories from category '{category}'"

                elif current_tool_call['name'] == 'search_memories':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error searching memories: {tool_result.get('error')}"
                        elif "message" in tool_result and "No memories found" in tool_result.get("message", ""):
                            result_summary = tool_result.get("message")
                        else:
                            count = tool_result.get("count", 0)
                            query = tool_result.get("query", "")
                            result_summary = f"Found {count} memories matching '{query}'"

                elif current_tool_call['name'] == 'get_functions':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error getting functions: {tool_result.get('error')}"
                        else:
                            count = tool_result.get("count", 0)
                            file_path = tool_result.get("file_path", "")
                            result_summary = f"Found {count} functions in {file_path}"

                elif current_tool_call['name'] == 'get_classes':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error getting classes: {tool_result.get('error')}"
                        else:
                            count = tool_result.get("count", 0)
                            file_path = tool_result.get("file_path", "")
                            result_summary = f"Found {count} classes in {file_path}"

                elif current_tool_call['name'] == 'get_variables':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error getting variables: {tool_result.get('error')}"
                        else:
                            count = tool_result.get("count", 0)
                            file_path = tool_result.get("file_path", "")
                            result_summary = f"Found {count} variables in {file_path}"

                elif current_tool_call['name'] == 'get_imports':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error getting imports: {tool_result.get('error')}"
                        else:
                            count = tool_result.get("count", 0)
                            file_path = tool_result.get("file_path", "")
                            result_summary = f"Found {count} imports in {file_path}"

                elif current_tool_call['name'] == 'run_command':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error running command: {tool_result.get('error')}"
                        elif "message" in tool_result:
                            result_summary = tool_result.get("message")
                        else:
                            command = tool_result.get("command", "")
                            exit_code = tool_result.get("exit_code")
                            if exit_code is not None:
                                result_summary = f"Command '{command}' completed with exit code {exit_code}"
                            else:
                                terminal_id = tool_result.get("terminal_id")
                                result_summary = f"Command '{command}' running in terminal {terminal_id}"

                elif current_tool_call['name'] == 'read_terminal':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error reading terminal: {tool_result.get('error')}"
                        else:
                            terminal_id = tool_result.get("terminal_id")
                            is_running = tool_result.get("is_running", False)
                            if is_running:
                                result_summary = f"Read output from terminal {terminal_id} (still running)"
                            else:
                                exit_code = tool_result.get("exit_code")
                                result_summary = f"Read output from terminal {terminal_id} (completed with exit code {exit_code})"

                elif current_tool_call['name'] == 'kill_terminal':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error killing terminal: {tool_result.get('error')}"
                        else:
                            terminal_id = tool_result.get("terminal_id")
                            success = tool_result.get("success", False)
                            if success:
                                result_summary = f"Successfully killed terminal {terminal_id}"
                            else:
                                result_summary = f"Failed to kill terminal {terminal_id}"

                elif current_tool_call['name'] == 'list_terminals':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error listing terminals: {tool_result.get('error')}"
                        else:
                            count = tool_result.get("count", 0)
                            result_summary = f"Found {count} active terminal sessions"

                elif current_tool_call['name'] == 'file_stats':
                    if isinstance(tool_result, dict):
                        file_path = tool_result.get("file_path", "unknown")
                        line_count = tool_result.get("line_count", 0)
                        size = tool_result.get("size_human", "0 B")
                        result_summary = f"Stats for {file_path}: {line_count} lines, {size}"

                elif current_tool_call['name'] == 'get_project_description':
                    if isinstance(tool_result, dict):
                        project_name = tool_result.get("project_name", "unknown")
                        purpose = tool_result.get("purpose", "unknown")
                        languages = tool_result.get("languages", [])
                        frameworks = tool_result.get("frameworks", [])

                        languages_str = ", ".join(languages) if languages else "unknown"
                        frameworks_str = ", ".join(frameworks) if frameworks else "unknown"

                        file_count = tool_result.get("file_count", 0)
                        dir_count = tool_result.get("directory_count", 0)

                        result_summary = f"Project: {project_name} - {purpose}"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}  Languages: {languages_str}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}  Frameworks: {frameworks_str}{Style.RESET_ALL}")
                        if file_count > 0:
                            print(f"{Fore.CYAN}  Files: {file_count} in {dir_count} directories{Style.RESET_ALL}")

                        result_summary = ""

                elif current_tool_call['name'] == 'ask_buddy':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            response = tool_result.get("response", "")
                            provider = tool_result.get("provider", "unknown")
                            model = tool_result.get("model", "unknown")
                            context_included = tool_result.get("context_included", False)
                            context_file = tool_result.get("context_file")

                            if len(response) > 50:
                                response_preview = response[:47] + "..."
                            else:
                                response_preview = response

                            context_info = ""
                            if context_included:
                                if context_file:
                                    context_info = f" (with context from {context_file})"
                                else:
                                    context_info = " (with project context)"
                            result_summary = f"Buddy ({provider}/{model}){context_info} responded: {response_preview}"
                elif current_tool_call['name'] in ['google_search', 'ddg_search', 'bing_news_search']:
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            query = tool_result.get("query", "")
                            count = tool_result.get("count", 0)
                            result_summary = f"Found {count} results for query: {query}"
                elif current_tool_call['name'] == 'fetch_webpage':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            url = tool_result.get("url", "")
                            title = tool_result.get("title", "")
                            truncated = tool_result.get("truncated", False)
                            returned_length = tool_result.get("returned_length", 0)
                            result_summary = f"Fetched {returned_length} chars from {url} - '{title}'{' (truncated)' if truncated else ''}"
                elif current_tool_call['name'] == 'get_base_knowledge':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            date = tool_result.get("formatted_date", "")
                            time = tool_result.get("formatted_time", "")
                            day = tool_result.get("day_of_week", "")
                            location = tool_result.get("user_location", "")
                            result_summary = f"Current info: {day}, {date} {time} in {location}"

                elif current_tool_call['name'] == 'get_file_description':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            description = tool_result.get("description", "")
                            if len(description) > 50:
                                description_preview = description[:47] + "..."
                            else:
                                description_preview = description
                            result_summary = f"Description: {description_preview}"

                elif current_tool_call['name'] == 'get_file_metadata':
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            result_summary = f"Error: {tool_result['error']}"
                        else:
                            file_name = tool_result.get("name", "unknown")
                            file_path = tool_result.get("path", "unknown")
                            file_extension = tool_result.get("extension", "unknown")
                            result_summary = f"Metadata for {file_name} ({file_extension})"

                if current_tool_call['name'] == 'ask_buddy':
                    if isinstance(tool_result, dict) and "response" in tool_result:
                        response = tool_result.get("response", "")
                        provider = tool_result.get("provider", "unknown")
                        model = tool_result.get("model", "unknown")

                        if len(response) > 50:
                            response_preview = response[:47] + "..."
                        else:
                            response_preview = response

                        context_included = tool_result.get("context_included", False)
                        context_file = tool_result.get("context_file")

                        context_info = ""
                        if context_included:
                            if context_file:
                                context_info = f" (with context from {context_file})"
                            else:
                                context_info = " (with project context)"

                        result_summary = f"Buddy ({provider}/{model}){context_info} responded: {response_preview}"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Buddy's full response:{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
                        result_summary = ""

                elif current_tool_call['name'] in ['google_search', 'ddg_search', 'bing_news_search']:
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        query = tool_result.get("query", "")
                        results = tool_result.get("results", [])
                        count = tool_result.get("count", 0)

                        result_summary = f"Found {count} results for query: {query}"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Search results for: {query}{Style.RESET_ALL}")

                        for i, result in enumerate(results, 1):
                            if current_tool_call['name'] == 'google_search':
                                print(f"{Fore.YELLOW}{i}. {result.get('title', 'No title')}{Style.RESET_ALL}")
                                print(f"{Fore.BLUE}{result.get('url', 'No URL')}{Style.RESET_ALL}")
                                print(f"{Fore.WHITE}{result.get('description', 'No description')}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}{i}.{Style.RESET_ALL}")
                                for key, value in result.items():
                                    print(f"{Fore.CYAN}{key.capitalize()}: {Style.RESET_ALL}{value}")
                            print()

                        formatted_search_results = []
                        for i, result in enumerate(results, 1):
                            if current_tool_call['name'] == 'google_search':
                                formatted_result = f"{i}. {result.get('title', 'No title')}\n{result.get('url', 'No URL')}\n{result.get('description', 'No description')}"
                            else:
                                formatted_result = f"{i}.\n" + "\n".join([f"{key.capitalize()}: {value}" for key, value in result.items()])
                            formatted_search_results.append(formatted_result)

                        tool_result["formatted_results"] = formatted_search_results
                        tool_result["detailed_summary"] = f"Search results for '{query}':\n\n" + "\n\n".join(formatted_search_results)

                        result_summary = ""

                elif current_tool_call['name'] == 'fetch_webpage':
                    if isinstance(tool_result, dict) and "content" in tool_result:
                        url = tool_result.get("url", "")
                        title = tool_result.get("title", "")
                        content = tool_result.get("content", "")
                        truncated = tool_result.get("truncated", False)
                        returned_length = tool_result.get("returned_length", 0)

                        result_summary = f"Fetched {returned_length} chars from {url} - '{title}'{' (truncated)' if truncated else ''}"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Title: {title}{Style.RESET_ALL}")
                        print(f"{Fore.BLUE}URL: {url}{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Content preview (first 200 chars):{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}{content[:200]}...{Style.RESET_ALL}")

                        result_summary = ""

                elif current_tool_call['name'] == 'get_base_knowledge':
                    if isinstance(tool_result, dict) and "todays_date" in tool_result:
                        date = tool_result.get("formatted_date", "")
                        time = tool_result.get("formatted_time", "")
                        day = tool_result.get("day_of_week", "")
                        location = tool_result.get("user_location", "")
                        timezone = tool_result.get("user_time_zone", "")

                        result_summary = f"Current info: {day}, {date} {time} in {location}"

                        print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Date: {date}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Time: {time}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Day: {day}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Location: {location}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Timezone: {timezone}{Style.RESET_ALL}")

                        result_summary = ""

                else:
                    result_str = str(tool_result)
                    if len(result_str) > 40:
                        result_str = result_str[:37] + "..."
                    result_summary = result_str

                print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")

            evaluator_response_str = await self._evaluate_step_and_get_next_action(
                query, dir_tree_context, gathered_information_for_this_query,
                current_tool_call, tool_result
            )

            if self._check_task_complete(evaluator_response_str):
                self.logger.info("LLM indicated task is complete.")
                break

            current_tool_call = self._extract_tool_call(evaluator_response_str)
            if not current_tool_call:
                self.logger.warning("StepEvaluator did not suggest a next tool and did not mark task as complete. Moving to synthesize.")
                break

            if iteration == max_tool_iterations - 1:
                self.logger.warning("Reached maximum tool iterations.")

        if not final_answer_displayed:
            final_llm_response_str = await self._synthesize_final_answer(
                query, dir_tree_context, gathered_information_for_this_query
            )

            formatted_response = self._format_response(final_llm_response_str)

            print(f"\n{Fore.BLUE}{Style.BRIGHT}✧ FINAL ANSWER ✧{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")

            final_answer_displayed = True

        if not formatted_response.strip():
            thinking_content = self._extract_thinking(final_llm_response_str)
            if thinking_content:
                formatted_response = f"(Synthesizer thought: {thinking_content})\nI seem to have gathered information but couldn't form a final answer. Please check my thought process."
            else:
                formatted_response = "I processed your query but could not generate a final answer. There might have been an issue in my reasoning process."

        lines = formatted_response.split('\n')
        in_code_block = False
        code_block_content = []

        for line in lines:
            if line.strip() == '```' or line.strip().startswith('```'):
                in_code_block = not in_code_block

                if in_code_block:
                    if line.strip() != '```':
                        lang = line.strip().replace('```', '')
                        print(f"{Fore.CYAN}```{lang}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.CYAN}```{Style.RESET_ALL}")
                    code_block_content = []
                else:
                    code_block_text = "\n".join(code_block_content)
                    if code_block_text.strip().endswith("..."):
                        lines_before_block = lines[:lines.index(line)]
                        for prev_line in reversed(lines_before_block[-5:]):
                            if any(known_file in prev_line for known_file in self.known_files):
                                for known_file in self.known_files:
                                    if known_file in prev_line:
                                        try:
                                            result = self.tools.read_file(known_file)
                                            if "content" in result and not "error" in result:
                                                full_content = result["content"].split('\n')
                                                code_block_content = full_content
                                                break
                                        except Exception as e:
                                            self.logger.error(f"Error reading file {known_file}: {e}")
                                        break

                    for code_line in code_block_content:
                        print(f"{Fore.YELLOW}{code_line}{Style.RESET_ALL}")

                    print(f"{Fore.CYAN}```{Style.RESET_ALL}")
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            line = re.sub(r'\*\*(.*?)\*\*', f'{Fore.WHITE}{Style.BRIGHT}\\1{Style.NORMAL}{Style.RESET_ALL}', line)
            if line.startswith('# '):
                print(f"{Fore.WHITE}{Style.BRIGHT}{line}{Style.RESET_ALL}")
            elif line.startswith('## '):
                print(f"{Fore.WHITE}{Style.BRIGHT}{line}{Style.RESET_ALL}")
            elif line.startswith('### '):
                print(f"{Fore.WHITE}{Style.BRIGHT}{line}{Style.RESET_ALL}")
            else:
                print(line)

        print(f"{Fore.BLUE}{'─' * 80}{Style.RESET_ALL}")

        self.add_to_history("assistant", formatted_response)
        return formatted_response

    def _format_response(self, raw_response: str) -> str:
        """Format the AI's response for better readability.

        Args:
            raw_response (str): The raw response text

        Returns:
            str: Formatted response text
        """
        patterns_to_remove = [
            r'<thinking_stage.*?>.*?</thinking_stage>',
            r'<thinking>.*?</thinking>',
            r'<tool_call_request>.*?</tool_call_request>',
            r'<tool_executed>.*?</tool_executed>',
            r'<tool_result>.*?</tool_result>',
            r'<task_complete>.*?</task_complete>',
        ]

        clean_response = raw_response
        for pattern in patterns_to_remove:
            clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL)

        clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)

        for file_path in self.known_files:
            if file_path in clean_response and not f"`{file_path}`" in clean_response:
                clean_response = clean_response.replace(file_path, f"`{file_path}`")

        code_blocks = re.findall(r'```(.*?)```', clean_response, re.DOTALL)
        for block in code_blocks:
            if '...' in block and block.strip().endswith('...'):
                lines_before_block = clean_response.split('```' + block)[0].split('\n')
                for line in reversed(lines_before_block[-5:]):
                    if any(known_file in line for known_file in self.known_files):
                        for known_file in self.known_files:
                            if known_file in line:
                                try:
                                    result = self.tools.read_file(known_file)
                                    if "content" in result and not "error" in result:
                                        full_content = result["content"]
                                        clean_response = clean_response.replace(
                                            '```' + block + '```',
                                            '```\n' + full_content + '\n```'
                                        )
                                        break
                                except Exception as e:
                                    self.logger.error(f"Error reading file {known_file}: {e}")
                                break

        return clean_response.strip()
