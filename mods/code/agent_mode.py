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

    SYSTEM_PROMPT = """You are an AI assistant with access to tools that can help you explore and understand a codebase.
Your primary goal is to answer the user's questions about the codebase thoroughly and accurately.

You will proceed in stages:
1.  **Planning:** Understand the query and determine initial information needs and tools.
2.  **Information Gathering:** Iteratively use tools, evaluating results at each step to decide the next action.
3.  **Synthesis:** Once sufficient information is gathered, formulate a comprehensive answer.

During the Information Gathering stage, you should:
- Be thorough in your exploration, using multiple tools when necessary
- Adapt your approach based on what you discover
- Continue gathering information until you have a complete understanding
- Consider alternative approaches if initial tools don't yield useful results
- Prioritize depth of understanding over breadth when appropriate

You will be provided with the current directory structure. Use this to inform your tool usage, especially for file paths.
Available tools:

1.  file_stats(path: str) - Get statistics about a file including line count, size, and other metadata.
    - path: Path to the file (can be imprecise, partial, or full path)
    - Returns: Dictionary with file statistics including full path, line count, size, etc.
    - Use this FIRST before reading a file to understand its size and structure
    - Example: file_stats("main.py") or file_stats("src/utils.js")

2.  read_file(path: str, line_start: int = None, line_end: int = None) - Read contents of a specific file.
    - path: Path to the file (can be imprecise, partial, or full path)
    - line_start: Optional starting line number (1-based, inclusive)
    - line_end: Optional ending line number (1-based, inclusive)
    - Returns: File content as string (full file or specified lines)
    - For large files, first use file_stats to get the line count, then read in chunks of 100-200 lines
    - Example: read_file("main.py") or read_file("utils.js", 10, 20)

3.  embed_search(query: str, max_results: int = 5) - Search the codebase using vector embeddings.
    - query: The search query string (natural language or code)
    - max_results: Maximum number of results to return (default: 5)
    - Returns: List of results containing file paths and matching code snippets
    - Best for concept-based queries and finding relevant code snippets
    - Example: embed_search("user authentication") or embed_search("database connection", 10)

4.  grep(search_pattern: str, file_pattern: str = None) - Search the codebase using regex patterns.
    - search_pattern: The regex pattern to search for
    - file_pattern: Optional filter for specific file types (e.g., "*.py", "*.js")
    - Returns: List of matches with file paths and line numbers
    - Best for exact text matches and finding specific strings
    - Example: grep("def process_data", "*.py") or grep("useState\\(")

5.  directory_tree(max_depth: int = None) - Generate/refresh the directory structure.
    - max_depth: Optional maximum depth to traverse (default: unlimited)
    - Returns: Dictionary with directory tree as string and statistics
    - Use to understand project structure or refresh the directory context
    - Example: directory_tree(3)

6.  find_functions(pattern: str, file_pattern: str = None) - Find function definitions matching a pattern.
    - pattern: Regex pattern to match function names
    - file_pattern: Optional filter for specific file types
    - Returns: List of function definitions with file paths and line numbers
    - Example: find_functions("process_.*", "*.py")

7.  find_classes(pattern: str, file_pattern: str = None) - Find class definitions matching a pattern.
    - pattern: Regex pattern to match class names
    - file_pattern: Optional filter for specific file types
    - Returns: List of class definitions with file paths and line numbers
    - Example: find_classes("User.*", "*.py")

8.  git_history(path: str, max_commits: int = 10) - Get git history for a file or directory.
    - path: Path to the file or directory
    - max_commits: Maximum number of commits to return (default: 10)
    - Returns: List of commits with author, date, and message
    - Example: git_history("src/main.py", 5)

9.  search_imports(module_name: str, file_pattern: str = None) - Find where modules are imported.
    - module_name: Name of the module to search for
    - file_pattern: Optional filter for specific file types
    - Returns: List of import statements with file paths and line numbers
    - Example: search_imports("os", "*.py")

10. find_usage(symbol: str, file_pattern: str = None) - Find where a function, class, or variable is used.
    - symbol: The symbol name to search for
    - file_pattern: Optional filter for specific file types
    - Returns: List of usages with file paths and line numbers
    - Example: find_usage("process_data", "*.py")

11. code_analysis(path: str) - Analyze code structure and dependencies.
    - path: Path to the file to analyze
    - Returns: Dictionary with code structure information
    - Example: code_analysis("src/main.py")

12. explain_code(path: str, line_start: int = None, line_end: int = None) - Generate an explanation of a code snippet.
    - path: Path to the file
    - line_start: Optional starting line number
    - line_end: Optional ending line number
    - Returns: Dictionary with code explanation
    - Example: explain_code("src/main.py", 10, 20)

13. get_project_description() - Get a description of the project.
    - Returns: Dictionary with project description, structure, and other relevant information
    - Use this to understand the overall project structure and purpose
    - Example: get_project_description()

14. ask_buddy(question: str) - Ask the buddy AI model for opinions or suggestions.
    - question: The question or request to ask the buddy AI
    - Returns: Dictionary with the buddy's response
    - Use this when you need a second opinion or help with a specific problem
    - Example: ask_buddy("What's the best way to implement authentication in this project?")

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
You are in the **Planning Stage**.
User Query: {user_query}
Directory Structure:
```
{directory_tree_context}
```
Your task is to analyze the user's query and the provided directory structure.
Determine the key pieces of information needed to answer the query and suggest the first one or two tools to gather this information.
If a file path is needed for `read_file`, ensure it is relative to the project root shown in the directory structure.

Consider which tools would be most effective for the user's query:
1. file_stats - To get statistics about a file (use FIRST before reading a file to understand its size and structure)
2. read_file - To read specific files (when you know which file to examine, use file_stats first for large files)
3. embed_search - For semantic search across the codebase (best for concept-based queries)
4. grep - For pattern-based search (best for exact text matches)
5. directory_tree - To explore the directory structure (useful for understanding project organization)
6. find_functions - To find function definitions (when looking for specific functionality)
7. find_classes - To find class definitions (when looking for object-oriented structures)
8. git_history - To check file history (useful for understanding changes over time)
9. search_imports - To find where modules are imported (helpful for dependency analysis)
10. find_usage - To find where symbols are used (good for understanding how components interact)
11. code_analysis - To analyze code structure (provides detailed insights into a file's composition)
12. explain_code - To get an explanation of code (useful for understanding complex snippets)
13. get_project_description - To get an overview of the project (useful for understanding the project structure and purpose)
14. ask_buddy - To get a second opinion from another AI model (useful when you need help with a specific problem)

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
You are in the **Information Gathering Stage**.
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

Your task is to evaluate the tool result in the context of the user's query and the information gathered so far.

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
4. grep - For pattern-based search
5. directory_tree - To explore the directory structure
6. find_functions - To find function definitions
7. find_classes - To find class definitions
8. git_history - To check file history
9. search_imports - To find where modules are imported
10. find_usage - To find where symbols are used
11. code_analysis - To analyze code structure
12. explain_code - To get an explanation of code
13. get_project_description - To get an overview of the project
14. ask_buddy - To get a second opinion from another AI model

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
    "question": "What's the best way to implement a REST API in this project based on the code I've seen so far?"
  }}
}}
</tool_call_request>

Example if task complete:
<thinking>I have read 'auth.py' and it contains the login function the user asked about. I also found the related middleware in 'middleware.py'. I now have sufficient information to explain how authentication works in this codebase.</thinking>
<task_complete>true</task_complete>
</thinking_stage>
"""

    FINAL_ANSWER_SYNTHESIZER_PROMPT_TEMPLATE = """<thinking_stage name="FinalAnswerSynthesizer">
You are in the **Synthesis Stage**.
User Query: {user_query}
Directory Structure:
```
{directory_tree_context}
```
All relevant information gathered for this query:
{all_gathered_info_summary}

Your task is to synthesize a comprehensive, well-formatted (using Markdown where appropriate) final answer to the user's query based *only* on the information provided above.

Follow these guidelines for creating a high-quality response:
1. Structure your answer logically with clear sections and headings where appropriate
2. Use code blocks with syntax highlighting for code examples
3. Provide file paths and line numbers when referencing specific code
4. Explain complex concepts in simple terms
5. Highlight important patterns, relationships, or architectural decisions
6. If relevant, explain how different components interact
7. Include a brief summary at the beginning for complex answers

Important:
- Do NOT call any tools. Do NOT make up information.
- If the gathered information is insufficient to answer the query, explicitly state that.
- Cite specific files and code snippets from the gathered information.
- Use proper formatting to make the answer readable (markdown, code blocks, etc.)
- For code explanations, focus on the most relevant parts rather than showing everything.

Enclose your thought process in <thinking> tags, followed by your final answer.

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
                        'embed_search', 'grep', 'read_file', 'directory_tree',
                        'find_functions', 'find_classes', 'git_history', 'search_imports',
                        'find_usage', 'code_analysis', 'explain_code', 'file_stats',
                        'get_project_description', 'ask_buddy'
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
            elif tool_name == 'grep':
                search_pattern = parameters.get('search_pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.grep(search_pattern, file_pattern)

                for item in result:
                    if "file_path" in item:
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
                    result = self.tools.ask_buddy(question)
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

            if (current_tool_call['name'] == 'embed_search' and isinstance(tool_result, list) and len(tool_result) == 0) or \
               (current_tool_call['name'] == 'grep' and isinstance(tool_result, list) and len(tool_result) == 0):
                self.logger.warning(f"No results found for {current_tool_call['name']}. Trying fallback strategy.")

                fallback_tool = None

                language_info = self.tools.get_project_languages()
                file_patterns = []

                if "error" not in language_info and language_info.get("languages"):
                    languages = language_info.get("languages", [])
                    extensions = language_info.get("extensions", {})

                    for lang in languages[:3]:
                        if lang in extensions and extensions[lang]:
                            for ext in extensions[lang]:
                                file_patterns.append(f"*{ext}")

                if not file_patterns:
                    file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.c", "*.cpp", "*.cs", "*.go", "*.rb", "*.php"]

                file_pattern_str = ",".join(file_patterns)

                if current_tool_call['name'] == 'embed_search':
                    search_query = current_tool_call['parameters'].get('query', '')
                    if search_query:
                        terms = [term for term in search_query.split() if len(term) > 3]
                        if terms:
                            pattern = '|'.join(terms)
                            fallback_tool = {
                                'name': 'grep',
                                'parameters': {
                                    'search_pattern': pattern,
                                    'file_pattern': file_pattern_str
                                }
                            }
                elif current_tool_call['name'] == 'grep':
                    search_pattern = current_tool_call['parameters'].get('search_pattern', '')
                    if search_pattern:
                        query = search_pattern.replace('|', ' ').replace('.*', ' ').replace('(', '').replace(')', '')
                        fallback_tool = {
                            'name': 'embed_search',
                            'parameters': {
                                'query': query,
                                'max_results': 5
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
                    elif fallback_tool['name'] in ['embed_search', 'grep'] and isinstance(fallback_result, list):
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

                elif current_tool_call['name'] == 'embed_search':
                    if isinstance(tool_result, list):
                        result_summary = f"Found {len(tool_result)} semantic matches"

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

                            if len(response) > 50:
                                response_preview = response[:47] + "..."
                            else:
                                response_preview = response

                            result_summary = f"Buddy ({provider}/{model}) responded: {response_preview}"

                            print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ {result_summary}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}Buddy's full response:{Style.RESET_ALL}")
                            print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
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
