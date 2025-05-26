"""Prompt Enhancer module for VerbalCodeAI.

This module provides a specialized AI agent that analyzes the codebase and enhances
user prompts with relevant context, code snippets, and technical details to create
more comprehensive and context-aware prompts.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from colorama import Fore, Style

from ..llms import generate_response
from .tools import CodebaseTools

logger = logging.getLogger("VerbalCodeAI.PromptEnhancer")


class PromptEnhancer:
    """Prompt Enhancer for VerbalCodeAI.

    This class implements a specialized AI agent that analyzes the codebase and
    enhances user prompts with relevant context, code snippets, and technical details.
    """

    SYSTEM_PROMPT = """You are an expert prompt enhancement specialist with deep knowledge of software development and codebase analysis. Your task is to transform basic user prompts into comprehensive, context-aware prompts that include relevant technical details from the analyzed codebase.

You have access to powerful tools that can help you explore and understand any codebase with perfect accuracy. Your goal is to enhance the user's original prompt by:

1. **Understanding the Intent**: Analyze the user's original prompt to understand their goals and requirements
2. **Gathering Context**: Use available tools to collect relevant information from the codebase
3. **Enhancing the Prompt**: Transform the original prompt into a comprehensive, detailed specification

## Enhancement Process:

### Stage 1: Analysis
- Understand what the user wants to accomplish
- Identify key concepts, technologies, or components mentioned
- Determine what type of codebase context would be most relevant

### Stage 2: Context Gathering
- Use tools to explore the codebase structure and patterns
- Find relevant files, functions, classes, and configurations
- Understand the project's architecture, dependencies, and conventions
- Look for existing implementations that relate to the user's request

### Stage 3: Prompt Enhancement
- Expand the original prompt with specific technical context
- Include relevant code snippets and file references
- Add implementation guidance based on existing patterns
- Provide technical constraints and considerations
- Maintain the original intent while making it more comprehensive

## Available Tools:

You have access to the same tools as the Agent Mode:
- **Search tools**: embed_search, semantic_search, grep, regex_advanced_search, file_type_search
- **File tools**: read_file, file_stats, directory_tree, get_file_description, get_file_metadata
- **Code analysis**: find_functions, find_classes, find_usage, cross_reference, code_analysis, get_functions, get_classes, get_variables, get_imports, explain_code
- **Project tools**: get_project_description, get_instructions
- **Version control**: git_history, version_control_search, search_imports

## Output Format:

Your final enhanced prompt should be formatted as well-structured Markdown with the following sections:

```markdown
# Enhanced Prompt: [Brief Title]

## Original Request
[User's original prompt]

## Task Description
[Expanded and clarified description of what needs to be accomplished]

## Technical Context
[Relevant information about the codebase, architecture, and existing patterns]

## Relevant Code References
[File paths, function names, class names, and code snippets that are relevant]

## Implementation Guidance
[Specific guidance based on existing code patterns and project conventions]

## Technical Constraints
[Any constraints, dependencies, or considerations based on the codebase analysis]

## Detailed Specifications
[Comprehensive specifications for implementation, including specific requirements]
```

When you need to use a tool, format the request within <tool_call_request> XML tags:
<tool_call_request>
{
  "name": "tool_name",
  "parameters": {
    "param1": "value1"
  }
}
</tool_call_request>

If you have gathered sufficient information to create the enhanced prompt, respond with <task_complete>true</task_complete>.
Your thought process should be enclosed in <thinking>...</thinking> tags.
"""

    PLANNER_PROMPT_TEMPLATE = """<thinking_stage name="PromptAnalyzer">
You are analyzing a user's prompt to enhance it with relevant codebase context.

User's Original Prompt: {user_prompt}
Directory Structure:
```
{directory_tree_context}
```

Your task is to:
1. Understand what the user wants to accomplish
2. Identify what type of codebase information would be most relevant
3. Plan which tools to use to gather the necessary context

Analyze the user's prompt and determine:
- What are they trying to build/implement/modify?
- What technologies, frameworks, or patterns might be involved?
- What parts of the codebase would be most relevant?
- What existing code patterns should inform the enhancement?

Start by getting a high-level understanding of the project, then drill down into specific areas that relate to the user's request.

Respond with your analysis in <thinking> tags, followed by the first <tool_call_request> to begin gathering context.
</thinking_stage>
"""

    CONTEXT_GATHERER_PROMPT_TEMPLATE = """<thinking_stage name="ContextGatherer">
You are gathering context from the codebase to enhance the user's prompt.

User's Original Prompt: {user_prompt}
Directory Structure:
```
{directory_tree_context}
```

Information gathered so far:
{gathered_info_summary}

Last tool call:
<tool_executed>
{last_tool_call_json}
</tool_executed>
Tool result:
<tool_result>
{last_tool_result_json}
</tool_result>

Evaluate the information you've gathered and decide:
1. Is this information relevant to the user's request?
2. Do you need more specific details about certain components?
3. Are there other areas of the codebase you should explore?
4. Do you have enough context to create a comprehensive enhanced prompt?

If you need more information, provide the next <tool_call_request>.
If you have sufficient context, respond with <task_complete>true</task_complete>.
</thinking_stage>
"""

    PROMPT_SYNTHESIZER_TEMPLATE = """<thinking_stage name="PromptSynthesizer">
You are creating the final enhanced prompt based on all gathered information.

User's Original Prompt: {user_prompt}
Directory Structure:
```
{directory_tree_context}
```

All gathered information:
{all_gathered_info_summary}

Create a comprehensive, well-formatted Markdown enhanced prompt that includes:

1. **Original Request**: The user's original prompt
2. **Task Description**: Expanded and clarified description
3. **Technical Context**: Relevant codebase information, architecture, patterns
4. **Relevant Code References**: Specific files, functions, classes with brief descriptions
5. **Implementation Guidance**: Specific guidance based on existing patterns
6. **Technical Constraints**: Dependencies, constraints, considerations
7. **Detailed Specifications**: Comprehensive implementation requirements

Make sure to:
- Include specific file paths and code references from your analysis
- Reference existing patterns and conventions found in the codebase
- Provide actionable, detailed guidance
- Maintain the original intent while making it much more comprehensive
- Use proper Markdown formatting for readability

Your response should be the complete enhanced prompt in Markdown format.
</thinking_stage>
"""

    def __init__(self, indexer=None):
        """Initialize the PromptEnhancer with an indexer.

        Args:
            indexer (FileIndexer, optional): The FileIndexer instance used to access the indexed codebase.
        """
        self.indexer = indexer
        self.tools = CodebaseTools(indexer)
        self.logger = logging.getLogger("VerbalCodeAI.PromptEnhancer")
        self.last_directory_tree_run_time = 0
        self.directory_tree_cache = None

    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call from the AI's response."""
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
                        'get_project_description', 'get_file_description',
                        'get_file_metadata', 'get_instructions',
                        'get_functions', 'get_classes', 'get_variables', 'get_imports'
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
        """Extract content within <thinking>...</thinking> tags."""
        match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _check_task_complete(self, response: str) -> bool:
        """Check for <task_complete>true</task_complete> in the response."""
        return bool(re.search(r"<task_complete>\s*true\s*</task_complete>", response, re.IGNORECASE))

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        tool_name = tool_call['name']
        parameters = tool_call.get('parameters', {})

        self.logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")

        try:
            if tool_name == 'embed_search':
                query = parameters.get('query', '')
                max_results = parameters.get('max_results', 5)
                result = self.tools.embed_search(query, max_results)
            elif tool_name == 'semantic_search':
                query = parameters.get('query', '')
                max_results = parameters.get('max_results', 5)
                search_mode = parameters.get('search_mode', 'comprehensive')
                result = self.tools.semantic_search(query, max_results, search_mode)
            elif tool_name == 'grep':
                search_pattern = parameters.get('search_pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.grep(search_pattern, file_pattern)
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
            elif tool_name == 'file_type_search':
                search_pattern = parameters.get('search_pattern', '')
                file_extensions = parameters.get('file_extensions', [])
                case_sensitive = parameters.get('case_sensitive', False)
                max_results = parameters.get('max_results', 100)
                result = self.tools.file_type_search(
                    search_pattern, file_extensions, case_sensitive, max_results
                )
            elif tool_name == 'read_file':
                path = parameters.get('path', '')
                line_start = parameters.get('line_start')
                line_end = parameters.get('line_end')
                result = self.tools.read_file(path, line_start, line_end)
            elif tool_name == 'directory_tree':
                max_depth = parameters.get('max_depth')
                result = self.tools.directory_tree(max_depth)
                self.last_directory_tree_run_time = time.time()
                self.directory_tree_cache = result
            elif tool_name == 'find_functions':
                pattern = parameters.get('pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_functions(pattern, file_pattern)
            elif tool_name == 'find_classes':
                pattern = parameters.get('pattern', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_classes(pattern, file_pattern)
            elif tool_name == 'find_usage':
                symbol = parameters.get('symbol', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.find_usage(symbol, file_pattern)
            elif tool_name == 'cross_reference':
                symbol = parameters.get('symbol', '')
                reference_type = parameters.get('reference_type', 'all')
                max_results = parameters.get('max_results', 20)
                result = self.tools.cross_reference(symbol, reference_type, max_results)
            elif tool_name == 'git_history':
                path = parameters.get('path', '')
                max_commits = parameters.get('max_commits', 10)
                result = self.tools.git_history(path, max_commits)
            elif tool_name == 'version_control_search':
                search_pattern = parameters.get('search_pattern', '')
                search_type = parameters.get('search_type', 'commit_message')
                max_results = parameters.get('max_results', 20)
                author = parameters.get('author')
                date_range = parameters.get('date_range')
                result = self.tools.version_control_search(
                    search_pattern, search_type, max_results, author, date_range
                )
            elif tool_name == 'search_imports':
                module_name = parameters.get('module_name', '')
                file_pattern = parameters.get('file_pattern')
                result = self.tools.search_imports(module_name, file_pattern)
            elif tool_name == 'code_analysis':
                path = parameters.get('path', '')
                result = self.tools.code_analysis(path)
            elif tool_name == 'explain_code':
                path = parameters.get('path', '')
                line_start = parameters.get('line_start')
                line_end = parameters.get('line_end')
                result = self.tools.explain_code(path, line_start, line_end)
            elif tool_name == 'file_stats':
                path = parameters.get('path', '')
                result = self.tools.file_stats(path)
            elif tool_name == 'get_project_description':
                result = self.tools.get_project_description()
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
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            result = {"error": f"Error executing tool {tool_name}: {str(e)}"}

        return result

    def _get_directory_tree_context(self, force_refresh: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Get the directory tree, either from cache or by executing the tool."""
        should_run_directory_tree = (
            force_refresh or
            time.time() - self.last_directory_tree_run_time > 300 or
            not self.directory_tree_cache
        )

        if should_run_directory_tree:
            print(f"\n{Fore.GREEN}Fetching directory structure for context...{Style.RESET_ALL}")
            dir_tree_tool_call = {"name": "directory_tree", "parameters": {"max_depth": 3}}
            dir_tree_result = self._execute_tool(dir_tree_tool_call)
            if "error" not in dir_tree_result:
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
        """Prepare a concise summary of gathered information for the LLM."""
        if not gathered_info:
            return "No information gathered yet."

        summary_parts = []
        for idx, item in enumerate(gathered_info):
            part = f"Item {idx+1}:\n"
            if item.get("type") == "tool_execution":
                tool_name = item.get('tool_name', '')
                part += f"  Tool: {tool_name}\n"
                part += f"  Parameters: {json.dumps(item.get('parameters'))}\n"

                result_summary = json.dumps(item.get('result', {}))
                if len(result_summary) > 1000 and tool_name != 'directory_tree':
                    result_summary = result_summary[:997] + "..."
                part += f"  Result: {result_summary}\n"
            else:
                part += f"  {json.dumps(item)}"
            summary_parts.append(part)
        return "\n".join(summary_parts)

    async def _call_llm_for_stage(self, prompt_template: str, context_vars: Dict[str, Any], stage_name: str) -> str:
        """Generic LLM call function for a stage."""
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

    async def enhance_prompt(self, user_prompt: str) -> str:
        """Enhance a user prompt with relevant codebase context.

        Args:
            user_prompt (str): The original user prompt to enhance.

        Returns:
            str: The enhanced prompt with codebase context.
        """
        if not self.indexer:
            return "Error: No code has been indexed yet. Please index a directory first."

        self.logger.info(f"Enhancing prompt: {user_prompt}")

        gathered_information: List[Dict[str, Any]] = []

        dir_tree_context, dir_tree_raw_result = self._get_directory_tree_context()
        if "error" in dir_tree_raw_result.get("tree", "").lower() or not dir_tree_context.strip():
            dir_tree_context, dir_tree_raw_result = self._get_directory_tree_context(force_refresh=True)
            if "error" in dir_tree_raw_result.get("tree", "").lower():
                error_msg = "Critical Error: Directory tree is unavailable. Cannot proceed."
                self.logger.error(error_msg)
                return error_msg

        planner_response_str = await self._call_llm_for_stage(
            self.PLANNER_PROMPT_TEMPLATE,
            {"user_prompt": user_prompt, "directory_tree_context": dir_tree_context},
            "PromptAnalyzer"
        )

        current_tool_call = self._extract_tool_call(planner_response_str)

        if not current_tool_call:
            self.logger.warning("Planner did not suggest an initial tool. Moving to synthesis.")
        else:
            max_tool_iterations = 8
            for iteration in range(max_tool_iterations):
                if not current_tool_call:
                    self.logger.info("No more tools to call. Moving to synthesis.")
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

                tool_result = self._execute_tool(current_tool_call)

                gathered_information.append({
                    "type": "tool_execution",
                    "tool_name": current_tool_call['name'],
                    "parameters": current_tool_call['parameters'],
                    "result": tool_result
                })

                if isinstance(tool_result, dict) and "error" in tool_result:
                    error_msg = tool_result['error']
                    if len(error_msg) > 40:
                        error_msg = error_msg[:37] + "..."
                    print(f"\r{Fore.RED}{tool_display}{param_display} ✗ {error_msg}{Style.RESET_ALL}")
                else:
                    result_count = len(tool_result) if isinstance(tool_result, list) else 1
                    print(f"\r{Fore.GREEN}{tool_display}{param_display} ✓ Found {result_count} results{Style.RESET_ALL}")

                evaluator_response_str = await self._call_llm_for_stage(
                    self.CONTEXT_GATHERER_PROMPT_TEMPLATE,
                    {
                        "user_prompt": user_prompt,
                        "directory_tree_context": dir_tree_context,
                        "gathered_info_summary": self._prepare_gathered_info_summary(gathered_information),
                        "last_tool_call_json": json.dumps(current_tool_call),
                        "last_tool_result_json": json.dumps(tool_result)
                    },
                    "ContextGatherer"
                )

                if self._check_task_complete(evaluator_response_str):
                    self.logger.info("LLM indicated context gathering is complete.")
                    break

                current_tool_call = self._extract_tool_call(evaluator_response_str)
                if not current_tool_call:
                    self.logger.warning("ContextGatherer did not suggest a next tool. Moving to synthesis.")
                    break

                if iteration == max_tool_iterations - 1:
                    self.logger.warning("Reached maximum tool iterations.")

        final_llm_response_str = await self._call_llm_for_stage(
            self.PROMPT_SYNTHESIZER_TEMPLATE,
            {
                "user_prompt": user_prompt,
                "directory_tree_context": dir_tree_context,
                "all_gathered_info_summary": self._prepare_gathered_info_summary(gathered_information)
            },
            "PromptSynthesizer"
        )

        enhanced_prompt = self._format_enhanced_prompt(final_llm_response_str)

        return enhanced_prompt

    def _format_enhanced_prompt(self, raw_response: str) -> str:
        """Format the AI's response to extract the enhanced prompt."""
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

        return clean_response.strip() 