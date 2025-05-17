"""Terminal utilities for VerbalCodeAI.

This module provides utilities for rendering text in the terminal,
including markdown rendering and AI response processing.
"""

import os
import platform
import re
import sys
import textwrap
import time
from typing import Callable, Optional, Tuple

from colorama import Back, Fore, Style, init

init()


def clear_screen() -> None:
    """Clear the terminal screen and scrollback buffer."""
    if platform.system() == "Windows":
        os.system("cls")
        sys.stdout.write("\033[2J\033[3J\033[H")
        sys.stdout.flush()
    else:
        os.system("clear")


def parse_thinking_blocks(response: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Extract and remove thinking blocks from AI responses.

    Args:
        response (str): The AI response text.

    Returns:
        Tuple[str, Optional[str], Optional[int]]:
            - The cleaned response.
            - The thinking blocks text (if any).
            - The number of tokens in thinking blocks (if any).
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
        return response, None, None

    thinking_text = "\n\n".join(matches)
    thinking_tokens = len(thinking_text.split())

    cleaned_response = thinking_pattern.sub("", response)

    alt_patterns = [
        r"<thinking>(.*?)</thinking>",
        r"\[thinking\](.*?)\[/thinking\]",
        r"\{thinking\}(.*?)\{/thinking\}",
    ]
    for pattern in alt_patterns:
        alt_thinking_pattern = re.compile(pattern, re.DOTALL)
        cleaned_response = alt_thinking_pattern.sub("", cleaned_response)

    return cleaned_response.strip(), thinking_text, thinking_tokens


def render_thinking_blocks(thinking_text: str, width: int = 80) -> str:
    """Render thinking blocks with special formatting.

    Args:
        thinking_text (str): The thinking blocks text to render.
        width (int): The maximum width for text wrapping. Defaults to 80.

    Returns:
        str: The rendered thinking blocks with terminal formatting.
    """
    if not thinking_text:
        return ""

    rendered_text = f"\n{Fore.YELLOW}{Style.BRIGHT}AI Thinking Process:{Style.RESET_ALL}\n"
    rendered_text += f"{Fore.YELLOW}{'-' * width}{Style.RESET_ALL}\n"

    wrapped_text = textwrap.fill(
        thinking_text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    rendered_text += f"{Fore.YELLOW}{wrapped_text}{Style.RESET_ALL}\n"
    rendered_text += f"{Fore.YELLOW}{'-' * width}{Style.RESET_ALL}\n"

    return rendered_text


def render_markdown(text: str, width: int = 80) -> str:
    """Render markdown text with colorama formatting.

    Args:
        text (str): The markdown text to render.
        width (int): The maximum width for text wrapping. Defaults to 80.

    Returns:
        str: The rendered text with terminal formatting.
    """
    rendered_text = ""

    bold_pattern = re.compile(r"(\*\*|__)(.*?)(\*\*|__)")
    italic_pattern = re.compile(r"(\*|_)(.*?)(\*|_)")
    strikethrough_pattern = re.compile(r"~~(.*?)~~")
    inline_code_pattern = re.compile(r"`(.*?)`")

    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2)

            header_text_formatted = header_text

            header_text_formatted = bold_pattern.sub(
                f"{Fore.WHITE}{Style.BRIGHT}\\2{Style.NORMAL}{Fore.RESET}",
                header_text_formatted,
            )

            header_text_formatted = italic_pattern.sub(
                f"{Fore.YELLOW}\\2{Fore.RESET}", header_text_formatted
            )

            header_text_formatted = inline_code_pattern.sub(
                f"{Fore.WHITE}{Back.BLACK}\\1{Back.RESET}{Fore.RESET}",
                header_text_formatted,
            )

            if level == 1:
                rendered_text += (
                    f"\n{Fore.CYAN}{Style.BRIGHT}{header_text_formatted}{Style.RESET_ALL}\n"
                )
                rendered_text += (
                    f"{Fore.CYAN}{'=' * min(len(header_text), width)}{Style.RESET_ALL}\n"
                )
            elif level == 2:
                rendered_text += (
                    f"\n{Fore.GREEN}{Style.BRIGHT}{header_text_formatted}{Style.RESET_ALL}\n"
                )
                rendered_text += (
                    f"{Fore.GREEN}{'-' * min(len(header_text), width)}{Style.RESET_ALL}\n"
                )
            elif level == 3:
                rendered_text += (
                    f"\n{Fore.YELLOW}{Style.BRIGHT}{header_text_formatted}{Style.RESET_ALL}\n"
                )
            elif level == 4:
                rendered_text += (
                    f"\n{Fore.MAGENTA}{Style.BRIGHT}{header_text_formatted}{Style.RESET_ALL}\n"
                )
            else:
                rendered_text += (
                    f"\n{Fore.WHITE}{Style.BRIGHT}{header_text_formatted}{Style.RESET_ALL}\n"
                )

            i += 1
            continue

        if line.startswith("```"):
            code_lang = line[3:].strip()
            code_block = []
            i += 1

            while i < len(lines) and not lines[i].startswith("```"):
                code_block.append(lines[i])
                i += 1

            if i < len(lines):
                i += 1

            code_text = "\n".join(code_block)
            rendered_text += f"{Fore.BLACK}{Back.WHITE} {code_lang} {Back.RESET}\n"
            rendered_text += f"{Fore.WHITE}{Back.BLACK}{code_text}{Style.RESET_ALL}\n"
            continue

        line_with_formatting = line

        line_with_formatting = bold_pattern.sub(
            f"{Fore.WHITE}{Style.BRIGHT}\\2{Style.NORMAL}{Fore.RESET}",
            line_with_formatting,
        )

        line_with_formatting = italic_pattern.sub(
            f"{Fore.YELLOW}\\2{Fore.RESET}", line_with_formatting
        )

        line_with_formatting = strikethrough_pattern.sub(
            f"{Fore.RED}{Style.DIM}\\1{Style.NORMAL}{Fore.RESET}",
            line_with_formatting,
        )

        line_with_formatting = inline_code_pattern.sub(
            f"{Fore.WHITE}{Back.BLACK}\\1{Back.RESET}{Fore.RESET}",
            line_with_formatting,
        )

        list_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
        if list_match:
            indent = len(list_match.group(1))
            list_text = list_match.group(2)

            if indent == 0:
                bullet_color = Fore.CYAN
                bullet = "•"
            elif indent <= 2:
                bullet_color = Fore.GREEN
                bullet = "◦"
            else:
                bullet_color = Fore.MAGENTA
                bullet = "▪"

            list_text_formatted = list_text

            list_text_formatted = bold_pattern.sub(
                f"{Fore.WHITE}{Style.BRIGHT}\\2{Style.NORMAL}{Fore.RESET}",
                list_text_formatted,
            )

            list_text_formatted = italic_pattern.sub(
                f"{Fore.YELLOW}\\2{Fore.RESET}", list_text_formatted
            )

            list_text_formatted = inline_code_pattern.sub(
                f"{Fore.WHITE}{Back.BLACK}\\1{Back.RESET}{Fore.RESET}",
                list_text_formatted,
            )

            wrapped_text = textwrap.fill(
                list_text_formatted,
                width=width - indent - 2,
                initial_indent=f"{' ' * indent}{bullet_color}{bullet}{Fore.RESET} ",
                subsequent_indent=f"{' ' * (indent + 2)}",
            )
            rendered_text += f"{wrapped_text}\n"
            i += 1
            continue

        num_list_match = re.match(r"^(\s*)(\d+)\.?\s+(.+)$", line)
        if num_list_match:
            indent = len(num_list_match.group(1))
            num = num_list_match.group(2)
            list_text = num_list_match.group(3)

            list_text_formatted = list_text

            list_text_formatted = bold_pattern.sub(
                f"{Fore.WHITE}{Style.BRIGHT}\\2{Style.NORMAL}{Fore.RESET}",
                list_text_formatted,
            )

            list_text_formatted = italic_pattern.sub(
                f"{Fore.YELLOW}\\2{Fore.RESET}", list_text_formatted
            )

            list_text_formatted = inline_code_pattern.sub(
                f"{Fore.WHITE}{Back.BLACK}\\1{Back.RESET}{Fore.RESET}",
                list_text_formatted,
            )

            wrapped_text = textwrap.fill(
                list_text_formatted,
                width=width - indent - len(num) - 2,
                initial_indent=f"{' ' * indent}{Fore.GREEN}{num}.{Fore.RESET} ",
                subsequent_indent=f"{' ' * (indent + len(num) + 2)}",
            )
            rendered_text += f"{wrapped_text}\n"
            i += 1
            continue

        if re.match(r"^-{3,}$|^_{3,}$|^\*{3,}$", line):
            rendered_text += f"{Fore.CYAN}{'-' * width}{Style.RESET_ALL}\n"
            i += 1
            continue

        if line.strip():
            wrapped_text = textwrap.fill(
                line_with_formatting,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            rendered_text += f"{wrapped_text}\n"
        else:
            rendered_text += "\n"

        i += 1

    return rendered_text


def get_terminal_size() -> Tuple[int, int]:
    """Get the terminal size.

    Returns:
        Tuple[int, int]:
            - The width of the terminal.
            - The height of the terminal.
    """
    try:
        columns, lines = os.get_terminal_size()
        return columns, lines
    except (AttributeError, OSError):
        return 80, 24


def create_spinner() -> Callable[[], None]:
    """Create a spinner animation for terminal output.

    Returns:
        Callable[[], None]: A function that updates the spinner animation.
    """
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = [0]

    def update_spinner() -> None:
        """Update the spinner animation."""
        try:
            terminal_width, _ = get_terminal_size()
        except:
            terminal_width = 80

        sys.stdout.write("\r" + " " * terminal_width)
        sys.stdout.write(f"\r{Fore.CYAN}{spinner_chars[i[0]]}{Style.RESET_ALL} ")
        sys.stdout.flush()
        i[0] = (i[0] + 1) % len(spinner_chars)

    return update_spinner


def stream_text(text: str, delay: float = 0.01) -> None:
    """Stream text to the terminal with a typing effect.

    Args:
        text (str): The text to stream.
        delay (float): The delay between characters in seconds. Defaults to 0.01.
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def create_progress_bar(total: int, width: int = 40) -> Callable[[int, str], None]:
    """Create a progress bar for terminal output.

    Args:
        total (int): The total number of items to process.
        width (int): The width of the progress bar in characters. Defaults to 40.

    Returns:
        Callable[[int, str], None]: A function that updates the progress bar.
    """
    def update_progress(current: int, status: str = "") -> None:
        """Update the progress bar.

        Args:
            current (int): The current number of items processed.
            status (str): The status message to display. Defaults to "".
        """
        percent = min(100, int(current * 100 / total))
        filled_width = int(width * current / total)
        bar = f"{Fore.GREEN}{'█' * filled_width}{Fore.WHITE}{'░' * (width - filled_width)}"

        try:
            terminal_width, _ = get_terminal_size()
        except:
            terminal_width = 80

        sys.stdout.write("\r" + " " * terminal_width)
        sys.stdout.write(f"\r{Fore.CYAN}[{bar}{Fore.CYAN}] {Fore.YELLOW}{percent}%{Fore.RESET} {status}")
        sys.stdout.flush()

        if current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return update_progress
