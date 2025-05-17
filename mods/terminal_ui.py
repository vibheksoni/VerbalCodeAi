"""Terminal UI components for VerbalCodeAI.

This module provides UI components and utilities for the terminal interface,
including streaming response handling, thinking animations, and other UI elements.
"""

import asyncio
import sys
import time
from typing import AsyncGenerator, Callable, Optional

from colorama import Fore, Style, init

from .terminal_utils import (
    get_terminal_size,
    parse_thinking_blocks,
    render_markdown,
    render_thinking_blocks,
)

init()


class StreamingResponseHandler:
    """Handler for streaming AI responses in the terminal."""

    def __init__(
        self,
        enable_markdown_rendering: bool = True,
        show_thinking_blocks: bool = False,
        logger=None,
    ):
        """Initialize the streaming response handler.

        Args:
            enable_markdown_rendering (bool): Whether to render markdown.
            show_thinking_blocks (bool): Whether to show thinking blocks.
            logger (Logger, optional): Logger instance.
        """
        self.enable_markdown_rendering = enable_markdown_rendering
        self.show_thinking_blocks = show_thinking_blocks
        self.logger = logger

    async def process_stream(
        self,
        response: AsyncGenerator[str, None],
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process a streaming response.

        Args:
            response (AsyncGenerator[str, None]): The streaming response generator.
            on_complete (Optional[Callable[[str], None]]): Callback function to call with the complete response when streaming is done.

        Returns:
            str: The complete response.
        """
        try:
            in_thinking_block = False
            thinking_buffer = ""
            thinking_dots = 0
            thinking_animation_timer = 0
            full_response = []
            thinking_start_time = None

            async for chunk in response:
                full_response.append(chunk)

                if not self.show_thinking_blocks:
                    thinking_buffer += chunk

                    if len(thinking_buffer) > 20:
                        thinking_buffer = thinking_buffer[-20:]

                    if not in_thinking_block and "<think>" in thinking_buffer:
                        in_thinking_block = True
                        thinking_start_time = time.time()
                        terminal_width, _ = get_terminal_size()
                        sys.stdout.write("\r" + " " * terminal_width)
                        sys.stdout.write(
                            f"\r{Fore.YELLOW}Thinking {Style.BRIGHT}●{Style.RESET_ALL}"
                        )
                        sys.stdout.flush()
                        thinking_dots = 1
                        thinking_animation_timer = 0
                        continue

                    if in_thinking_block and "</think>" in thinking_buffer:
                        in_thinking_block = False

                        try:
                            full_text = "".join(full_response)
                            last_think_start = full_text.rfind("<think>")
                            last_think_end = full_text.rfind("</think>")

                            thinking_time_str = ""
                            if thinking_start_time is not None:
                                thinking_time = time.time() - thinking_start_time
                                thinking_time_str = f" in {thinking_time:.2f} seconds"
                                thinking_start_time = None

                            terminal_width, _ = get_terminal_size()
                            sys.stdout.write("\r" + " " * terminal_width)

                            sys.stdout.write(
                                f"\r{Fore.GREEN}Thinking Completed!{Style.RESET_ALL}\n"
                            )
                            sys.stdout.flush()

                            await asyncio.sleep(0.2)

                            thinking_tokens = None
                            if (
                                last_think_start != -1
                                and last_think_end != -1
                                and last_think_start < last_think_end
                            ):
                                thinking_content = full_text[last_think_start + len("<think>") : last_think_end]
                                thinking_tokens = len(thinking_content.split())

                                sys.stdout.write(
                                    f"{Fore.YELLOW}[STAT] AI thought for approximately {thinking_tokens} tokens{thinking_time_str}.{Style.RESET_ALL}\n"
                                )
                                sys.stdout.flush()
                            else:
                                sys.stdout.write(
                                    f"{Fore.YELLOW}[STAT] AI thinking process completed{thinking_time_str}. (Could not determine token count){Style.RESET_ALL}\n"
                                )
                                sys.stdout.flush()
                                if self.logger:
                                    self.logger.warning(
                                        f"Could not determine thinking tokens. last_think_start: {last_think_start}, "
                                        f"last_think_end: {last_think_end}. Check stream for <think>/</think> tags. "
                                        f"Sample full_text (first 100 chars): '{full_text[:100]}'"
                                    )

                        except Exception as e:
                            if self.logger:
                                self.logger.warning(
                                    f"Error processing thinking blocks: {e}"
                                )
                            terminal_width, _ = get_terminal_size()
                            sys.stdout.write("\r" + " " * terminal_width)
                            sys.stdout.write(
                                f"\r{Fore.GREEN}Thinking Completed!{Style.RESET_ALL}\n"
                            )
                            sys.stdout.flush()
                            await asyncio.sleep(0.2)
                            thinking_time_str_fallback = ""
                            if thinking_start_time is not None:
                                thinking_time_fb = time.time() - thinking_start_time
                                thinking_time_str_fallback = f" in {thinking_time_fb:.2f} seconds"
                                thinking_start_time = None
                            sys.stdout.write(
                                f"{Fore.YELLOW}[STAT] AI thinking process completed{thinking_time_str_fallback}.{Style.RESET_ALL}\n"
                            )
                            sys.stdout.flush()
                        
                        continue

                    if in_thinking_block:
                        thinking_animation_timer += 1
                        if thinking_animation_timer % 5 == 0:
                            thinking_dots = (thinking_dots % 5) + 1
                            terminal_width, _ = get_terminal_size()
                            sys.stdout.write("\r" + " " * terminal_width)
                            sys.stdout.write(
                                f"\r{Fore.YELLOW}Thinking {Style.BRIGHT}{'●' * thinking_dots}{Style.RESET_ALL}"
                            )
                            sys.stdout.flush()
                        continue

                if not in_thinking_block or self.show_thinking_blocks:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

            if full_response and not "".join(full_response).endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()

            complete_response = "".join(full_response)

            if on_complete:
                on_complete(complete_response)

            return complete_response

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in streaming process: {e}")
            print(f"\n{Fore.RED}Error in streaming: {e}{Style.RESET_ALL}")
            return "".join(full_response) if full_response else ""


def display_response(
    response: str,
    enable_markdown_rendering: bool = True,
    show_thinking_blocks: bool = False,
) -> None:
    """Display an AI response in the terminal.

    Args:
        response (str): The AI response to display.
        enable_markdown_rendering (bool): Whether to render markdown.
        show_thinking_blocks (bool): Whether to show thinking blocks.
    """
    terminal_width, _ = get_terminal_size()

    cleaned_response, thinking_text, thinking_tokens = parse_thinking_blocks(response)

    if enable_markdown_rendering:
        rendered_response = render_markdown(
            cleaned_response, width=terminal_width - 2
        )
    else:
        rendered_response = cleaned_response

    if thinking_tokens:
        if show_thinking_blocks and thinking_text:
            thinking_rendered = render_thinking_blocks(
                thinking_text, width=terminal_width - 2
            )
            print(thinking_rendered)
        else:
            print(
                f"{Fore.YELLOW}AI thought for approximately {thinking_tokens} tokens before responding.{Style.RESET_ALL}"
            )

    print(rendered_response)
