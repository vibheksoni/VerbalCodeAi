import os
import sys
import time
from typing import List

from colorama import Fore, Style, init

init()


def display_animated_banner(
    frames: List[str] = None,
    frame_delay: float = 0.5,
    color: str = Fore.CYAN,
    clear_screen: bool = True,
) -> None:
    """Display an animated ASCII art banner.

    Args:
        frames (List[str], optional): List of ASCII art frames to display.
            Defaults to ANIMATION_FRAMES.
        frame_delay (float, optional): Delay between frames in seconds.
            Defaults to 0.5.
        color (str, optional): Color to use for the banner.
            Defaults to Fore.CYAN.
        clear_screen (bool, optional): Whether to clear the screen before displaying.
            Defaults to True.
    """
    if frames is None:
        frames = ANIMATION_FRAMES

    if clear_screen:
        os.system("cls" if os.name == "nt" else "clear")

    for frame in frames:
        sys.stdout.write("\033[H")
        sys.stdout.write(f"{color}{Style.BRIGHT}{frame}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(frame_delay)

    print()


VERBAL_CODE_AI = r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
"""

ANIMATION_FRAMES = [
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [⣾]
    > Initializing AI engine...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [⣽]
    > Initializing AI engine...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [⣻]
    > Initializing AI engine...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [⢿]
    > Initializing AI engine...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [⡿]
    > Initializing AI engine...
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [DONE]
    > Initializing AI engine... [DONE]
    """,
    r"""
 _    __          __          __   ______          __        ___    _
| |  / /__  _____/ /_  ____ _/ /  / ____/___  ____/ /__     /   |  (_)
| | / / _ \/ ___/ __ \/ __ `/ /  / /   / __ \/ __  / _ \   / /| | / /
| |/ /  __/ /  / /_/ / /_/ / /  / /___/ /_/ / /_/ /  __/  / ___ |/ /
|___/\___/_/  /_.___/\__,_/_/   \____/\____/\__,_/\___/  /_/  |_/_/
    > AI Assistant for Code
    > Analyzing your codebase... [DONE]
    > Initializing AI engine... [DONE]
    > Ready to help with your questions!
    """,
]