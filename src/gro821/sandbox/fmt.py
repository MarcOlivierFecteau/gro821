from typing import Final


class ANSI:
    """Useful ANSI escape codes."""

    RESET: Final = "\x1b[0m"
    BOLD: Final = "\x1b[1m"
    DIM: Final = "\x1b[2m"
    ITALIC: Final = "\x1b[3m"
    UNDERLINE: Final = "\x1b[4m"
    STRIKE: Final = "\x1b[9m"

    BLACK: Final = "\x1b[30m"
    RED: Final = "\x1b[31m"
    GREEN: Final = "\x1b[32m"
    YELLOW: Final = "\x1b[33m"
    BLUE: Final = "\x1b[34m"
    MAGENTA: Final = "\x1b[35m"
    CYAN: Final = "\x1b[36m"
    WHITE: Final = "\x1b[37m"
    DEFAULT: Final = "\x1b[39m"

    BLACK_BG: Final = "\x1b[40m"
    RED_BG: Final = "\x1b[41m"
    GREEN_BG: Final = "\x1b[42m"
    YELLOW_BG: Final = "\x1b[43m"
    BLUE_BG: Final = "\x1b[44m"
    MAGENTA_BG: Final = "\x1b[45m"
    CYAN_BG: Final = "\x1b[46m"
    WHITE_BG: Final = "\x1b[47m"
    DEFAULT_BG: Final = "\x1b[49m"

    BRIGHT_BLACK: Final = "\x1b[90m"
    BRIGHT_RED: Final = "\x1b[91m"
    BRIGHT_GREEN: Final = "\x1b[92m"
    BRIGHT_YELLOW: Final = "\x1b[93m"
    BRIGHT_BLUE: Final = "\x1b[94m"
    BRIGHT_MAGENTA: Final = "\x1b[95m"
    BRIGHT_CYAN: Final = "\x1b[96m"
    BRIGHT_WHITE: Final = "\x1b[97m"

    BRIGHT_BLACK_BG: Final = "\x1b[100m"
    BRIGHT_RED_BG: Final = "\x1b[101m"
    BRIGHT_GREEN_BG: Final = "\x1b[102m"
    BRIGHT_YELLOW_BG: Final = "\x1b[103m"
    BRIGHT_BLUE_BG: Final = "\x1b[104m"
    BRIGHT_MAGENTA_BG: Final = "\x1b[105m"
    BRIGHT_CYAN_BG: Final = "\x1b[106m"
    BRIGHT_WHITE_BG: Final = "\x1b[107m"


def highlight(text: str) -> str:
    return f"{ANSI.YELLOW_BG}{text}{ANSI.RESET}"


hl: Final = highlight


def bold(text: str) -> str:
    return f"{ANSI.BOLD}{text}{ANSI.RESET}"


b: Final = bold


def italic(text: str) -> str:
    return f"{ANSI.ITALIC}{text}{ANSI.RESET}"


it: Final = italic


def newline() -> None:
    print()


LF: Final = newline


def pprint_bool(condition: bool, text: tuple[str, str] | None = None) -> str:
    """`text`: (true_text, false_text)."""
    if text is None:
        text = ("Yes", "No")
    if condition:
        return f"{ANSI.BRIGHT_GREEN}{text[0]}{ANSI.RESET}"
    return f"{ANSI.RED}{text[1]}{ANSI.RESET}"
