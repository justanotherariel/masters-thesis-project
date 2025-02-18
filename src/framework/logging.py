"""A logger that logs to the terminal and to W&B."""

import logging
import os
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import wandb

_LOGGER: Optional["Logger"] = None


class Logger:
    """A logger that logs to the terminal and to W&B."""

    def __new__(cls, log_path: Optional[Path] = None):
        global _LOGGER
        if _LOGGER is None:
            _LOGGER = super().__new__(cls)
            _LOGGER._init(log_path)
        return _LOGGER

    def _init(self, log_path: Optional[Path]) -> None:
        """Initialize the logger."""
        # Setup Logger Class
        self.logger = logging.getLogger("main")
        self.logger.setLevel(logging.DEBUG)

        # Handle KeyboardInterrupts
        def handler(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: TracebackType | None,
        ) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.logger.error(
                "A wild %s appeared!",
                exc_type.__name__,
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = handler

    def __getattr__(self, name: str) -> Any:
        """Pass the attribute request to the logger."""
        return getattr(self.logger, name)

    def log_section_separator(self, message: str) -> None:
        """Print a section separator.

        :param message: title of the section
        """
        try:
            separator_length = min(os.get_terminal_size().columns, 120)
        except OSError:
            separator_length = 120
        separator_char = "="
        title_char = " "
        separator = separator_char * separator_length
        title_padding = (separator_length - len(message)) // 2
        centered_title = (
            f"{title_char * title_padding}{message}{title_char * title_padding}"
            if len(message) % 2 == 0
            else f"{title_char * title_padding}{message}{title_char * (title_padding + 1)}"
        )

        self.info("\n%s\n%s\n%s\n", separator, centered_title, separator)

    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        """Log a message to an external service.

        :param message: The message to log
        :param kwargs: Any additional arguments
        """
        if wandb.run:
            if message.get("type") == "wandb_plot" and message["plot_type"] == "line_series":
                plot_data = message["data"]
                # Construct the plot here using the provided data
                plot = wandb.plot.line_series(
                    xs=plot_data["xs"],
                    ys=plot_data["ys"],
                    keys=plot_data["keys"],
                    title=plot_data["title"],
                    xname=plot_data["xname"],
                )
                wandb.log({plot_data["title"]: plot}, **kwargs)
            else:
                wandb.log(message, **kwargs)
