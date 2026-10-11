import logging
import logging.handlers as handlers


class RadQALLMEvalLogger:
    def __init__(
        self,
        app_name: str,
        log_file_path: str,
        level=logging.INFO,
        max_bytes=500000,
        backup_count=5,
    ):
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        log_handler = handlers.RotatingFileHandler(
            filename=log_file_path, maxBytes=max_bytes, backupCount=backup_count
        )
        log_handler.setLevel(level)
        log_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(log_handler)

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        return self.logger
