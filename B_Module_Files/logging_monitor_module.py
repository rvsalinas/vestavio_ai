"""
logging_monitor_module.py
Absolute File Path:
  /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/logging_detection_module.py

PURPOSE:
  - Configures and monitors application logs.
  - Uses a RotatingFileHandler for log rotation.
  - Optionally sends email alerts on CRITICAL logs via SMTP.

NOTES:
  - If environment variables for SMTP are set (SMTP_SERVER, SMTP_USER, etc.),
    and `enable_email_logs=True`, then a CRITICAL log triggers an email alert.
  - If these environment variables are missing, email alerts are skipped.
"""

import logging
import os
from logging.handlers import RotatingFileHandler, SMTPHandler
from typing import Optional


class LoggingMonitor:
    """
    A class for configuring and monitoring logs:
      - Rotating file handler for normal logs.
      - Optional email alerts for CRITICAL logs (if SMTP env vars are set).
    """

    def __init__(
        self,
        log_file: str = "F_Log_Files/application.log",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
        enable_email_logs: bool = False
    ):
        """
        :param log_file: Name/path of the log file.
        :param max_bytes: Max size in bytes before rotating.
        :param backup_count: Number of backups to keep.
        :param enable_email_logs: If True, configures SMTPHandler for CRITICAL logs.
        """
        self.logger = logging.getLogger("LoggingMonitor")
        self.logger.setLevel(logging.INFO)

        # 1. Rotating file handler
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 2. Optional email alerts for CRITICAL logs
        if enable_email_logs:
            smtp_server = os.getenv("SMTP_SERVER", "")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            from_email = os.getenv("FROM_EMAIL", "")
            alert_email = os.getenv("ALERT_EMAIL", "")
            smtp_user = os.getenv("SMTP_USER", "")
            smtp_password = os.getenv("SMTP_PASSWORD", "")

            # Only configure if all required SMTP vars exist
            if all([smtp_server, smtp_port, from_email, alert_email, smtp_user, smtp_password]):
                email_handler = SMTPHandler(
                    mailhost=(smtp_server, smtp_port),
                    fromaddr=from_email,
                    toaddrs=[alert_email],
                    subject="CRITICAL Log Alert",
                    credentials=(smtp_user, smtp_password),
                    secure=()
                )
                email_handler.setLevel(logging.CRITICAL)
                email_formatter = logging.Formatter(
                    "Time: %(asctime)s\nLevel: %(levelname)s\nMessage: %(message)s\n"
                )
                email_handler.setFormatter(email_formatter)
                self.logger.addHandler(email_handler)
            else:
                self.logger.warning(
                    "Email logging is enabled but SMTP environment variables are missing. "
                    "Skipping email alerts."
                )

    def log_info(self, message: str):
        """Log an info-level message."""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log a warning-level message."""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log an error-level message."""
        self.logger.error(message)

    def log_critical(self, message: str):
        """Log a critical-level message. Triggers email if SMTP is configured."""
        self.logger.critical(message)


if __name__ == "__main__":
    # Example usage
    monitor = LoggingMonitor(enable_email_logs=False)
    monitor.log_info("This is an info log.")
    monitor.log_warning("This is a warning log.")
    monitor.log_error("This is an error log.")
    # Will not send an email unless SMTP env vars are set and enable_email_logs=True
    monitor.log_critical("Critical issue detected!")