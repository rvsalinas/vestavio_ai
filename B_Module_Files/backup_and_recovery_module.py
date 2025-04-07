"""
backup_and_recovery_module.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/backup_and_recovery_module.py

PURPOSE:
  - Facilitates scheduled or on-demand backups of important directories or files.
  - Allows restoration from a specified backup archive.
  - Optionally provides selective restore of specific files.

NOTES:
  - This module can be expanded to store backups on S3 or other remote destinations.
  - For scheduling backups, see the commented code with `schedule`.
  - Ensure you have read/write permission for backup/restore directories.
"""

import os
import logging
import datetime
import tarfile
from typing import List, Optional

# If you want to schedule backups, you can uncomment these lines:
# import schedule
# import time
# from threading import Thread

class BackupAndRecovery:
    """
    A class for handling backup and recovery of critical files or directories.
    Supports creating tar.gz archives, optionally on a schedule, 
    and restoring all or part of the archive.
    """

    def __init__(
        self,
        backup_dir: str = "backups",
        enable_schedule: bool = False,
        schedule_interval_minutes: int = 60
    ):
        """
        :param backup_dir: Directory where backup archives will be stored.
        :param enable_schedule: If True, will schedule backups automatically 
                                (requires calling start_scheduler()).
        :param schedule_interval_minutes: How often (in minutes) backups run if scheduling is enabled.
        """
        self.backup_dir = backup_dir
        self.enable_schedule = enable_schedule
        self.schedule_interval_minutes = schedule_interval_minutes
        os.makedirs(self.backup_dir, exist_ok=True)

        self.logger = logging.getLogger("BackupAndRecovery")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized BackupAndRecovery with backup_dir={self.backup_dir}, "
                         f"enable_schedule={enable_schedule}")

        # If scheduling is desired:
        # if self.enable_schedule:
        #     schedule.every(self.schedule_interval_minutes).minutes.do(self.scheduled_backup_job)

    def backup_paths(
        self, 
        paths_to_backup: List[str],
        backup_name: Optional[str] = None,
        compression: str = "gz"
    ) -> str:
        """
        Create a tar archive (by default .tar.gz) of the specified paths.
        
        :param paths_to_backup: List of file/directory paths to include in the backup.
        :param backup_name: Custom name for the backup file; if omitted, uses a timestamp.
        :param compression: "gz" or "bz2" or none. 
        :return: The path to the created backup archive.
        """
        if backup_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.tar.{compression}"

        backup_path = os.path.join(self.backup_dir, backup_name)

        mode = f"w:{compression}" if compression else "w"
        self.logger.info(f"Creating backup: {backup_path}")
        with tarfile.open(backup_path, mode) as tar:
            for path in paths_to_backup:
                if os.path.exists(path):
                    arcname = os.path.basename(path.rstrip("/"))
                    tar.add(path, arcname=arcname)
                    self.logger.info(f"Added '{path}' to backup.")
                else:
                    self.logger.warning(f"Path not found, skipping: {path}")
        self.logger.info(f"Backup created successfully at {backup_path}")
        return backup_path

    def restore_backup(
        self, 
        backup_file: str, 
        restore_dir: str = ".", 
        files_to_extract: Optional[List[str]] = None
    ):
        """
        Restore from a tar(.gz, .bz2) backup into the specified restore_dir.
        
        :param backup_file: The name of the tar archive inside backup_dir, or full path.
        :param restore_dir: Destination directory where files are extracted.
        :param files_to_extract: If specified, only these files (arc-names) are extracted from the tar.
        """
        if not os.path.isabs(backup_file):
            backup_file = os.path.join(self.backup_dir, backup_file)

        if not os.path.exists(backup_file):
            self.logger.error(f"Backup file not found: {backup_file}")
            return

        os.makedirs(restore_dir, exist_ok=True)
        self.logger.info(f"Restoring backup '{backup_file}' to '{restore_dir}'.")

        # Detect compression automatically:
        try:
            with tarfile.open(backup_file, "r:*") as tar:
                if files_to_extract:
                    for member_name in files_to_extract:
                        member = tar.getmember(member_name)
                        tar.extract(member, path=restore_dir)
                        self.logger.info(f"Extracted '{member_name}' to '{restore_dir}'.")
                else:
                    tar.extractall(path=restore_dir)
                    self.logger.info(f"Extracted all contents to '{restore_dir}'.")
        except (tarfile.TarError, KeyError) as e:
            self.logger.error(f"Error extracting from {backup_file}: {e}", exc_info=True)
        else:
            self.logger.info("Restore completed successfully.")

    # Example method if you want to do scheduled backups automatically:
    # def scheduled_backup_job(self):
    #     paths_to_backup = ["your_project_folder", "some_other_folder"]
    #     try:
    #         self.backup_paths(paths_to_backup)
    #         self.logger.info("Scheduled backup job completed.")
    #     except Exception as e:
    #         self.logger.error(f"Scheduled backup job failed: {e}", exc_info=True)

    # def start_scheduler(self):
    #     def run_scheduler():
    #         while True:
    #             schedule.run_pending()
    #             time.sleep(1)
    #     t = Thread(target=run_scheduler, daemon=True)
    #     t.start()
    #     self.logger.info("Backup scheduler started in the background.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick demonstration:
    br = BackupAndRecovery(backup_dir="backups_demo")

    # Create a backup
    backup_created = br.backup_paths(["A_Core_App_Files", "B_Module_Files"], backup_name="my_backup_demo.tar.gz")

    # # Restore it (optional demonstration)
    # br.restore_backup("my_backup_demo.tar.gz", restore_dir="restored_demo")
    
    # The code above will create 'my_backup_demo.tar.gz' in 'backups_demo' 
    # containing "A_Core_App_Files" and "B_Module_Files". 
    # Adjust as needed.