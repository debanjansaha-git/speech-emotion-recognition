import os
import sys
import logging

logs_dir = "logs"
log_path = os.path.join(logs_dir, "running_logs.log")
os.makedirs(logs_dir, exist_ok=True)

# Set the logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
