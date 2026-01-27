"""
Logging utilities.
"""

import os
from torch.utils.tensorboard import SummaryWriter
import logging


def setup_logger(log_dir, name="train"):
    """
    Setup TensorBoard logger and file logger.

    Args:
        log_dir: Directory to save logs
        name: Logger name

    Returns:
        tb_writer: TensorBoard writer
        logger: File logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    # File logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return tb_writer, logger

