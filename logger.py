import logging
import os


def get_logger():
    rank = int(os.getenv("LOCAL_RANK", 0))

    # Set logging level: INFO for rank 0, ERROR for others
    level = logging.INFO if rank == 0 else logging.ERROR

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler with custom format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s - [Rank {rank}] - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = get_logger()
    logger.info("Hello, world!")